"""Official YOLOv13 backbone + reconstructed WIO-FPN, CP-Head, Bio-Mamba."""
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from . import ROOT, VENDOR
from .wavelet import WIONeck
from .prototype import PrototypeBank
from .mamba import BioMamba
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors, dist2bbox


def read_config(path):
    with open(path, encoding='utf-8') as file:
        return yaml.safe_load(file)


class CPHead(nn.Module):
    def __init__(self, channels, classes=4, embedding_dim=128, momentum=0.1,
                 temperature=0.1, prototypes=True):
        super().__init__()
        baseline = Detect(nc=classes, ch=tuple(channels))
        self.regression = baseline.cv2
        self.reg_max = baseline.reg_max
        self.classes = classes
        self.use_prototypes = prototypes
        self.embedding = nn.ModuleList(nn.Sequential(nn.Conv2d(c, embedding_dim, 1), nn.ReLU(),
                                                     nn.Conv2d(embedding_dim, embedding_dim, 1)) for c in channels)
        self.background = nn.ModuleList(nn.Conv2d(embedding_dim, 1, 1) for _ in channels)
        self.linear_classes = None if prototypes else nn.ModuleList(nn.Conv2d(embedding_dim, classes, 1) for _ in channels)
        self.bank = PrototypeBank(classes, embedding_dim, momentum, temperature)
        self.register_buffer('stride', torch.tensor([8 * 2 ** i for i in range(len(channels))]).float())

    def forward(self, features):
        embeddings = [layer(x) for layer, x in zip(self.embedding, features)]
        z = torch.cat([x.flatten(2).transpose(1, 2) for x in embeddings], 1)
        regression = torch.cat([layer(x).flatten(2).transpose(1, 2)
                                for layer, x in zip(self.regression, features)], 1)
        bg = torch.cat([layer(x).flatten(2).transpose(1, 2)
                        for layer, x in zip(self.background, embeddings)], 1)
        if self.use_prototypes:
            cls = self.bank.logits(z)
        else:
            cls = torch.cat([layer(x).flatten(2).transpose(1, 2)
                             for layer, x in zip(self.linear_classes, embeddings)], 1)
        return {'regression': regression, 'embeddings': z, 'logits': torch.cat((cls, bg), -1),
                'features': features}

    def decode(self, raw):
        anchors, strides = make_anchors(raw['features'], self.stride, 0.5)
        reg = raw['regression']
        bins = torch.arange(self.reg_max, device=reg.device, dtype=reg.dtype)
        distances = reg.reshape(*reg.shape[:2], 4, self.reg_max).softmax(-1) @ bins
        boxes = dist2bbox(distances, anchors, xywh=False) * strides
        if self.use_prototypes:
            labels, margins = self.bank.confidence(raw['embeddings'])
            foreground = 1 - raw['logits'].float().softmax(-1)[..., -1]
            scores = F.one_hot(labels, self.classes).float() * (margins * foreground)[..., None]
        else:
            scores = raw['logits'].float().softmax(-1)[..., :self.classes]
        return boxes, scores


class SpectralYOLO(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = deepcopy(config)
        cfg = config['model']
        upstream = read_config(VENDOR / 'ultralytics/cfg/models/v13/yolov13.yaml')
        upstream['scale'] = cfg.get('scale', 'l')
        upstream['nc'] = len(config['classes'])
        # Parse official modules directly: avoids constructing a disposable neck
        # and removes the otherwise expensive dummy forward at construction.
        from ultralytics.nn.tasks import parse_model
        upstream['head'] = []
        backbone, _ = parse_model(upstream, ch=3, verbose=False)
        self.backbone = backbone
        self.feature_indices = (4, 6, 8)
        width_factor, max_channels = config_channels(cfg.get('scale', 'l'))
        from ultralytics.utils.ops import make_divisible
        in_channels = [make_divisible(min(c, max_channels) * width_factor, 8) for c in (512, 512, 1024)]
        width = cfg.get('neck_width', 256)
        levels = cfg.get('pyramid_levels', 5)
        self.neck = WIONeck(in_channels, width, levels, cfg.get('wio', True))
        self.head = CPHead([width] * levels, len(config['classes']), cfg.get('embedding_dim', 128),
                           cfg.get('prototype_momentum', 0.1), config['loss'].get('temperature', 0.1),
                           cfg.get('cp_head', True))
        temporal = config['temporal']
        self.forecast_enabled = cfg.get('bio_mamba', True)
        self.forecast = BioMamba(width, temporal.get('hidden_dim', 256), temporal.get('state_dim', 16),
                                  temporal.get('layers', 2), temporal.get('grid', 8),
                                  temporal.get('horizons_weeks', [1, 2, 4]), temporal.get('dropout', 0.1),
                                  temporal.get('backend', 'reference')) if self.forecast_enabled else None

    def features(self, images):
        outputs = []
        x = images
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.feature_indices:
                outputs.append(x)
        return self.neck(outputs)

    def forward(self, images):
        return self.head(self.features(images))

    def forecast_images(self, images, output_size=None, detach_backbone=None):
        if self.forecast is None:
            raise RuntimeError('Bio-Mamba disabled by this ablation configuration')
        if images.ndim != 5:
            raise ValueError('Sequence images must be [B,T,3,H,W]')
        detach = self.config['temporal'].get('detach_backbone', True) if detach_backbone is None else detach_backbone
        if detach:
            # Frozen temporal feature extraction must not update BatchNorm stats.
            was_backbone, was_neck = self.backbone.training, self.neck.training
            self.backbone.eval(); self.neck.eval()
            with torch.no_grad():
                features = torch.stack([self.features(images[:, t])[0] for t in range(images.shape[1])], 1)
            self.backbone.train(was_backbone); self.neck.train(was_neck)
        else:
            features = torch.stack([self.features(images[:, t])[0] for t in range(images.shape[1])], 1)
        return self.forecast(features, output_size)


def config_channels(scale):
    return {'n': (0.25, 1024), 's': (0.5, 1024), 'l': (1., 512), 'x': (1.5, 512)}[scale]


def load_checkpoint(path, device='cpu'):
    # Own checkpoints contain tensors, dicts and primitives only.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = SpectralYOLO(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model, checkpoint
