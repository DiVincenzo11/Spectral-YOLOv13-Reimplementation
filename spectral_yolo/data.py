"""YOLO detection dataset and explicitly timestamped forecasting manifests."""
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from .model import read_config


def detection_loader(data_yaml, config, split='train', batch_size=None):
    spec = read_config(data_yaml)
    names = spec['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if names != config['classes']:
        raise ValueError(f'Dataset class order {names} differs from model {config["classes"]}')
    path = Path(spec.get('path', '.'))
    if not path.is_absolute():
        path = Path(data_yaml).resolve().parent / path
    source = Path(spec[split])
    if not source.is_absolute():
        source = path / source
    settings = config['train']
    hyp = get_cfg()
    hyp.mosaic = settings.get('mosaic', 1.) if split == 'train' else 0.
    hyp.mixup = 0.
    hyp.copy_paste = 0.
    dataset = YOLODataset(img_path=str(source), data={'names': names, 'nc': len(names)},
                          imgsz=settings['image_size'], augment=split == 'train',
                          hyp=hyp, rect=False, batch_size=batch_size or settings['batch_size'],
                          stride=32, pad=0.0, prefix=f'{split}: ')
    loader = DataLoader(dataset, batch_size=batch_size or settings['batch_size'],
                         shuffle=split == 'train', num_workers=settings.get('workers', 0),
                         collate_fn=YOLODataset.collate_fn, generator=torch.Generator().manual_seed(config['seed']))
    return loader


def move_detection(batch, device):
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    result['img'] = result['img'].float() / 255. if result['img'].dtype == torch.uint8 else result['img'].float()
    return result


def read_yolo_labels(path, classes=4):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Missing label file {path}; create an empty file for a verified background image.')
    rows = [list(map(float, line.split())) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    labels = np.array(rows, dtype=np.float32).reshape(-1, 5)
    if len(labels):
        if not np.isfinite(labels).all() or np.any(labels[:, 0] != labels[:, 0].astype(int)):
            raise ValueError(f'Invalid class/NaN in {path}')
        if (labels[:, 0] < 0).any() or (labels[:, 0] >= classes).any():
            raise ValueError(f'Class out of range in {path}')
        if (labels[:, 1:] < 0).any() or (labels[:, 1:] > 1).any() or (labels[:, 3:] <= 0).any():
            raise ValueError(f'Expected normalized xywh with positive extent: {path}')
    return torch.from_numpy(labels)


def validate_manifest(records, horizons=(1, 2, 4)):
    seen_ids = set()
    split_files = {}
    split_intervals = {}
    for record in records:
        identifier = record['sample_id']
        if identifier in seen_ids:
            raise ValueError(f'Duplicate sample_id: {identifier}')
        seen_ids.add(identifier)
        split = record['split']
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val or test')
        inputs, targets = record['input_paths'], record['target_paths']
        times, future = record['input_times_days'], record['target_times_days']
        if len(times) != len(inputs) or not times or len(targets) != len(horizons) or len(future) != len(horizons):
            raise ValueError(f'Invalid temporal lengths in {identifier}')
        if not np.isfinite(times + future).all() or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError(f'Input timestamps must be finite and strictly ordered: {identifier}')
        if min(future) <= max(times):
            raise ValueError(f'Future leakage in {identifier}')
        if not np.allclose(np.array(future) - times[-1], np.array(horizons) * 7, atol=1e-5):
            raise ValueError(f'Target times must match {horizons} weeks after current frame: {identifier}')
        split_files.setdefault(split, set()).update(inputs + targets)
        group = record.get('sequence_id', record['site_id'])
        split_intervals.setdefault((group, split), []).append((times[0], max(future)))
    splits = list(split_files)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            if split_files[a] & split_files[b]:
                raise ValueError(f'File leakage across {a}/{b}')
    # Separate observations AND prediction windows, not just last input frame.
    groups = {group for group, _ in split_intervals}
    for group in groups:
        bounds = {split: (min(a for a, _ in intervals), max(b for _, b in intervals))
                  for (g, split), intervals in split_intervals.items() if g == group}
        for i, a in enumerate(splits):
            for b in splits[i + 1:]:
                if a in bounds and b in bounds:
                    lo1, hi1 = bounds[a]; lo2, hi2 = bounds[b]
                    if max(lo1, lo2) <= min(hi1, hi2):
                        raise ValueError(f'Temporal-window overlap for sequence {group}: {a}/{b}')


class TemporalDataset(Dataset):
    def __init__(self, manifest, config, split='train'):
        self.root = Path(manifest).resolve().parent
        records = [json.loads(line) for line in Path(manifest).read_text(encoding='utf-8').splitlines() if line.strip()]
        validate_manifest(records, config['temporal']['horizons_weeks'])
        self.records = [r for r in records if r['split'] == split]
        if not self.records:
            raise ValueError(f'No {split} temporal samples')
        self.size = config['train']['image_size']
        self.classes = len(config['classes'])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        images = []
        for p in record['input_paths']:
            image = np.asarray(Image.open(self.root / p).convert('RGB').resize((self.size, self.size))).copy()
            images.append(torch.from_numpy(image).permute(2, 0, 1).float() / 255)
        targets = []
        for p in record['target_paths']:
            array = np.load(self.root / p, allow_pickle=False).astype(np.float32)
            if array.ndim != 2 or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
                raise ValueError(f'Health map must be finite [H,W] in [0,1]: {p}')
            targets.append(torch.from_numpy(array)[None])
        labels = read_yolo_labels(self.root / record['last_label_path'], self.classes)
        return {'images': torch.stack(images), 'targets': torch.stack(targets), 'labels': labels,
                'sample_id': record['sample_id'], 'site_id': record['site_id']}


def temporal_collate(samples):
    if len({sample['images'].shape for sample in samples}) != 1:
        raise ValueError('Batch sequences must share length/size; use batch=1 for variable lengths.')
    labels = torch.cat([s['labels'] for s in samples])
    return {'images': torch.stack([s['images'] for s in samples]),
            'targets': torch.stack([s['targets'] for s in samples]),
            'cls': labels[:, :1], 'bboxes': labels[:, 1:],
            'batch_idx': torch.cat([torch.full((len(s['labels']),), i) for i, s in enumerate(samples)]),
            'sample_ids': [s['sample_id'] for s in samples]}
