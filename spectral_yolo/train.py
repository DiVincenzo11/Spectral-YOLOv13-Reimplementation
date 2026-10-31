"""Train detection, forecasting, or their combined objective on supplied data."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import SpectralYOLO, read_config, load_checkpoint
from .loss import DetectionLoss
from .mamba import forecasting_loss
from .wavelet import SpectralAugment
from .data import detection_loader, move_detection, TemporalDataset, temporal_collate


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    torch.set_num_threads(args.threads)
    config = read_config(args.config)
    if args.epochs is not None: config['train']['epochs'] = args.epochs
    if args.batch is not None:
        config['train']['batch_size'] = args.batch
        config['train']['temporal_batch_size'] = args.batch
    if args.image_size: config['train']['image_size'] = args.image_size
    if args.scale: config['model']['scale'] = args.scale
    if args.no_wio: config['model']['wio'] = False
    if args.no_cp: config['model']['cp_head'] = False
    if args.no_forecast: config['model']['bio_mamba'] = False
    settings = config['train']
    if settings['epochs'] < 1 or min(settings['batch_size'], settings['temporal_batch_size']) < 1:
        raise ValueError('Epochs and batch sizes must be positive')
    if settings['image_size'] < 64 or settings['image_size'] % 32:
        raise ValueError('image_size must be a multiple of 32 and at least 64 for this backbone')
    set_seed(config['seed'])
    device = torch.device(args.device)
    if args.checkpoint:
        model, saved = load_checkpoint(args.checkpoint, device)
        # Architecture and loss settings must match; training epochs may change.
        if saved['config']['model'] != config['model'] or saved['config']['temporal'] != config['temporal']:
            raise ValueError('Checkpoint architecture/temporal config differs; use its original config.')
        model.config = config
    else:
        model = SpectralYOLO(config).to(device)
    model.train()
    if args.task == 'detect':
        if not args.data: raise ValueError('--data is required for detection training')
        loader = detection_loader(args.data, config)
        if model.forecast is not None:
            model.forecast.requires_grad_(False)
    else:
        if not args.manifest: raise ValueError('--manifest required for forecast/joint training')
        if model.forecast is None: raise ValueError('Forecast task needs bio_mamba=true')
        dataset = TemporalDataset(args.manifest, config, 'train')
        loader = DataLoader(dataset, batch_size=settings['temporal_batch_size'], shuffle=True,
                            num_workers=settings.get('workers', 0), collate_fn=temporal_collate)
        if args.task == 'forecast':
            model.backbone.requires_grad_(False); model.neck.requires_grad_(False); model.head.requires_grad_(False)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if getattr(parameter, '_no_weight_decay', False) or parameter.ndim < 2 else decay).append(parameter)
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': settings['weight_decay']},
                                   {'params': no_decay, 'weight_decay': 0}], lr=settings['learning_rate'])
    amp = bool(settings.get('amp', True) and device.type == 'cuda')
    # torch.cuda.amp.GradScaler also exists in the paper's PyTorch 2.2.
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = DetectionLoss(model)
    augmentation = SpectralAugment().to(device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    history = []
    epochs = settings['epochs']
    total_steps = epochs * len(loader)
    warmup_steps = settings['warmup_epochs'] * len(loader)
    step = 0
    for epoch in range(epochs):
        if args.task == 'detect' and settings.get('close_mosaic', 0) and epoch == max(0, epochs - settings['close_mosaic']):
            from ultralytics.cfg import get_cfg
            loader.dataset.close_mosaic(get_cfg())
        for index, batch in enumerate(loader):
            if args.max_batches and index >= args.max_batches: break
            if step < warmup_steps:
                factor = (step + 1) / max(1, warmup_steps)
            else:
                factor = .01 + .99 * (1 + math.cos(math.pi * (step - warmup_steps) /
                                                    max(1, total_steps - warmup_steps))) / 2
            for group in optimizer.param_groups:
                group['lr'] = settings['learning_rate'] * factor
            optimizer.zero_grad(set_to_none=True)
            stats = {}
            context = torch.autocast(device_type='cuda', dtype=torch.float16) if amp else nullcontext()
            with context:
                if args.task == 'detect':
                    batch = move_detection(batch, device)
                    if settings.get('spectral_augmentation', True):
                        batch['img'] = augmentation(batch['img'])
                    loss, stats = criterion(model(batch['img']), batch)
                else:
                    images = batch['images'].to(device)
                    targets = batch['targets'].to(device)
                    # Same sampled corruption applies independently per frame; no spatial
                    # mosaic for temporal sequences, so coordinates stay aligned.
                    if settings.get('spectral_augmentation', True):
                        images = torch.stack([augmentation(images[:, t]) for t in range(images.shape[1])], 1)
                    predicted = model.forecast_images(images, output_size=targets.shape[-2:])
                    fractions = config['temporal'].get('curriculum_fractions', [1/3, 2/3])
                    progress = (epoch + 1) / epochs
                    nactive = 1 if progress <= fractions[0] else 2 if progress <= fractions[1] else 3
                    forecast = forecasting_loss(predicted, targets, list(range(nactive)),
                                                  config['temporal'].get('smoothness_weight', .01))
                    loss = config['loss']['forecast'] * forecast
                    stats['forecast'] = forecast.detach()
                    if args.task == 'joint':
                        detection_batch = {'img': images[:, -1], **{k: batch[k].to(device) for k in ['cls', 'bboxes', 'batch_idx']}}
                        detection_loss, det_stats = criterion(model(images[:, -1]), detection_batch)
                        loss = loss + detection_loss
                        stats.update(det_stats)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, batch {index}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], settings['grad_clip'])
            if not torch.isfinite(grad_norm): raise FloatingPointError('Nonfinite gradient norm')
            scaler.step(optimizer); scaler.update()
            row = {'epoch': epoch + 1, 'step': step, 'loss': float(loss.detach()),
                   'grad_norm': float(grad_norm), 'lr': optimizer.param_groups[0]['lr'],
                   **{k: float(v) for k, v in stats.items()}}
            history.append(row); print(json.dumps(row), flush=True)
            step += 1
        torch.save({'format_version': 1, 'config': config, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                    'task': args.task, 'data_source': args.data or args.manifest,
                    'status': 'reimplementation checkpoint; NOT original paper weights'}, out / 'last.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return history


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/paper.yaml')
    p.add_argument('--task', choices=['detect', 'forecast', 'joint'], default='detect')
    p.add_argument('--data'); p.add_argument('--manifest'); p.add_argument('--checkpoint')
    p.add_argument('--out', default='runs/train'); p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int); p.add_argument('--batch', type=int); p.add_argument('--image-size', type=int)
    p.add_argument('--scale', choices=['n', 's', 'l', 'x']); p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--no-wio', action='store_true'); p.add_argument('--no-cp', action='store_true')
    p.add_argument('--no-forecast', action='store_true')
    return p


if __name__ == '__main__':
    train(parser().parse_args())
