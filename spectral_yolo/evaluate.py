"""Evaluate a saved checkpoint; never substitute paper numbers for measurements."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ultralytics.utils.ops import xywh2xyxy
from .model import load_checkpoint
from .data import detection_loader, move_detection, TemporalDataset, temporal_collate
from .metrics import DetectionMetrics, postprocess


@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(args.threads)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    cfg = model.config
    if args.task == 'detect':
        if not args.data: raise ValueError('--data required for detection evaluation')
        if checkpoint.get('task') == 'forecast':
            raise ValueError('A forecast-only checkpoint has no validated trained detector; use a detect/joint checkpoint.')
        loader = detection_loader(args.data, cfg, args.split, args.batch)
        metrics = DetectionMetrics(cfg['classes'])
        for batch in loader:
            batch = move_detection(batch, args.device)
            boxes, scores = model.head.decode(model(batch['img']))
            results = postprocess(boxes, scores, args.conf)
            h, w = batch['img'].shape[-2:]
            for i, prediction in enumerate(results):
                mask = batch['batch_idx'].long() == i
                gt = xywh2xyxy(batch['bboxes'][mask] * boxes.new_tensor([w, h, w, h]))
                metrics.update(prediction, gt, batch['cls'][mask].flatten())
        report = metrics.compute()
    else:
        if not args.manifest: raise ValueError('--manifest required for forecasting evaluation')
        if checkpoint.get('task') == 'detect':
            raise ValueError('Detection-only training leaves Bio-Mamba untrained; use forecast/joint checkpoint.')
        dataset = TemporalDataset(args.manifest, cfg, args.split)
        loader = DataLoader(dataset, batch_size=args.batch, collate_fn=temporal_collate)
        absolute, squared, count = None, None, 0
        for batch in loader:
            target = batch['targets'].to(args.device)
            prediction = model.forecast_images(batch['images'].to(args.device), target.shape[-2:])
            error = prediction - target
            a = error.abs().sum((0, 2, 3, 4)); s = error.square().sum((0, 2, 3, 4))
            absolute = a if absolute is None else absolute + a
            squared = s if squared is None else squared + s
            count += error.shape[0] * error.shape[2] * error.shape[3] * error.shape[4]
        report = {'samples': len(dataset), 'horizons_weeks': cfg['temporal']['horizons_weeks'],
                  'mae': (absolute / count).tolist(), 'rmse': (squared / count).sqrt().tolist()}
    report.update({'checkpoint': str(args.checkpoint), 'split': args.split,
                   'data': str(args.data or args.manifest),
                   'status': 'Measured on supplied data only; NOT reproduced CR-Mix++ paper results'})
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--task', choices=['detect', 'forecast'], default='detect')
    p.add_argument('--data'); p.add_argument('--manifest'); p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--batch', type=int, default=1); p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4); p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--out', default='runs/evaluation.json')
    return p


if __name__ == '__main__': evaluate(parser().parse_args())
