"""Validate files, class order, temporal causality, and disjoint split windows."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral_yolo.model import read_config
from spectral_yolo.data import TemporalDataset, detection_loader


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/paper.yaml'); p.add_argument('--data'); p.add_argument('--manifest')
    args = p.parse_args(); cfg = read_config(args.config)
    if not (args.data or args.manifest): p.error('Supply --data or --manifest')
    for split in ('train', 'val', 'test'):
        if args.data:
            loader = detection_loader(args.data, cfg, split, 1)
            print(f'{split}: {len(loader.dataset)} detection images')
        if args.manifest:
            ds = TemporalDataset(args.manifest, cfg, split)
            for sample in ds: pass
            print(f'{split}: {len(ds)} temporal samples; files, ranges and causal windows checked')
