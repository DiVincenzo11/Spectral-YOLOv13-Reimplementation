"""Generate obviously synthetic geometric data for plumbing checks, NOT corals."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import yaml


def generate(root, size=64, seed=42):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    classes = ['Healthy', 'Sub-healthy', 'Bleached', 'Dead']
    colors = [(30, 180, 70), (240, 170, 40), (240, 230, 220), (90, 90, 100)]
    for split, n in [('train', 12), ('val', 4), ('test', 4)]:
        for part in ('images', 'labels'): (root / part / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = rng.integers(0, 25, (size, size, 3), dtype=np.uint8)
            image = Image.fromarray(arr); draw = ImageDraw.Draw(image)
            cls = i % 4
            draw.rectangle((size//4, size//4, 3*size//4, 3*size//4), fill=colors[cls])
            draw.text((2, 2), 'SYNTHETIC', fill='white')
            image.save(root / 'images' / split / f'{i:03d}.png')
            (root / 'labels' / split / f'{i:03d}.txt').write_text(f'{cls} 0.5 0.5 0.5 0.5\n')
    (root / 'data.yaml').write_text(yaml.safe_dump({'path': '.', 'train': 'images/train', 'val': 'images/val',
                                                   'test': 'images/test', 'names': classes}), encoding='utf-8')
    records = []
    temporal = root / 'temporal'; temporal.mkdir(exist_ok=True)
    for split, start in [('train', 0), ('val', 100), ('test', 200)]:
        for sample in range(2 if split == 'train' else 1):
            prefix = f'{split}_{sample}'
            input_paths, target_paths = [], []
            for t in range(4):
                p = temporal / f'{prefix}_input{t}.png'
                image = Image.new('RGB', (size, size), colors[sample % 4])
                ImageDraw.Draw(image).text((2, 2), 'SYNTHETIC', fill='white')
                image.save(p); input_paths.append(p.relative_to(root).as_posix())
            for horizon in (1, 2, 4):
                p = temporal / f'{prefix}_week{horizon}.npy'
                np.save(p, np.full((8, 8), 1 - horizon * .15, dtype=np.float32))
                target_paths.append(p.relative_to(root).as_posix())
            label = temporal / f'{prefix}_last.txt'; label.write_text(f'{sample % 4} 0.5 0.5 0.5 0.5\n')
            records.append({'sample_id': prefix, 'site_id': 'SYNTHETIC_SITE', 'sequence_id': f'SYNTHETIC_REGION_{sample}',
                            'split': split, 'input_paths': input_paths,
                            'input_times_days': [start + t for t in (-3, -2, -1, 0)],
                            'target_paths': target_paths, 'target_times_days': [start + 7, start + 14, start + 28],
                            'last_label_path': label.relative_to(root).as_posix()})
    (root / 'temporal.jsonl').write_text('\n'.join(json.dumps(r) for r in records) + '\n', encoding='utf-8')
    (root / 'PROVENANCE.txt').write_text('Synthetic rectangles and constant health maps for software checks only.\n'
                                      'Not CR-Mix++, not coral images, no biological validation.\n', encoding='utf-8')
    print(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='data/synthetic'); p.add_argument('--size', type=int, default=64)
    args = p.parse_args(); generate(args.out, args.size)
