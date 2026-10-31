"""Render detections from a checkpoint, with visible reconstruction provenance."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from .model import load_checkpoint
from .metrics import postprocess


@torch.no_grad()
def predict(args):
    torch.set_num_threads(args.threads)
    model, saved = load_checkpoint(args.checkpoint, args.device)
    if saved.get('task') == 'forecast':
        raise ValueError('Forecast-only checkpoint does not establish a trained detector.')
    if model.head.use_prototypes and not model.head.bank.seen.any():
        raise ValueError('Prototype bank is uninitialized; train before running detection.')
    model.eval()
    image = Image.open(args.image).convert('RGB')
    size = model.config['train']['image_size']
    array = np.asarray(image.resize((size, size))).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1)[None].float().to(args.device) / 255
    boxes, scores = model.head.decode(model(tensor))
    rows = postprocess(boxes, scores, args.conf)[0].cpu()
    width, height = image.size
    rows[:, [0, 2]] *= width / size; rows[:, [1, 3]] *= height / size
    rows[:, [0, 2]] = rows[:, [0, 2]].clamp(0, width)
    rows[:, [1, 3]] = rows[:, [1, 3]].clamp(0, height)
    draw = ImageDraw.Draw(image)
    palette = ['#24a563', '#eca733', '#dc5655', '#8e8e8e']
    detections = []
    for x1, y1, x2, y2, confidence, cls in rows.tolist():
        label = model.config['classes'][int(cls)]
        draw.rectangle((x1, y1, x2, y2), outline=palette[int(cls)], width=2)
        draw.text((x1, max(0, y1 - 12)), f'{label} {confidence:.3f}', fill=palette[int(cls)])
        detections.append({'class': label, 'score': confidence, 'xyxy': [x1, y1, x2, y2]})
    draw.rectangle((0, 0, width, 19), fill='black')
    draw.text((3, 3), 'PAPER REIMPLEMENTATION - supplied checkpoint', fill='white')
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    output.with_suffix('.json').write_text(json.dumps({'detections': detections,
        'checkpoint': str(args.checkpoint), 'source': str(args.image),
        'status': 'Model output, not paper validation or biological ground truth'}, indent=2), encoding='utf-8')
    print(f'Saved {len(detections)} detections to {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--image', required=True)
    p.add_argument('--out', default='runs/prediction.png'); p.add_argument('--device', default='cpu')
    p.add_argument('--conf', type=float, default=.25); p.add_argument('--threads', type=int, default=4)
    predict(p.parse_args())
