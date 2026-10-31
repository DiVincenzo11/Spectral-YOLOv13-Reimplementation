"""Class-aware NMS and dataset-level 101-point AP, IoU=.50:.05:.95."""
import numpy as np
import torch
from torchvision.ops import batched_nms, box_iou


def postprocess(boxes, scores, conf=0.001, nms_iou=0.7, max_det=300):
    results = []
    for image_boxes, image_scores in zip(boxes, scores):
        confidence, labels = image_scores.max(-1)
        mask = confidence >= conf
        b, c, label = image_boxes[mask], confidence[mask], labels[mask]
        if len(b):
            keep = batched_nms(b.float(), c.float(), label, nms_iou)[:max_det]
            b, c, label = b[keep], c[keep], label[keep]
        results.append(torch.cat((b, c[:, None], label[:, None].float()), 1))
    return results


class DetectionMetrics:
    def __init__(self, names):
        self.names = names
        self.thresholds = np.arange(0.5, 0.951, 0.05)
        self.predictions = []
        self.counts = np.zeros(len(names), dtype=int)
        self.images = 0

    def update(self, prediction, gt_boxes, gt_labels):
        prediction, gt_boxes, gt_labels = prediction.cpu(), gt_boxes.cpu(), gt_labels.cpu().long()
        self.images += 1
        self.counts += np.bincount(gt_labels.numpy(), minlength=len(self.names))
        prediction = prediction[prediction[:, 4].argsort(descending=True)]
        correct = np.zeros((len(prediction), len(self.thresholds)), dtype=bool)
        if len(prediction) and len(gt_boxes):
            overlaps = box_iou(prediction[:, :4], gt_boxes).numpy()
            for t, threshold in enumerate(self.thresholds):
                matched = set()
                for p in range(len(prediction)):
                    candidates = [g for g in range(len(gt_boxes)) if g not in matched and
                                  gt_labels[g].item() == int(prediction[p, 5]) and overlaps[p, g] >= threshold]
                    if candidates:
                        g = max(candidates, key=lambda g: overlaps[p, g])
                        matched.add(g); correct[p, t] = True
        for i, row in enumerate(prediction.numpy()):
            self.predictions.append((float(row[4]), int(row[5]), correct[i]))

    def compute(self):
        rows, ap_all = [], []
        for c, name in enumerate(self.names):
            entries = sorted((p for p in self.predictions if p[1] == c), reverse=True, key=lambda p:p[0])
            if self.counts[c] == 0:
                rows.append({'class': name, 'ground_truth': 0, 'ap50': None, 'ap50_95': None})
                continue
            if entries:
                tp = np.stack([e[2] for e in entries]).cumsum(0)
                fp = np.arange(1, len(entries) + 1)[:, None] - tp
                recall = tp / self.counts[c]
                precision = tp / np.maximum(tp + fp, 1)
                ap = []
                for t in range(len(self.thresholds)):
                    envelope = np.maximum.accumulate(precision[::-1, t])[::-1]
                    ap.append(np.mean([envelope[recall[:, t] >= r].max() if (recall[:, t] >= r).any() else 0.
                                       for r in np.linspace(0, 1, 101)]))
                ap = np.array(ap)
            else:
                ap = np.zeros(len(self.thresholds))
            ap_all.append(ap)
            rows.append({'class': name, 'ground_truth': int(self.counts[c]),
                         'ap50': float(ap[0]), 'ap50_95': float(ap.mean())})
        aggregate = np.mean(ap_all, 0) if ap_all else np.zeros(len(self.thresholds))
        return {'images': self.images, 'map50': float(aggregate[0]), 'map50_95': float(aggregate.mean()),
                'per_class': rows, 'ap_protocol': '101-point precision envelope; classes without GT excluded'}
