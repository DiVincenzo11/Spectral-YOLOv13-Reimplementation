"""CIoU + categorical CE + supervised contrastive loss; eqs. 6 and 9.

Task-aligned anchor assignment and the regression distribution decode reuse
the official YOLOv13 implementation. Background class and DFL are explicit
choices (DFL loss disabled in the paper-oriented default).
"""
import torch
from torch.nn import functional as F
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors, dist2bbox
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.ops import xywh2xyxy
from .prototype import supervised_contrastive


class DetectionLoss:
    def __init__(self, model):
        self.head = model.head
        self.config = model.config['loss']
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=model.head.classes, alpha=0.5, beta=6.)

    def __call__(self, raw, batch, update_bank=True):
        head, cfg = self.head, self.config
        regression, logits = raw['regression'].float(), raw['logits'].float()
        batch_size, nanchors, _ = regression.shape
        anchors, strides = make_anchors(raw['features'], head.stride, 0.5)
        bins = torch.arange(head.reg_max, device=regression.device).float()
        distances = regression.reshape(batch_size, nanchors, 4, head.reg_max).softmax(-1) @ bins
        boxes = dist2bbox(distances, anchors, xywh=False)
        classes, gt_boxes = [], []
        counts = [(batch['batch_idx'].long() == b).sum().item() for b in range(batch_size)]
        max_count = max(counts, default=0)
        gt_labels = logits.new_zeros(batch_size, max_count, 1)
        gt_bboxes = logits.new_zeros(batch_size, max_count, 4)
        height, width = batch['img'].shape[-2:]
        scale = logits.new_tensor([width, height, width, height])
        for b, count in enumerate(counts):
            mask = batch['batch_idx'].long() == b
            gt_labels[b, :count] = batch['cls'][mask].reshape(-1, 1)
            gt_bboxes[b, :count] = xywh2xyxy(batch['bboxes'][mask] * scale)
        target_labels, target_boxes, target_scores, foreground, _ = self.assigner(
            logits.detach().softmax(-1)[..., :head.classes], (boxes.detach() * strides),
            anchors * strides, gt_labels, gt_bboxes, gt_bboxes.sum(-1, keepdim=True) > 0)
        foreground = foreground.bool()
        labels = target_labels.long().clone()
        labels[~foreground] = head.classes
        if foreground.any() and update_bank and head.use_prototypes:
            head.bank.update(raw['embeddings'][foreground], labels[foreground])
            # Recalculate after first-batch class-wise initialization. This uses a
            # cloned bank, so subsequent updates cannot corrupt the saved graph.
            logits = torch.cat((head.bank.logits(raw['embeddings']), raw['logits'][..., -1:]), -1)
        weights = logits.new_ones(head.classes + 1)
        weights[-1] = cfg.get('background_weight', 0.1)
        classification = F.cross_entropy(logits.reshape(-1, head.classes + 1), labels.flatten(), weight=weights)
        zero = regression.sum() * 0
        loc, dfl, con = zero, zero, zero
        if foreground.any():
            targets_grid = target_boxes / strides
            quality = target_scores.sum(-1)[foreground]
            denominator = quality.sum().clamp_min(1e-6)
            iou = bbox_iou(boxes[foreground], targets_grid[foreground], xywh=False, CIoU=True).flatten()
            loc = ((1 - iou) * quality).sum() / denominator
            if cfg.get('dfl', 0) > 0:
                _, dfl = BboxLoss(head.reg_max).to(regression.device)(regression, boxes, anchors,
                          targets_grid, target_scores, target_scores.sum().clamp_min(1), foreground)
            if head.use_prototypes:
                con = supervised_contrastive(raw['embeddings'][foreground], labels[foreground],
                                               cfg.get('temperature', 0.1), cfg.get('max_contrastive_samples', 512))
        total = cfg.get('localization', 1.) * loc + cfg.get('classification', .3) * classification
        total = total + cfg.get('contrastive', .7) * con + cfg.get('dfl', 0.) * dfl
        return total, {'ciou': loc.detach(), 'ce': classification.detach(), 'supcon': con.detach(),
                       'dfl': dfl.detach(), 'foreground_anchors': int(foreground.sum())}
