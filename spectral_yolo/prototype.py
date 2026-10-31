"""Prototype initialization and momentum update (equations 7-8)."""
import torch
from torch import nn
from torch.nn import functional as F


class PrototypeBank(nn.Module):
    def __init__(self, classes=4, dim=128, momentum=0.1, temperature=0.1):
        super().__init__()
        self.momentum = momentum
        self.temperature = temperature
        self.register_buffer('centers', F.normalize(torch.randn(classes, dim), dim=-1))
        self.register_buffer('seen', torch.zeros(classes, dtype=torch.bool))
        self.register_buffer('counts', torch.zeros(classes, dtype=torch.long))

    @torch.no_grad()
    def update(self, embeddings, labels):
        # Only matched foreground training features may update the bank.
        for c in range(len(self.centers)):
            chosen = embeddings[labels == c]
            if not len(chosen):
                continue
            mean = F.normalize(chosen.detach().float(), dim=-1).mean(0)
            if not self.seen[c]:
                self.centers[c].copy_(mean)
                self.seen[c] = True
            else:
                self.centers[c].lerp_(mean, self.momentum)
            self.counts[c] += len(chosen)

    def distances(self, embeddings):
        # Cloning is necessary: later EMA updates must not invalidate autograd.
        centers = F.normalize(self.centers.detach().clone(), dim=-1)
        return (2 - 2 * F.normalize(embeddings.float(), dim=-1) @ centers.T).clamp_min(0)

    def logits(self, embeddings):
        return -self.distances(embeddings) / self.temperature

    def confidence(self, embeddings):
        """Nearest normalized prototype + relative first/second distance margin.

        Formula unspecified in paper; this is an explicit reconstruction choice.
        """
        distances = self.distances(embeddings)
        if self.seen.any():
            distances = distances.masked_fill(~self.seen, torch.inf)
        values, indices = distances.topk(2, dim=-1, largest=False)
        margin = (values[..., 1] - values[..., 0]) / values[..., 1].clamp_min(1e-6)
        margin = torch.where(torch.isinf(values[..., 1]) & torch.isfinite(values[..., 0]),
                             torch.ones_like(margin), margin)
        return indices[..., 0], margin.clamp(0, 1)


def supervised_contrastive(embeddings, labels, temperature=0.1, max_samples=512):
    """Equation 6, excluding self-pairs and anchors without another positive."""
    if len(labels) > max_samples:
        index = torch.randperm(len(labels), device=labels.device)[:max_samples]
        embeddings, labels = embeddings[index], labels[index]
    if len(labels) < 2:
        return embeddings.sum() * 0
    z = F.normalize(embeddings.float(), dim=-1)
    similarity = z @ z.T / temperature
    self_mask = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positives = (labels[:, None] == labels[None, :]) & ~self_mask
    count = positives.sum(-1)
    valid = count > 0
    if not valid.any():
        return embeddings.sum() * 0
    logprob = similarity - similarity.masked_fill(self_mask, -torch.inf).logsumexp(-1, keepdim=True)
    # masked_fill avoids 0 * inf in degenerate cases.
    return -(logprob.masked_fill(~positives, 0).sum(-1)[valid] / count[valid]).mean()
