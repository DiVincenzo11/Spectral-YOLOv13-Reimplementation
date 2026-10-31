"""Forward, backward, prototype updates, checkpoint roundtrip and 672-step scan."""
import json
from pathlib import Path
import sys
import tempfile
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral_yolo.model import SpectralYOLO, read_config, load_checkpoint
from spectral_yolo.loss import DetectionLoss
from spectral_yolo.mamba import ReferenceMamba, forecasting_loss


def main():
    torch.set_num_threads(4); torch.manual_seed(42)
    root = Path(__file__).resolve().parents[1]
    config = read_config(root / 'configs/smoke.yaml')
    model = SpectralYOLO(config)
    model.train()
    image = torch.rand(2, 3, 64, 64)
    batch = {'img': image, 'cls': torch.tensor([[0], [1], [2], [3]]).float(),
             'batch_idx': torch.tensor([0, 0, 1, 1]),
             'bboxes': torch.tensor([[.3,.3,.3,.3],[.7,.7,.3,.3],[.3,.3,.3,.3],[.7,.7,.3,.3]])}
    raw = model(image)
    criterion = DetectionLoss(model)
    det, stats = criterion(raw, batch)
    det.backward()
    assert torch.isfinite(det)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(model.head.bank.seen)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.neck.filters.parameters())
    model.zero_grad(set_to_none=True)
    sequence = torch.rand(1, 4, 3, 64, 64)
    predicted = model.forecast_images(sequence, (8, 8))
    future = torch.rand_like(predicted)
    fore = forecasting_loss(predicted, future)
    fore.backward()
    assert model.backbone[0].conv.weight.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.forecast.parameters())
    model.eval()
    with torch.no_grad():
        raw = model(image); before = model.head.decode(raw)[0]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'check.pt'
        torch.save({'config': config, 'model': model.state_dict()}, path)
        reloaded, _ = load_checkpoint(path)
        reloaded.eval()
        with torch.no_grad(): after = reloaded.head.decode(reloaded(image))[0]
        torch.testing.assert_close(before, after)
    long_scan = ReferenceMamba(dim=16, state_dim=4)
    with torch.no_grad(): long_out = long_scan(torch.rand(1, 672, 16))
    assert long_out.shape == (1,672,16) and torch.isfinite(long_out).all()
    report = {'status': 'PASS: software checks only, no real coral data or paper metric reproduction',
              'torch': torch.__version__, 'device': 'cpu', 'model_scale': 'n',
              'detection_loss': float(det.detach()), 'forecast_loss': float(fore.detach()),
              'foreground_anchors': stats['foreground_anchors'],
              'prototype_seen': model.head.bank.seen.tolist(), 'forecast_shape': list(predicted.shape),
              'selective_scan_672_shape': list(long_out.shape), 'checkpoint_roundtrip': True,
              'backbone_and_wavelet_gradients': True, 'forecast_detach_verified': True}
    out = root / 'artifacts/verification/smoke.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
