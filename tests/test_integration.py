from pathlib import Path
import copy
import pytest
import torch
from spectral_yolo.model import SpectralYOLO, read_config
from spectral_yolo.loss import DetectionLoss
from spectral_yolo.mamba import forecasting_loss

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)


def sample_batch():
    return {'img': torch.rand(2,3,64,64), 'batch_idx':torch.tensor([0,1]),
            'cls':torch.tensor([[0.],[1.]]),'bboxes':torch.tensor([[.5,.5,.5,.5],[.5,.5,.4,.4]])}


@pytest.mark.parametrize('wio,cp,bio',[(True,True,True),(False,True,True),(True,False,True),(True,True,False)])
def test_ablation_backward(wio,cp,bio):
    cfg=read_config(ROOT/'configs/smoke.yaml')
    cfg['model'].update(wio=wio,cp_head=cp,bio_mamba=bio)
    model=SpectralYOLO(cfg).train()
    batch=sample_batch()
    raw=model(batch['img'])
    loss,stats=DetectionLoss(model)(raw,batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert bool(stats['foreground_anchors'])


def test_joint_loss_and_detach_modes():
    cfg=read_config(ROOT/'configs/smoke.yaml')
    model=SpectralYOLO(cfg).train()
    images=torch.rand(1,4,3,64,64)
    predicted=model.forecast_images(images,(8,8),detach_backbone=True)
    forecasting_loss(predicted,torch.rand_like(predicted)).backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    model.zero_grad(set_to_none=True)
    predicted=model.forecast_images(images,(8,8),detach_backbone=False)
    forecasting_loss(predicted,torch.rand_like(predicted)).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone.parameters())


def test_empty_ground_truth_is_finite():
    cfg=read_config(ROOT/'configs/smoke.yaml');model=SpectralYOLO(cfg).train()
    batch={'img':torch.rand(2,3,64,64),'batch_idx':torch.empty(0),'cls':torch.empty(0,1),'bboxes':torch.empty(0,4)}
    loss,stats=DetectionLoss(model)(model(batch['img']),batch)
    loss.backward()
    assert torch.isfinite(loss) and stats['foreground_anchors']==0
    assert not model.head.bank.seen.any()


def test_five_scale_pyramid_and_eval_bank_readonly():
    cfg=read_config(ROOT/'configs/smoke.yaml');cfg['model']['pyramid_levels']=5
    model=SpectralYOLO(cfg).eval()
    initial=model.head.bank.centers.clone()
    with torch.no_grad():
        raw=model(torch.rand(1,3,128,128))
        boxes,scores=model.head.decode(raw)
    assert [f.shape[-1] for f in raw['features']]==[16,8,4,2,1]
    assert boxes.shape[-1]==4 and scores.shape[-1]==4
    assert torch.isfinite(boxes).all() and torch.isfinite(scores).all()
    assert torch.equal(initial,model.head.bank.centers)
