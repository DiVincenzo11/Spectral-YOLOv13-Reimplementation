import copy
import torch
import pytest
from spectral_yolo.wavelet import WaveletGate
from spectral_yolo.prototype import PrototypeBank, supervised_contrastive
from spectral_yolo.mamba import ReferenceMamba, selective_scan
from spectral_yolo.metrics import DetectionMetrics
from spectral_yolo.data import validate_manifest

torch.set_num_threads(2)


@pytest.mark.parametrize('shape', [(2,3,16,20), (1,2,5,7), (1,2,2,2), (1,2,1,1)])
def test_wavelet_identity_and_gradient(shape):
    module = WaveletGate(shape[1])
    for gate in module.gates:
        gate.strength.data.zero_()
    x = torch.randn(shape, requires_grad=True)
    y = module(x)
    torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_prototype_ema_and_confidence():
    bank = PrototypeBank(2,2,momentum=.2)
    bank.update(torch.tensor([[1.,0.],[0.,1.]]), torch.tensor([0,1]))
    first = bank.centers.clone()
    bank.update(torch.tensor([[0.,1.]]),torch.tensor([0]))
    torch.testing.assert_close(bank.centers[0], .8*first[0]+.2*torch.tensor([0.,1.]))
    label, margin = bank.confidence(torch.tensor([[0.,1.]]))
    assert label.item() == 1 and 0 <= margin.item() <= 1
    saved = bank.centers.clone()
    bank.confidence(torch.rand(3,2))
    assert torch.equal(saved, bank.centers)


def test_contrastive_separation_and_singletons():
    labels = torch.tensor([0,0,1,1])
    good = torch.tensor([[1.,0.],[1.,.01],[0.,1.],[.01,1.]], requires_grad=True)
    bad = torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    loss = supervised_contrastive(good,labels)
    assert loss < supervised_contrastive(bad,labels)
    loss.backward(); assert torch.isfinite(good.grad).all()
    single = supervised_contrastive(good[:1],labels[:1]); assert single.item() == 0


def test_mamba_causal_and_backward():
    torch.manual_seed(4)
    module = ReferenceMamba(8,4)
    a = torch.randn(2,6,8,requires_grad=True)
    b = a.detach().clone(); b[:,4:] += 10
    y = module(a)
    torch.testing.assert_close(y[:,:4], module(b)[:,:4])
    y.square().mean().backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0


def test_selective_scan_against_scalar_recurrence():
    torch.manual_seed(7)
    u, delta, z = torch.randn(1,2,5),torch.rand(1,2,5),torch.randn(1,2,5)
    a,b,c,d = -torch.rand(2,3),torch.randn(1,3,5),torch.randn(1,3,5),torch.randn(2)
    actual = selective_scan(u,delta,a,b,c,d,z)
    expected = torch.zeros_like(actual)
    for channel in range(2):
        states = [torch.tensor(0.) for _ in range(3)]
        for t in range(5):
            for n in range(3):
                states[n] = (delta[0,channel,t]*a[channel,n]).exp()*states[n]+delta[0,channel,t]*b[0,n,t]*u[0,channel,t]
            value = sum(states[n]*c[0,n,t] for n in range(3))+d[channel]*u[0,channel,t]
            expected[0,channel,t] = value*torch.nn.functional.silu(z[0,channel,t])
    torch.testing.assert_close(actual,expected)


def test_ap_perfect_missing_and_duplicates():
    metric = DetectionMetrics(['a','b'])
    metric.update(torch.tensor([[0.,0.,10.,10.,.9,0.]]),torch.tensor([[0.,0.,10.,10.]]),torch.tensor([0]))
    assert metric.compute()['map50_95'] == pytest.approx(1.)
    assert metric.compute()['per_class'][1]['ap50'] is None
    empty = DetectionMetrics(['a'])
    empty.update(torch.empty(0,6),torch.tensor([[0.,0.,10.,10.]]),torch.tensor([0]))
    assert empty.compute()['map50'] == 0
    # High-confidence false positive preceding a true positive lowers AP.
    fp = DetectionMetrics(['a'])
    fp.update(torch.tensor([[20.,20.,30.,30.,.95,0.],[0.,0.,10.,10.,.9,0.]]),
              torch.tensor([[0.,0.,10.,10.]]),torch.tensor([0]))
    assert fp.compute()['map50'] == pytest.approx(.5)


def test_manifest_rejects_future_and_cross_split_leakage():
    r = {'sample_id':'one','site_id':'site','split':'train','input_paths':['a','b'],
         'target_paths':['c','d','e'],'input_times_days':[-1,0],'target_times_days':[7,14,28]}
    validate_manifest([r])
    bad = copy.deepcopy(r); bad['target_times_days'][0] = -1
    with pytest.raises(ValueError,match='leakage'): validate_manifest([bad])
    second = copy.deepcopy(r); second['sample_id']='two'; second['split']='test'
    with pytest.raises(ValueError,match='leakage'): validate_manifest([r,second])
    second['input_paths']=['x','y'];second['target_paths']=['z','u','v']
    with pytest.raises(ValueError,match='overlap'): validate_manifest([r,second])
