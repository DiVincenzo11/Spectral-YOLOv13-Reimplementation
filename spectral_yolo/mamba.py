"""Portable Mamba-1 selective SSM; no GRU/LSTM fallback.

The algebra follows state-spaces/mamba v1.0.1's non-fused implementation.
Reference scan computes in fp32 and has O(T) arithmetic, with a Python loop.
It is a correctness path, not a claim of fused-CUDA throughput.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def selective_scan(u, delta, a, b, c, d, z):
    # u,delta,z: [B,D,T]; A: [D,N]; B,C: [B,N,T].
    dtype = u.dtype
    u, delta, a, b, c = u.float(), delta.float(), a.float(), b.float(), c.float()
    state = u.new_zeros(u.shape[0], u.shape[1], a.shape[-1])
    ys = []
    for t in range(u.shape[-1]):
        step = delta[:, :, t, None]
        state = torch.exp(step * a[None]) * state + step * b[:, None, :, t] * u[:, :, t, None]
        ys.append((state * c[:, None, :, t]).sum(-1))
    return ((torch.stack(ys, -1) + u * d.float()[None, :, None]) * F.silu(z.float())).to(dtype)


class ReferenceMamba(nn.Module):
    def __init__(self, dim=256, state_dim=16, expand=2, conv_kernel=4, backend='reference'):
        super().__init__()
        if backend not in ('reference', 'cuda'):
            raise ValueError('backend must be reference or cuda')
        self.backend = backend
        self.inner = dim * expand
        self.state_dim = state_dim
        self.dt_rank = math.ceil(dim / 16)
        self.in_proj = nn.Linear(dim, self.inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.inner, self.inner, conv_kernel,
                                groups=self.inner, padding=conv_kernel - 1)
        self.x_proj = nn.Linear(self.inner, self.dt_rank + state_dim * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner)
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
        dt = torch.exp(torch.rand(self.inner) * math.log(100) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.arange(1, state_dim + 1).float().log().repeat(self.inner, 1))
        self.D = nn.Parameter(torch.ones(self.inner))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.inner, dim, bias=False)

    def forward(self, inputs):
        length = inputs.shape[1]
        x, z = self.in_proj(inputs).transpose(1, 2).chunk(2, 1)
        x = F.silu(self.conv1d(x)[..., :length])
        dt, b, c = self.x_proj(x.transpose(1, 2)).split([self.dt_rank, self.state_dim, self.state_dim], -1)
        # Apply bias once, before softplus.
        delta = F.softplus(self.dt_proj(dt)).transpose(1, 2)
        a = -self.A_log.float().exp()
        b, c = b.transpose(1, 2).contiguous(), c.transpose(1, 2).contiguous()
        if self.backend == 'cuda':
            if not x.is_cuda:
                raise RuntimeError('CUDA scan requested on a CPU device; use backend=reference.')
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            except ImportError as error:
                raise RuntimeError('Install compatible mamba-ssm CUDA kernels or select reference.') from error
            y = selective_scan_fn(x, delta, a, b, c, self.D.float(), z=z, delta_softplus=False)
        else:
            y = selective_scan(x, delta, a, b, c, self.D, z)
        return self.out_proj(y.transpose(1, 2))


class BioMamba(nn.Module):
    def __init__(self, in_channels=256, hidden_dim=256, state_dim=16, layers=2,
                 grid=8, horizons=(1, 2, 4), temporal_dropout=0.1, backend='reference'):
        super().__init__()
        self.grid, self.horizons = grid, tuple(horizons)
        self.temporal_dropout = temporal_dropout
        self.compress = nn.Conv2d(in_channels, hidden_dim, 1)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.blocks = nn.ModuleList(ReferenceMamba(hidden_dim, state_dim, backend=backend) for _ in range(layers))
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in horizons)
        self.decode = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
                                    nn.SiLU(), nn.Conv2d(hidden_dim // 2, 1, 1))

    def forward(self, feature_sequence, output_size=None):
        if feature_sequence.ndim != 5 or feature_sequence.shape[1] < 1:
            raise ValueError('Expected nonempty [batch,time,channels,height,width] features')
        batch, length, channels, height, width = feature_sequence.shape
        x = self.compress(feature_sequence.reshape(batch * length, channels, height, width))
        x = F.adaptive_avg_pool2d(x, (self.grid, self.grid))
        dim = x.shape[1]
        x = x.reshape(batch, length, dim, self.grid, self.grid)
        if self.training and self.temporal_dropout:
            keep = torch.rand(batch, length, 1, 1, 1, device=x.device) >= self.temporal_dropout
            keep[:, -1] = True  # Always retain the current observation.
            x = x * keep
        # One temporal stream per spatial cell, preserving spatial information.
        x = x.permute(0, 3, 4, 1, 2).reshape(batch * self.grid ** 2, length, dim)
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        last = x[:, -1]
        maps = []
        for head in self.heads:
            latent = head(last).reshape(batch, self.grid, self.grid, dim).permute(0, 3, 1, 2)
            heatmap = self.decode(latent)
            if output_size is not None:
                heatmap = F.interpolate(heatmap, size=output_size, mode='bilinear', align_corners=False)
            maps.append(heatmap.sigmoid())
        return torch.stack(maps, 1)  # [B,3,1,H,W], health index 0=dead, 1=healthy


def forecasting_loss(predicted, target, active_horizons=None, smoothness=0.01):
    if predicted.shape != target.shape:
        raise ValueError(f'Forecast target shape {target.shape} != prediction {predicted.shape}')
    if active_horizons is not None:
        predicted, target = predicted[:, active_horizons], target[:, active_horizons]
    mse = F.mse_loss(predicted, target)
    dy = (predicted[..., 1:, :] - predicted[..., :-1, :]).abs().mean() if predicted.shape[-2] > 1 else mse * 0
    dx = (predicted[..., 1:] - predicted[..., :-1]).abs().mean() if predicted.shape[-1] > 1 else mse * 0
    return mse + smoothness * (dx + dy)
