"""Two-level Db4 wavelet gating: paper section 3.2, equations 1-5."""
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class SubbandGates(nn.Module):
    def __init__(self, channels, initial=(0.5, 0.5, 0.5, 0.7)):
        super().__init__()
        # Order LL, LH, HL, HH; alpha, gamma, delta, beta.
        self.filters = nn.ModuleList(nn.Conv2d(channels, channels, 1) for _ in range(4))
        self.strength = nn.Parameter(torch.tensor(initial).float())

    def apply_band(self, x, band):
        sign = 1.0 if band < 2 else -1.0
        return x * (1.0 + sign * self.strength[band] * self.filters[band](x).sigmoid())


class WaveletGate(nn.Module):
    """Recursive filtering of LL and all detail bands, with explicit boundary policy.

    Paper does not specify multilevel gating ties or boundary extension. We use
    independent gates per depth and symmetric extension; see docs/IMPLEMENTATION.md.
    """
    def __init__(self, channels, levels=2, wave='db4', mode='symmetric'):
        super().__init__()
        self.levels = levels
        self.dwt = DWTForward(J=1, wave=wave, mode=mode)
        self.idwt = DWTInverse(wave=wave, mode=mode)
        self.gates = nn.ModuleList(SubbandGates(channels) for _ in range(levels))

    def _filter(self, x, depth):
        height, width = x.shape[-2:]
        # Explicit even padding keeps odd feature maps reconstructible.
        x = F.pad(x, (0, width % 2, 0, height % 2), mode='replicate')
        ll, detail = self.dwt(x)
        gate = self.gates[depth]
        ll = gate.apply_band(ll, 0)
        if depth + 1 < self.levels:
            ll = self._filter(ll, depth + 1)
        # pytorch_wavelets detail order: LH, HL, HH.
        bands = torch.stack([gate.apply_band(detail[0][:, :, i], i + 1) for i in range(3)], 2)
        return self.idwt((ll, [bands]))[..., :height, :width]

    def forward(self, x):
        return self._filter(x, 0)


class WIONeck(nn.Module):
    """Db4 filtering followed by top-down upsample/concatenate FPN fusion."""
    def __init__(self, in_channels, width=256, pyramid_levels=5, enabled=True):
        super().__init__()
        if pyramid_levels not in (3, 5):
            raise ValueError('pyramid_levels must be 3 (P3-P5) or 5 (P3-P7)')
        self.project = nn.ModuleList(nn.Conv2d(c, width, 1) for c in in_channels)
        self.extra = nn.ModuleList(nn.Conv2d(width, width, 3, stride=2, padding=1)
                                   for _ in range(pyramid_levels - 3))
        self.filters = nn.ModuleList(WaveletGate(width) if enabled else nn.Identity()
                                     for _ in range(pyramid_levels))
        # GroupNorm avoids batch-size-one failures on small P7 maps.
        def block(cin):
            return nn.Sequential(nn.Conv2d(cin, width, 3, padding=1, bias=False),
                                 nn.GroupNorm(min(16, width), width), nn.SiLU())
        self.fuse = nn.ModuleList(block(width * 2) for _ in range(pyramid_levels - 1))
        self.output = nn.ModuleList(block(width) for _ in range(pyramid_levels))

    def forward(self, xs):
        xs = [proj(x) for proj, x in zip(self.project, xs)]
        for down in self.extra:
            xs.append(down(xs[-1]))
        xs = [gate(x) for gate, x in zip(self.filters, xs)]
        for i in range(len(xs) - 2, -1, -1):
            up = F.interpolate(xs[i + 1], size=xs[i].shape[-2:], mode='nearest')
            xs[i] = self.fuse[i](torch.cat((xs[i], up), 1))
        return [layer(x) for layer, x in zip(self.output, xs)]


class SpectralAugment(nn.Module):
    """Paper section 3.7.2: sample HH multiplier uniformly in [0.5, 2.0]."""
    def __init__(self):
        super().__init__()
        self.dwt = DWTForward(J=1, wave='db4', mode='symmetric')
        self.idwt = DWTInverse(wave='db4', mode='symmetric')

    def forward(self, images):
        ll, yh = self.dwt(images)
        factor = torch.empty(images.shape[0], 1, 1, 1, device=images.device).uniform_(0.5, 2.0)
        details = torch.stack((yh[0][:, :, 0], yh[0][:, :, 1], yh[0][:, :, 2] * factor), 2)
        out = self.idwt((ll, [details]))
        return out[..., :images.shape[-2], :images.shape[-1]].clamp(0, 1)
