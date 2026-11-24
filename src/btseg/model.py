"""PyTorch 2-D U-Net with Monte-Carlo Dropout and deep-ensemble inference.

Requires `torch` (see requirements-torch.txt).  Imported lazily by the rest of
the package so that the CPU fallback works without it.

Uncertainty is obtained in two ways:

* **MC Dropout** (Gal & Ghahramani, 2016): dropout layers stay active at test
  time and T stochastic passes are averaged.
* **Deep ensembles** (Lakshminarayanan et al., 2017): M independently trained
  networks are queried once each.

Both return a (T, H, W) stack of foreground probabilities that is fed to
`btseg.uncertainty.decompose`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int, p_drop: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Classic encoder-decoder with skip connections (Ronneberger et al., 2015)."""

    def __init__(self, in_channels: int = 1, n_classes: int = 1, base: int = 16,
                 depth: int = 4, p_drop: float = 0.2):
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList([DoubleConv(in_channels, chs[0], p_drop)])
        for i in range(depth):
            self.enc.append(DoubleConv(chs[i], chs[i + 1], p_drop))
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([DoubleConv(chs[i] * 2, chs[i], p_drop) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            s = skips.pop()
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([s, x], 1))
        return self.head(x)  # logits


def enable_dropout(model: nn.Module) -> None:
    """Put the network in eval mode but keep Dropout layers stochastic."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def mc_dropout_samples(model: UNet, x: torch.Tensor, T: int = 20) -> np.ndarray:
    """x: (1,C,H,W) -> (T,H,W) foreground probabilities."""
    enable_dropout(model)
    return np.stack([torch.sigmoid(model(x))[0, 0].cpu().numpy() for _ in range(T)])


@torch.no_grad()
def ensemble_samples(models: list[UNet], x: torch.Tensor) -> np.ndarray:
    out = []
    for m in models:
        m.eval()
        out.append(torch.sigmoid(m(x))[0, 0].cpu().numpy())
    return np.stack(out)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (den + eps)).mean()


def combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + soft_dice_loss(logits, target)
