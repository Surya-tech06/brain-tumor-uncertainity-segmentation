import numpy as np
import pytest

torch = pytest.importorskip("torch")

from btseg.model import UNet, combined_loss, mc_dropout_samples  # noqa: E402


def test_unet_forward_and_mc_dropout_is_stochastic():
    net = UNet(in_channels=1, base=4, depth=2, p_drop=0.5)
    x = torch.randn(1, 1, 32, 32)
    assert net(x).shape == (1, 1, 32, 32)
    s = mc_dropout_samples(net, x, T=4)
    assert s.shape == (4, 32, 32) and s.std(0).max() > 0


def test_loss_decreases_on_toy_batch():
    net = UNet(in_channels=1, base=4, depth=2, p_drop=0.0)
    x = torch.randn(2, 1, 32, 32); y = (x > 0).float()
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    first = combined_loss(net(x), y).item()
    for _ in range(30):
        opt.zero_grad(); l = combined_loss(net(x), y); l.backward(); opt.step()
    assert l.item() < first
