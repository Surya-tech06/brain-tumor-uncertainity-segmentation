import numpy as np
import pytest

from btseg import metrics as M
from btseg.synthetic import generate_slice, make_dataset, normalize
from btseg.uncertainty import binary_entropy, decompose


def test_entropy_bounds():
    assert binary_entropy(np.array([0.5]))[0] == pytest.approx(1.0, abs=1e-6)
    assert binary_entropy(np.array([0.0, 1.0])).max() < 1e-5


def test_decomposition_identical_samples_have_no_epistemic():
    s = np.full((8, 4, 4), 0.3, np.float32)
    d = decompose(s)
    assert np.allclose(d["epistemic"], 0, atol=1e-5)
    assert np.allclose(d["total"], d["aleatoric"], atol=1e-5)


def test_decomposition_disagreeing_samples_have_epistemic():
    s = np.stack([np.zeros((4, 4)), np.ones((4, 4))] * 4).astype(np.float32)
    d = decompose(s)
    assert np.allclose(d["total"], 1.0, atol=1e-4)  # mean prob 0.5
    assert np.allclose(d["epistemic"], 1.0, atol=1e-4)  # members are certain but disagree


def test_dice_iou_hd95():
    a = np.zeros((20, 20), bool); a[5:15, 5:15] = True
    assert M.dice(a, a) == 1 and M.iou(a, a) == 1 and M.hd95(a, a) == 0
    b = np.zeros_like(a); b[5:15, 10:20] = True
    assert M.dice(a, b) == pytest.approx(0.5)
    assert M.dice(np.zeros_like(a), np.zeros_like(a)) == 1.0


def test_ece_perfectly_calibrated_is_small():
    rng = np.random.default_rng(0)
    p = rng.random((200, 200)).astype(np.float32)
    y = (rng.random(p.shape) < p)
    e, *_ = M.ece(p, y, np.ones_like(y, bool))
    assert e < 0.02


def test_synthetic_shapes_and_labels():
    x, y, b = make_dataset(5, seed=3, size=64)
    assert x.shape == (5, 2, 64, 64) and y.shape == (5, 64, 64) and b.shape == (5, 64, 64)
    assert set(np.unique(y)) <= {0, 1}
    img, m, br = generate_slice(np.random.default_rng(1), 64, with_tumor=False)
    assert m.sum() == 0
    z = normalize(img, br)
    assert abs(z[0][br].mean()) < 1e-3


def test_review_score_zero_without_prediction():
    from btseg.uncertainty import review_score
    assert review_score(np.ones((8, 8)), np.zeros((8, 8), bool)) == 0.0
