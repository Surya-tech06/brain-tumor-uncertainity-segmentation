"""CPU-only fallback backend: a bagged decision-tree ensemble.

This backend exists so the whole pipeline (segmentation + uncertainty maps +
web app + experiments) runs on any laptop without a GPU or PyTorch.  Each tree
of the random forest is treated as one stochastic "sample" p_t(x), exactly like
one MC-Dropout pass or one deep-ensemble member, so the same epistemic /
aleatoric decomposition applies.  It is a *baseline*, not a substitute for the
U-Net in `model.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier

from .synthetic import make_dataset, normalize
from .uncertainty import decompose, review_score


def estimate_brain_mask(img: np.ndarray) -> np.ndarray:
    """Cheap skull-less brain mask for uploads that come without one."""
    sm = ndi.gaussian_filter(img.astype(np.float32), 1.0)
    m = sm > 0.2 * np.percentile(sm, 99)
    m = ndi.binary_fill_holes(ndi.binary_opening(m, iterations=2))
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    m = ndi.binary_erosion(m, iterations=3)  # drop the skull rim
    return m


def extract_features(z: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """Per-pixel features from a z-scored single-channel slice -> (H*W, F)."""
    h, w = z.shape
    g = lambda s: ndi.gaussian_filter(z, s)  # noqa: E731
    g1, g2, g4, g8 = g(1), g(2), g(4), g(8)
    gm = ndi.gaussian_gradient_magnitude(z, 1.5)
    gm3 = ndi.gaussian_gradient_magnitude(z, 3.0)
    lap = ndi.gaussian_laplace(z, 2.0)
    mu5 = ndi.uniform_filter(z, 5)
    sd5 = np.sqrt(np.clip(ndi.uniform_filter(z**2, 5) - mu5**2, 0, None))
    mu9 = ndi.uniform_filter(z, 9)
    sd9 = np.sqrt(np.clip(ndi.uniform_filter(z**2, 9) - mu9**2, 0, None))
    ys, xs = np.nonzero(brain)
    cy, cx = (ys.mean(), xs.mean()) if len(ys) else (h / 2, w / 2)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    dy, dx = (yy - cy) / h, (xx - cx) / w
    r = np.hypot(dy, dx)
    feats = [z, g1, g2, g4, g8, z - g8, g2 - g8, gm, gm3, lap, sd5, sd9, dy, dx, r]
    return np.stack(feats, -1).reshape(h * w, -1).astype(np.float32)


class ClassicalEngine:
    name = "bagged-tree ensemble (CPU)"

    def __init__(self, n_trees: int = 64, seed: int = 0):
        self.rf = RandomForestClassifier(
            n_estimators=n_trees, max_depth=16, min_samples_leaf=10, max_features=0.5,
            bootstrap=True, n_jobs=2, random_state=seed,
        )
        self.review_threshold = 0.25
        self.fitted = False

    # ---------------------------------------------------------------- training
    def fit(self, images, masks, brains, per_image: int = 2500, seed: int = 0):
        rng = np.random.default_rng(seed)
        X, y = [], []
        for img, m, b in zip(images, masks, brains):
            z = normalize(img[:1], b)[0]
            F = extract_features(z, b)
            near = ndi.binary_dilation(m.astype(bool), iterations=6) & b  # hard region
            idx_hard = np.flatnonzero(near.ravel())
            idx_all = np.flatnonzero(b.ravel())
            k = per_image // 2
            pick = np.concatenate([
                rng.choice(idx_hard, min(k, len(idx_hard)), replace=False) if len(idx_hard) else [],
                rng.choice(idx_all, min(per_image - k, len(idx_all)), replace=False),
            ]).astype(int)
            X.append(F[pick]), y.append(m.ravel()[pick])
        self.rf.fit(np.concatenate(X), np.concatenate(y))
        self.fitted = True
        return self

    def calibrate_review_threshold(self, images, masks, brains, quantile: float = 0.8):
        scores = [self.predict(i[:1], b)["review"] for i, b in zip(images, brains)]
        self.review_threshold = float(np.quantile(scores, quantile))
        return self.review_threshold

    # --------------------------------------------------------------- inference
    def predict(self, image: np.ndarray, brain: np.ndarray | None = None, thr: float = 0.5):
        """image: (C,H,W) or (H,W); only channel 0 (FLAIR-like) is used."""
        img = image[None] if image.ndim == 2 else image[:1]
        if brain is None:
            brain = estimate_brain_mask(img[0])
        z = normalize(img, brain)[0]
        F = extract_features(z, brain)
        h, w = z.shape
        samples = np.stack([t.predict_proba(F)[:, 1] for t in self.rf.estimators_]).reshape(-1, h, w)
        samples *= brain[None]
        out = decompose(samples)
        for k in out:
            out[k] = out[k] * brain
        mask = (out["prob"] >= thr) & brain
        out.update(mask=mask, brain=brain, review=review_score(out["total"], mask))
        out["needs_review"] = bool(out["review"] > self.review_threshold)
        return out

    # ------------------------------------------------------------- persistence
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.rf, path, compress=3)
        path.with_suffix(".json").write_text(json.dumps({"review_threshold": self.review_threshold}))

    @classmethod
    def load(cls, path):
        eng = cls()
        eng.rf = joblib.load(path)
        meta = Path(path).with_suffix(".json")
        if meta.exists():
            eng.review_threshold = json.loads(meta.read_text())["review_threshold"]
        eng.fitted = True
        return eng

    @classmethod
    def train_default(cls, n_train: int = 60, n_val: int = 30, seed: int = 1):
        xs, ys, bs = make_dataset(n_train, seed=seed)
        eng = cls().fit(xs, ys, bs)
        xv, yv, bv = make_dataset(n_val, seed=seed + 100)
        eng.calibrate_review_threshold(xv, yv, bv)
        return eng
