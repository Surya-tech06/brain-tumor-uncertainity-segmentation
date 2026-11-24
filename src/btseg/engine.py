"""Unified inference interface used by the web app and the CLI."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .classical import ClassicalEngine, estimate_brain_mask
from .synthetic import normalize
from .uncertainty import decompose, review_score

ROOT = Path(__file__).resolve().parents[2]
CLASSICAL_PATH = ROOT / "models" / "classical_rf.joblib"
UNET_PATH = ROOT / "models" / "unet_mcd.pt"


class TorchEngine:
    """MC-Dropout U-Net inference. Needs torch and trained weights."""

    def __init__(self, weights: Path = UNET_PATH, T: int = 20, device: str | None = None):
        import torch

        from .model import UNet

        self.torch = torch
        self.T = T
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(weights, map_location=self.device)
        cfg = ckpt.get("config", {})
        self.model = UNet(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.in_channels = cfg.get("in_channels", 1)
        self.review_threshold = ckpt.get("review_threshold", 0.25)
        self.name = f"U-Net + MC-Dropout (T={T}, {self.device})"

    def predict(self, image: np.ndarray, brain: np.ndarray | None = None, thr: float = 0.5):
        from .model import mc_dropout_samples

        img = image[None] if image.ndim == 2 else image
        img = img[: self.in_channels]
        if img.shape[0] < self.in_channels:  # replicate missing channels
            img = np.concatenate([img] + [img[:1]] * (self.in_channels - img.shape[0]))
        if brain is None:
            brain = estimate_brain_mask(img[0])
        x = self.torch.from_numpy(normalize(img, brain))[None].to(self.device)
        out = decompose(mc_dropout_samples(self.model, x, self.T))
        for k in out:
            out[k] = out[k] * brain
        mask = (out["prob"] >= thr) & brain
        out.update(mask=mask, brain=brain, review=review_score(out["total"], mask))
        out["needs_review"] = bool(out["review"] > self.review_threshold)
        return out


def load_engine(backend: str = "auto"):
    """backend: 'auto' | 'unet' | 'classical'.

    'auto' uses the trained U-Net when torch + weights are present, otherwise the
    CPU fallback (trained once on synthetic data and cached in models/)."""
    if backend in ("auto", "unet") and UNET_PATH.exists():
        try:
            return TorchEngine()
        except ImportError:
            if backend == "unet":
                raise
    if backend == "unet":
        raise FileNotFoundError(f"No U-Net weights at {UNET_PATH}; run scripts/train_unet.py first.")
    if CLASSICAL_PATH.exists():
        return ClassicalEngine.load(CLASSICAL_PATH)
    eng = ClassicalEngine.train_default()
    eng.save(CLASSICAL_PATH)
    return eng
