"""Synthetic brain-MRI phantom generator.

Real BraTS data cannot be redistributed (and needs registration), so the
repository ships a procedural phantom that mimics the properties that matter
for uncertainty research:

* bias-field (intensity inhomogeneity) and Rician noise,
* soft tumour boundaries (partial-volume effect), i.e. genuine aleatoric
  ambiguity at the edge of the lesion,
* bright non-tumour distractor lesions, i.e. genuine epistemic ambiguity,
* variable lesion contrast, so some cases are intrinsically hard.

Channel 0 imitates FLAIR, channel 1 imitates post-contrast T1 (T1ce).
The ground truth is a binary *whole-tumour* mask.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _smooth_noise(rng: np.random.Generator, shape, sigma: float) -> np.ndarray:
    n = ndi.gaussian_filter(rng.standard_normal(shape), sigma)
    return (n / (n.std() + 1e-8)).astype(np.float32)


def _level_set(rng, shape, center, radii, angle, irregularity=0.25):
    """Irregular ellipse level set: values < 1 are inside the blob."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    y, x = yy - center[0], xx - center[1]
    ca, sa = np.cos(angle), np.sin(angle)
    xr, yr = ca * x + sa * y, -sa * x + ca * y
    d = (xr / radii[1]) ** 2 + (yr / radii[0]) ** 2
    return d + 0.6 * irregularity * _smooth_noise(rng, shape, sigma=h / 12)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_slice(
    rng: np.random.Generator,
    size: int = 128,
    noise: float = 0.05,
    contrast: float = 1.0,
    with_tumor: bool = True,
    n_distractors: int | None = None,
):
    """Return (image[2,H,W] float32, mask[H,W] uint8, brain[H,W] bool)."""
    h = w = size
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cy, cx = h / 2 + rng.uniform(-2, 2), w / 2 + rng.uniform(-2, 2)
    ry, rx = 0.40 * h * rng.uniform(0.95, 1.05), 0.34 * w * rng.uniform(0.95, 1.05)
    brain_lvl = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    brain_lvl += 0.04 * _smooth_noise(rng, (h, w), sigma=h / 10)
    brain = brain_lvl < 1.0
    skull = (brain_lvl >= 1.0) & (brain_lvl < 1.18)

    # healthy tissue: grey/white-matter texture + dark ventricles
    tissue = 0.50 + 0.07 * _smooth_noise(rng, (h, w), h / 9) + 0.03 * _smooth_noise(rng, (h, w), 1.5)
    for sgn in (-1, 1):
        v = ((yy - cy) / (0.10 * h)) ** 2 + ((xx - cx - sgn * 0.07 * w) / (0.035 * w)) ** 2
        tissue -= 0.38 * _sigmoid((1 - v) * 4)
    flair = tissue.copy()
    t1ce = 0.45 + 0.5 * (tissue - 0.5) + 0.03 * _smooth_noise(rng, (h, w), 1.5)

    # distractor hyper-intensities (small white-matter lesions) -> not tumour
    if n_distractors is None:
        n_distractors = int(rng.integers(0, 4))
    for _ in range(n_distractors):
        ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0.1, 0.8)
        c = (cy + rad * ry * np.sin(ang) * 0.85, cx + rad * rx * np.cos(ang) * 0.85)
        lv = _level_set(rng, (h, w), c, (rng.uniform(0.02, 0.045) * h,) * 2, 0.0, 0.2)
        flair += rng.uniform(0.15, 0.35) * _sigmoid((1 - lv) * 5)

    mask = np.zeros((h, w), np.uint8)
    if with_tumor:
        ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0.15, 0.55)
        c = (cy + rad * ry * np.sin(ang), cx + rad * rx * np.cos(ang))
        r_edema = rng.uniform(0.09, 0.17) * h
        theta = rng.uniform(0, np.pi)
        ratio = rng.uniform(0.7, 1.3)
        edema = _level_set(rng, (h, w), c, (r_edema, r_edema * ratio), theta, 0.35)
        core = _level_set(rng, (h, w), c, (r_edema * 0.45, r_edema * 0.45 * ratio), theta, 0.35)
        a_edema = _sigmoid((1 - edema) * 6)  # soft boundary -> partial volume
        a_core = _sigmoid((1 - core) * 6)
        ring = np.exp(-((core - 1.0) ** 2) / 0.08)
        flair += contrast * (0.32 * a_edema + 0.22 * a_core)
        t1ce += contrast * (0.04 * a_edema - 0.12 * a_core + 0.45 * ring)
        mask = ((a_edema > 0.5) & brain).astype(np.uint8)

    bias = 1.0 + 0.15 * _smooth_noise(rng, (h, w), h / 3)
    imgs = []
    for ch in (flair, t1ce):
        x = np.where(brain, ch, 0.0) + 0.12 * skull
        x = x * bias
        n1, n2 = rng.normal(0, noise, x.shape), rng.normal(0, noise, x.shape)
        imgs.append(np.sqrt((x + n1) ** 2 + n2**2))  # Rician noise
    return np.stack(imgs).astype(np.float32), mask, brain


def make_dataset(n: int, seed: int = 0, size: int = 128, noise_range=(0.03, 0.07),
                 tumor_prob: float = 0.85, contrast_range=(0.6, 1.2)):
    """Generate `n` slices. Returns images[N,2,H,W], masks[N,H,W], brains[N,H,W]."""
    rng = np.random.default_rng(seed)
    xs, ys, bs = [], [], []
    for _ in range(n):
        img, m, b = generate_slice(
            rng, size, noise=rng.uniform(*noise_range),
            contrast=rng.uniform(*contrast_range), with_tumor=rng.random() < tumor_prob,
        )
        xs.append(img), ys.append(m), bs.append(b)
    return np.stack(xs), np.stack(ys), np.stack(bs)


def normalize(img: np.ndarray, brain: np.ndarray | None = None) -> np.ndarray:
    """Per-channel z-score inside the brain mask (standard MRI pre-processing)."""
    img = np.asarray(img, np.float32)
    out = np.zeros_like(img)
    for c in range(img.shape[0]):
        ch = img[c]
        sel = ch[brain] if brain is not None and brain.any() else ch.ravel()
        out[c] = (ch - sel.mean()) / (sel.std() + 1e-6)
        if brain is not None:
            out[c][~brain] = 0.0
    return out
