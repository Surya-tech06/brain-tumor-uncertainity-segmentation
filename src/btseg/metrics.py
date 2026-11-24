"""Segmentation-accuracy, calibration and uncertainty-quality metrics."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2 * (pred & gt).sum() / s)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    u = (pred | gt).sum()
    return 1.0 if u == 0 else float((pred & gt).sum() / u)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance in pixels (nan if one mask empty)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    d_to_g = ndi.distance_transform_edt(~sg)[sp]
    d_to_p = ndi.distance_transform_edt(~sp)[sg]
    return float(np.percentile(np.hstack([d_to_g, d_to_p]), 95))


def brier(prob: np.ndarray, gt: np.ndarray, region: np.ndarray) -> float:
    return float(((prob[region] - gt[region].astype(np.float32)) ** 2).mean())


def ece(prob: np.ndarray, gt: np.ndarray, region: np.ndarray, bins: int = 15):
    """Expected calibration error on foreground probability (Guo et al., 2017).

    Returns (ece, bin_confidence, bin_accuracy, bin_count) for reliability plots."""
    p, y = prob[region].ravel(), gt[region].ravel().astype(np.float32)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    conf, acc, cnt = np.zeros(bins), np.zeros(bins), np.zeros(bins)
    for b in range(bins):
        s = idx == b
        cnt[b] = s.sum()
        if s.any():
            conf[b], acc[b] = p[s].mean(), y[s].mean()
    e = float((cnt / max(cnt.sum(), 1) * np.abs(conf - acc)).sum())
    return e, conf, acc, cnt


def error_auroc(unc: np.ndarray, pred: np.ndarray, gt: np.ndarray, region: np.ndarray) -> float:
    """How well does uncertainty rank mis-classified pixels above correct ones?"""
    err = (pred.astype(bool) != gt.astype(bool))[region].ravel()
    if err.all() or not err.any():
        return float("nan")
    return float(roc_auc_score(err, unc[region].ravel()))


def retained_dice(pred, gt, unc, region, fractions=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2)):
    """Dice after discarding the `f` most uncertain pixels (inside `region`).

    A useful uncertainty map makes Dice rise monotonically as f grows."""
    thr_vals = unc[region]
    out = []
    for f in fractions:
        if f == 0:
            keep = region
        else:
            keep = region & (unc <= np.quantile(thr_vals, 1 - f))
        out.append(dice(pred[keep], gt[keep]))
    return np.array(out)


def boundary_bands(gt: np.ndarray, width: int = 2):
    """Split the image into (boundary band, interior, exterior) of the GT lesion."""
    gt = gt.astype(bool)
    dil, ero = ndi.binary_dilation(gt, iterations=width), ndi.binary_erosion(gt, iterations=width)
    return dil & ~ero, ero, ~dil
