"""Convert BraTS NIfTI volumes into 2-D .npz slices for scripts/train_unet.py.

Usage:
    pip install nibabel
    python scripts/prepare_brats.py --brats-dir /path/to/BraTS2021_Training --out data/brats_npz

Expected layout (BraTS 2021 naming):  <case>/<case>_flair.nii.gz and <case>/<case>_seg.nii.gz
Whole tumour = any non-zero label.  Each slice is cropped to the brain's bounding box,
resized to --size x --size, and saved as image[1,H,W] (FLAIR) + mask[H,W].
Slices with fewer than --min-brain brain pixels are skipped; slices without tumour are
kept with probability --neg-keep so the model also sees tumour-free anatomy.

Dataset: https://www.synapse.org/#!Synapse:syn51156910 (BraTS) - registration required.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brats-dir", required=True)
    ap.add_argument("--out", default="data/brats_npz")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--min-brain", type=int, default=2000)
    ap.add_argument("--neg-keep", type=float, default=0.15)
    ap.add_argument("--stride", type=int, default=2, help="take every n-th axial slice")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for case in sorted(Path(a.brats_dir).iterdir()):
        fl, sg = case / f"{case.name}_flair.nii.gz", case / f"{case.name}_seg.nii.gz"
        if not (fl.exists() and sg.exists()):
            continue
        vol, seg = nib.load(fl).get_fdata().astype(np.float32), nib.load(sg).get_fdata() > 0
        for k in range(0, vol.shape[2], a.stride):
            sl, m = vol[:, :, k], seg[:, :, k]
            if (sl > 0).sum() < a.min_brain:
                continue
            if not m.any() and rng.random() > a.neg_keep:
                continue
            ys, xs = np.nonzero(sl > 0)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            sl, m = sl[y0:y1, x0:x1], m[y0:y1, x0:x1]
            sl = cv2.resize(sl, (a.size, a.size), interpolation=cv2.INTER_LINEAR)
            m = cv2.resize(m.astype(np.uint8), (a.size, a.size), interpolation=cv2.INTER_NEAREST)
            np.savez_compressed(out / f"{case.name}_{k:03d}.npz", image=sl[None], mask=m)
            n += 1
    print(f"wrote {n} slices to {out}")


if __name__ == "__main__":
    main()
