"""Train the MC-Dropout U-Net.

Examples
--------
# quick sanity run on the built-in phantom (CPU is fine)
python scripts/train_unet.py --data synthetic --epochs 15

# real data prepared with scripts/prepare_brats.py (npz files: image[C,H,W], mask[H,W])
python scripts/train_unet.py --data data/brats_npz --channels 1 --epochs 60 --base 32

Writes models/unet_mcd.pt (weights + config + calibrated review threshold) and
models/unet_history.json.  Requires PyTorch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from btseg.metrics import dice  # noqa: E402
from btseg.model import UNet, combined_loss, mc_dropout_samples  # noqa: E402
from btseg.synthetic import make_dataset, normalize  # noqa: E402
from btseg.uncertainty import decompose, review_score  # noqa: E402


class SliceDataset(Dataset):
    def __init__(self, images, masks, brains, channels: int, augment: bool):
        self.x = np.stack([normalize(i[:channels], b) for i, b in zip(images, brains)])
        self.y = masks.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x, y = self.x[i], self.y[i][None]
        if self.augment:
            if np.random.rand() < 0.5:
                x, y = x[..., ::-1], y[..., ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            x = x * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.1, 0.1)
        return torch.from_numpy(np.ascontiguousarray(x, np.float32)), torch.from_numpy(np.ascontiguousarray(y, np.float32))


def load_npz_folder(folder: Path):
    xs, ys, bs = [], [], []
    for f in sorted(folder.glob("*.npz")):
        d = np.load(f)
        img = d["image"].astype(np.float32)
        xs.append(img), ys.append(d["mask"].astype(np.uint8)), bs.append(img[0] > 0)
    return np.stack(xs), np.stack(ys), np.stack(bs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--p-drop", type=float, default=0.2)
    ap.add_argument("--n-train", type=int, default=400, help="synthetic only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=20, help="MC-dropout passes for validation")
    ap.add_argument("--out", default=str(ROOT / "models" / "unet_mcd.pt"))
    a = ap.parse_args()

    torch.manual_seed(a.seed), np.random.seed(a.seed)
    if a.data == "synthetic":
        xs, ys, bs = make_dataset(a.n_train + 60, seed=a.seed)
    else:
        xs, ys, bs = load_npz_folder(Path(a.data))
    n_val = max(int(0.15 * len(xs)), 1)
    tr = SliceDataset(xs[:-n_val], ys[:-n_val], bs[:-n_val], a.channels, True)
    va = SliceDataset(xs[-n_val:], ys[-n_val:], bs[-n_val:], a.channels, False)
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(in_channels=a.channels, n_classes=1, base=a.base, depth=4, p_drop=a.p_drop)
    model = UNet(**cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    hist, best = [], -1.0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(a.epochs):
        model.train()
        losses = []
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = combined_loss(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            d = [dice(torch.sigmoid(model(torch.from_numpy(va.x[i])[None].to(dev)))[0, 0].cpu().numpy() > 0.5,
                      va.y[i]) for i in range(len(va))]
        rec = dict(epoch=ep + 1, loss=float(np.mean(losses)), val_dice=float(np.mean(d)))
        hist.append(rec)
        print(rec, flush=True)
        if rec["val_dice"] > best:
            best = rec["val_dice"]
            torch.save({"state_dict": model.state_dict(), "config": cfg}, a.out)

    # calibrate the 'needs review' threshold on the validation split (80th percentile)
    ckpt = torch.load(a.out, map_location=dev)
    model.load_state_dict(ckpt["state_dict"])
    scores = []
    for i in range(len(va)):
        o = decompose(mc_dropout_samples(model, torch.from_numpy(va.x[i])[None].to(dev), a.T))
        scores.append(review_score(o["total"], o["prob"] > 0.5))
    ckpt["review_threshold"] = float(np.quantile(scores, 0.8))
    torch.save(ckpt, a.out)
    Path(a.out).with_name("unet_history.json").write_text(json.dumps(hist, indent=2))
    print(f"best val Dice {best:.4f}; review threshold {ckpt['review_threshold']:.3f}; saved {a.out}")


if __name__ == "__main__":
    main()
