# Uncertainty-Quantified Deep Learning for Brain Tumor Segmentation from MRI

Healthcare-AI project (B.Tech CSE, Jain (Deemed-to-be) University) — **Surya U**.

A segmentation pipeline that does not just outline a tumour but also says **how sure it is, pixel by pixel**, splitting
uncertainty into *aleatoric* (noisy / blurry data) and *epistemic* (model disagreement), and flags slices that deserve a
clinician's second look. Includes a web app, a PyTorch U-Net with Monte-Carlo Dropout, a CPU-only fallback, a
reproducible experiment script and the full project report.

> ⚠ Research prototype. **Not a medical device.** Never use it for diagnosis or treatment decisions.

## What is in the repo

| Path | Purpose |
|---|---|
| `src/btseg/model.py` | 2-D U-Net with Dropout2d, MC-Dropout & deep-ensemble sampling, Dice+BCE loss (PyTorch) |
| `src/btseg/uncertainty.py` | entropy / aleatoric / epistemic (mutual-information) decomposition, image-level review score |
| `src/btseg/metrics.py` | Dice, IoU, HD95, ECE, Brier, uncertainty-AUROC, retained-Dice |
| `src/btseg/classical.py` | CPU fallback: bagged-tree ensemble (each tree = one stochastic sample) |
| `src/btseg/synthetic.py` | procedural MRI phantom (bias field, Rician noise, soft edges, distractor lesions) |
| `src/btseg/engine.py` | picks U-Net if weights exist, otherwise the CPU fallback |
| `app/` | Flask web app (upload a slice or generate a phantom; see 3 uncertainty maps + stats) |
| `scripts/run_experiments.py` | reproduces every number/figure in the report → `docs/results.json`, `docs/figures/` |
| `scripts/train_unet.py` | trains the U-Net (synthetic data or BraTS slices) |
| `scripts/prepare_brats.py` | converts BraTS NIfTI volumes to 2-D `.npz` slices |
| `docs/` | report (`.docx`), results, figures |
| `tests/` | pytest suite (core maths, web API, optional torch tests) |

## Quick start (no GPU needed)

```bash
pip install -r requirements.txt
python -m pytest -q
python app/app.py            # first run trains + caches the CPU model (~30 s), then open http://127.0.0.1:5000
python scripts/run_experiments.py
```

## Deep-learning backend (U-Net + MC Dropout)

```bash
pip install -r requirements-torch.txt
python scripts/train_unet.py --data synthetic --epochs 15            # smoke run
# real data (BraTS needs free registration at synapse.org):
python scripts/prepare_brats.py --brats-dir /path/to/BraTS2021_Training --out data/brats_npz
python scripts/train_unet.py --data data/brats_npz --epochs 60 --base 32
python app/app.py --backend unet
```

`models/unet_mcd.pt` is picked up automatically by `app/app.py`.

## Method in one paragraph

T stochastic forward passes give foreground probabilities p_t(x). The mean is the prediction; total uncertainty is the
entropy of the mean; aleatoric is the mean of per-pass entropies; epistemic = total − aleatoric (mutual information).
An image-level *review score* (mean entropy in/around the predicted tumour) ranks slices for human review.

## Reported results (synthetic phantom, CPU backend, 100 held-out slices)

See `docs/results.json`. Highlights: Dice 0.860 ± 0.089 (single tree 0.796, Otsu 0.445); uncertainty-vs-error AUROC 0.95;
Dice on pixels left after discarding the 20 % most uncertain: 0.960; image-level review score vs Dice Spearman ρ = −0.83.
**The U-Net has not been trained on real BraTS data inside this repository** — run the commands above and put your own
numbers in the report's Table 5.3.

## Limitations

Synthetic data only for the shipped numbers; 2-D slices; binary whole-tumour label; single FLAIR channel in the app.

## License
MIT
