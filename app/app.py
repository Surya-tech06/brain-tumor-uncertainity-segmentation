"""Flask web app: upload an MRI slice (or generate a phantom) and get the tumour
segmentation together with pixel-wise uncertainty maps.

    python app/app.py            # http://127.0.0.1:5000

RESEARCH PROTOTYPE - not a medical device, not for clinical use.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btseg.engine import load_engine  # noqa: E402
from btseg.metrics import dice  # noqa: E402
from btseg.synthetic import generate_slice  # noqa: E402

SIZE = 128
UP = 3  # display up-scaling
CMAP = matplotlib.colormaps["magma"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = load_engine(app.config.get("BACKEND", "auto"))
    return _engine


# ------------------------------------------------------------------ helpers
def _png_b64(rgb: np.ndarray) -> str:
    rgb = cv2.resize(rgb, None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _gray_rgb(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
    g = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return np.repeat((g * 255).astype(np.uint8)[..., None], 3, -1)


def _outline(rgb, mask, color):
    out = rgb.copy()
    m = mask.astype(np.uint8)
    edge = m - cv2.erode(m, np.ones((3, 3), np.uint8))
    out[edge.astype(bool)] = color
    return out


def _heat(unc, base_rgb, alpha=0.85):
    heat = (CMAP(np.clip(unc, 0, 1))[..., :3] * 255).astype(np.uint8)
    a = np.clip(unc * 3, 0, 1)[..., None] * alpha
    return (base_rgb * (1 - a) + heat * a).astype(np.uint8)


def read_upload(file) -> np.ndarray:
    name = (file.filename or "").lower()
    data = file.read()
    if name.endswith(".npy"):
        arr = np.load(io.BytesIO(data), allow_pickle=False).astype(np.float32)
        arr = arr[0] if arr.ndim == 3 else arr
        if arr.ndim != 2:
            raise ValueError("`.npy` must hold an (H,W) or (C,H,W) array")
    elif name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError("could not decode image")
        arr = arr.astype(np.float32)
    else:
        raise ValueError("unsupported file type (use .png, .jpg or .npy)")
    return cv2.resize(arr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def analyse(img: np.ndarray, gt: np.ndarray | None = None, brain: np.ndarray | None = None):
    eng = engine()
    out = eng.predict(img, brain)
    mask, tot, epi, ale = out["mask"], out["total"], out["epistemic"], out["aleatoric"]
    base = _gray_rgb(img if img.ndim == 2 else img[0])
    seg = _outline(base, mask, (255, 60, 60))
    if gt is not None:
        seg = _outline(seg, gt, (60, 220, 90))
    stats = {
        "tumour_pixels": int(mask.sum()),
        "tumour_fraction_of_brain": round(float(mask.sum() / max(out["brain"].sum(), 1)), 4),
        "mean_entropy_in_brain_bits": round(float(tot[out["brain"]].mean()), 4),
        "mean_epistemic_bits": round(float(epi[out["brain"]].mean()), 4),
        "mean_aleatoric_bits": round(float(ale[out["brain"]].mean()), 4),
        "review_score": round(float(out["review"]), 4),
        "review_threshold": round(float(getattr(eng, "review_threshold", 0.0)), 4),
        "needs_review": bool(out["needs_review"]),
        "uncertain_pixels_pct": round(float((tot > 0.8)[out["brain"]].mean() * 100), 2),
        "backend": eng.name,
    }
    if gt is not None:
        stats["dice_vs_ground_truth"] = round(dice(mask, gt), 4)
    return {
        "images": {
            "input": _png_b64(base), "segmentation": _png_b64(seg),
            "entropy": _png_b64(_heat(tot, base)), "epistemic": _png_b64(_heat(epi, base)),
            "aleatoric": _png_b64(_heat(ale, base)),
        },
        "stats": stats,
    }


# ------------------------------------------------------------------- routes
@app.get("/")
def index():
    return render_template("index.html", size=SIZE * UP)


@app.get("/api/health")
def health():
    return jsonify(status="ok", backend=engine().name)


@app.get("/api/sample")
def sample():
    seed = int(request.args.get("seed", 0))
    noise = float(request.args.get("noise", 0.05))
    contrast = float(request.args.get("contrast", 1.0))
    rng = np.random.default_rng(seed)
    img, mask, brain = generate_slice(rng, SIZE, noise=noise, contrast=contrast, with_tumor=bool(rng.random() < 0.9))
    res = analyse(img, gt=mask if mask.any() else None, brain=brain)
    res["stats"]["source"] = f"synthetic phantom (seed {seed}, noise {noise})"
    return jsonify(res)


@app.post("/api/segment")
def segment():
    f = request.files.get("file")
    if f is None:
        return jsonify(error="no file uploaded"), 400
    try:
        img = read_upload(f)
    except Exception as e:  # noqa: BLE001
        return jsonify(error=str(e)), 400
    res = analyse(img)
    res["stats"]["source"] = f"upload: {f.filename}"
    return jsonify(res)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=["auto", "unet", "classical"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    app.config["BACKEND"] = a.backend
    engine()  # warm up (trains + caches the CPU model on first run)
    app.run(host=a.host, port=a.port, debug=False)
