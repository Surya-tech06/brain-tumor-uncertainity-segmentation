"""Draw the workflow / architecture diagrams used in the report."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
BLUE, ORANGE, GREY, GREEN = "#1f5fa8", "#d9822b", "#e9edf2", "#1d7a4d"


def box(ax, x, y, w, h, text, fc=GREY, ec="#44505c", fs=8.5, tc="#16202c"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)


def arrow(ax, p, q, c="#44505c"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=c, lw=1.2))


def workflow():
    fig, ax = plt.subplots(figsize=(9.5, 3.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    steps = [("MRI slice\n(FLAIR)", GREY), ("Pre-processing\nbrain mask,\nz-score", GREY), ("Stochastic\nsegmenter\n(T passes)", BLUE),
             ("Mean probability\n→ tumour mask", GREY), ("Uncertainty\ndecomposition", ORANGE), ("Review flag\n+ maps + stats", GREEN)]
    xs = [0.1, 1.75, 3.4, 5.05, 6.7, 8.35]
    for (t, c), x in zip(steps, xs):
        box(ax, x, 1.6, 1.45, 1.1, t, fc=c, tc="white" if c != GREY else "#16202c")
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 1.47, 2.15), (b - 0.02, 2.15))
    box(ax, 3.4, 0.2, 1.45, 0.8, "MC-Dropout U-Net\nor deep ensemble", fc="white", fs=7.5)
    arrow(ax, (4.12, 1.02), (4.12, 1.58))
    box(ax, 6.7, 0.2, 1.45, 0.8, "aleatoric · epistemic\ntotal entropy", fc="white", fs=7.5)
    arrow(ax, (7.42, 1.02), (7.42, 1.58))
    plt.savefig(FIG / "fig_workflow.png", bbox_inches="tight", dpi=170); plt.close()


def unet():
    fig, ax = plt.subplots(figsize=(9, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ch = [16, 32, 64, 128, 256]
    for i, c in enumerate(ch[:-1]):
        y = 4.0 - i * 0.95
        box(ax, 0.2 + i * 0.55, y, 1.3, 0.6, f"Conv×2 · {c}\nBN · ReLU · Drop", fc=BLUE, tc="white", fs=7)
        box(ax, 8.5 - i * 0.55, y, 1.3, 0.6, f"Conv×2 · {c}\nBN · ReLU · Drop", fc=ORANGE, tc="white", fs=7)
        arrow(ax, (1.5 + i * 0.55, y + 0.3), (8.48 - i * 0.55, y + 0.3), c="#8a96a3")
        if i < 3:
            arrow(ax, (0.85 + i * 0.55, y - 0.02), (0.85 + (i + 1) * 0.55, y - 0.33))
            arrow(ax, (9.15 - (i + 1) * 0.55, y - 0.33), (9.15 - i * 0.55, y - 0.02))
    box(ax, 3.4, 0.35, 3.2, 0.7, "Bottleneck · Conv×2 · 256\nDropout2d (kept active at test time)", fc=GREEN, tc="white", fs=7.5)
    arrow(ax, (2.45, 1.45), (3.4, 0.8)); arrow(ax, (6.6, 0.8), (7.55, 1.45))
    ax.text(0.2, 4.85, "input\n1×128×128", fontsize=7.5); ax.text(8.9, 4.85, "1×1 conv → logit → σ", fontsize=7.5)
    ax.text(5, 3.9, "skip connections (concatenate)", fontsize=7.5, ha="center", color="#5d6b7b")
    plt.savefig(FIG / "fig_unet.png", bbox_inches="tight", dpi=170); plt.close()


def system():
    fig, ax = plt.subplots(figsize=(9.5, 3.8)); ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.1, 1.4, 1.8, 1.2, "Web browser\nHTML · CSS · JS\nupload / sample", fc=GREY)
    box(ax, 2.5, 1.4, 2.0, 1.2, "Flask app\n/api/sample\n/api/segment\n/api/health", fc=BLUE, tc="white")
    box(ax, 5.1, 2.3, 2.1, 1.1, "Engine selector\nload_engine()", fc=GREY)
    box(ax, 7.8, 2.8, 2.0, 0.8, "U-Net + MC-Dropout\n(PyTorch)", fc=ORANGE, tc="white")
    box(ax, 7.8, 1.6, 2.0, 0.8, "Bagged-tree ensemble\n(CPU fallback)", fc=ORANGE, tc="white")
    box(ax, 5.1, 0.5, 2.1, 1.1, "Uncertainty module\nentropy · MI · review score", fc=GREEN, tc="white")
    box(ax, 7.8, 0.4, 2.0, 0.8, "Synthetic phantom /\nBraTS slices", fc=GREY)
    arrow(ax, (1.92, 2.0), (2.48, 2.0)); arrow(ax, (2.48, 1.8), (1.92, 1.8))
    arrow(ax, (4.52, 2.2), (5.08, 2.7)); arrow(ax, (7.22, 2.9), (7.78, 3.1)); arrow(ax, (7.22, 2.6), (7.78, 2.1))
    arrow(ax, (4.52, 1.7), (5.08, 1.2)); arrow(ax, (7.8, 1.9), (7.22, 1.3)); arrow(ax, (7.8, 0.8), (7.22, 0.9))
    plt.savefig(FIG / "fig_system.png", bbox_inches="tight", dpi=170); plt.close()


if __name__ == "__main__":
    workflow(); unet(); system(); print("ok")
