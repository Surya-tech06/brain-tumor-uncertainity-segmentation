"""Reproduce every number and figure reported in the project report.

    python scripts/run_experiments.py            # ~2 minutes on 2 CPU cores

Evaluates the CPU bagged-tree backend on held-out synthetic phantoms:
segmentation accuracy, calibration, uncertainty quality, selective prediction
and robustness under noise (domain shift).  Writes docs/results.json and the
figures in docs/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from skimage.filters import threshold_otsu  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btseg import metrics as M  # noqa: E402
from btseg.classical import ClassicalEngine, extract_features  # noqa: E402
from btseg.synthetic import make_dataset, normalize  # noqa: E402
from btseg.uncertainty import binary_entropy  # noqa: E402

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C = {"ens": "#1f5fa8", "single": "#d9822b", "grey": "#777777"}

N_TRAIN, N_VAL, N_TEST = 80, 30, 100
FRACS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)


def single_tree_prob(eng, img, brain):
    z = normalize(img[:1], brain)[0]
    F = extract_features(z, brain)
    p = eng.rf.estimators_[0].predict_proba(F)[:, 1].reshape(z.shape)
    return p * brain


def main():
    print("training ...")
    xs, ys, bs = make_dataset(N_TRAIN, seed=11)
    eng = ClassicalEngine(n_trees=64).fit(xs, ys, bs)
    xv, yv, bv = make_dataset(N_VAL, seed=12)
    thr = eng.calibrate_review_threshold(xv, yv, bv)
    xt, yt, bt = make_dataset(N_TEST, seed=2024)

    rows, outs = [], []
    for i in range(N_TEST):
        o = eng.predict(xt[i], bt[i])
        outs.append(o)
        gt, br, pred = yt[i], bt[i], o["mask"]
        has = gt.any()
        e_ens, conf_e, acc_e, cnt_e = M.ece(o["prob"], gt, br)
        p1 = single_tree_prob(eng, xt[i], br)
        e_one, conf_1, acc_1, cnt_1 = M.ece(p1, gt, br)
        sm = (xt[i][0] > threshold_otsu(xt[i][0][br])) & br
        bnd, inr, ext = M.boundary_bands(gt)
        rows.append(dict(
            has_tumor=bool(has), dice=M.dice(pred, gt), iou=M.iou(pred, gt), hd95=M.hd95(pred, gt),
            dice_single=M.dice(p1 >= 0.5, gt), dice_otsu=M.dice(sm, gt),
            ece=e_ens, ece_single=e_one, brier=M.brier(o["prob"], gt, br), brier_single=M.brier(p1, gt, br),
            auroc_total=M.error_auroc(o["total"], pred, gt, br),
            auroc_epi=M.error_auroc(o["epistemic"], pred, gt, br),
            auroc_alea=M.error_auroc(o["aleatoric"], pred, gt, br),
            retained=M.retained_dice(pred, gt, o["total"], br, FRACS).tolist(),
            ent_boundary=float(o["total"][bnd & br].mean()) if has else np.nan,
            ent_interior=float(o["total"][inr & br].mean()) if inr.any() else np.nan,
            ent_exterior=float(o["total"][ext & br].mean()),
            review=o["review"], flagged=o["needs_review"],
            pred_area=int(pred.sum()), gt_area=int(gt.sum()),
            _conf=conf_e.tolist(), _acc=acc_e.tolist(), _cnt=cnt_e.tolist(),
            _conf1=conf_1.tolist(), _acc1=acc_1.tolist(), _cnt1=cnt_1.tolist(),
        ))

    pos = [r for r in rows if r["has_tumor"]]
    nm = lambda k, rs=rows: float(np.nanmean([r[k] for r in rs]))  # noqa: E731
    sd = lambda k, rs=rows: float(np.nanstd([r[k] for r in rs]))  # noqa: E731
    res = {"config": dict(n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, n_trees=64, review_threshold=thr,
                          n_tumor_slices=len(pos), n_tumor_free=len(rows) - len(pos))}

    res["segmentation"] = {
        "ensemble": dict(dice=nm("dice", pos), dice_sd=sd("dice", pos), iou=nm("iou", pos), hd95=nm("hd95", pos)),
        "single_tree": dict(dice=nm("dice_single", pos), dice_sd=sd("dice_single", pos)),
        "otsu": dict(dice=nm("dice_otsu", pos), dice_sd=sd("dice_otsu", pos)),
    }
    res["tumor_free_false_positive_rate"] = float(np.mean([r["pred_area"] > 20 for r in rows if not r["has_tumor"]]))
    res["calibration"] = dict(
        ece_ensemble=nm("ece"), ece_single=nm("ece_single"),
        brier_ensemble=nm("brier"), brier_single=nm("brier_single"))
    res["uncertainty_quality"] = dict(
        auroc_total=nm("auroc_total"), auroc_epistemic=nm("auroc_epi"), auroc_aleatoric=nm("auroc_alea"))
    ret = np.nanmean([r["retained"] for r in pos], 0)
    res["retained_dice"] = dict(zip([str(f) for f in FRACS], ret.tolist()))
    res["entropy_by_region"] = dict(boundary=nm("ent_boundary", pos), interior=nm("ent_interior", pos),
                                    exterior=nm("ent_exterior", pos))

    # image-level failure detection
    d = np.array([r["dice"] for r in pos])
    rv = np.array([r["review"] for r in pos])
    rho, p_rho = spearmanr(rv, d)
    fail = d < 0.80
    res["failure_detection"] = dict(
        spearman_review_vs_dice=float(rho), p_value=float(p_rho), failure_def="Dice < 0.80",
        n_failures=int(fail.sum()),
        auroc_review_for_failure=float(roc_auc_score(fail, rv)) if fail.any() and not fail.all() else None)
    flagged = np.array([r["flagged"] for r in rows])
    keep = [r for r, f in zip(rows, flagged) if not f and r["has_tumor"]]
    res["selective_prediction"] = dict(
        flagged_fraction=float(flagged.mean()), dice_all=nm("dice", pos), dice_unflagged=nm("dice", keep),
        n_flagged_tumor=int(sum(1 for r, f in zip(rows, flagged) if f and r["has_tumor"])))

    # image-level risk-coverage: send the k% highest review scores to a clinician
    allrev = np.array([r["review"] for r in rows])
    order_ = np.argsort(allrev)
    rc = {}
    for k in (0.0, 0.1, 0.2, 0.3, 0.5):
        kept = order_[: int(round(len(rows) * (1 - k)))]
        ds_ = [rows[i]["dice"] for i in kept if rows[i]["has_tumor"]]
        rc[str(k)] = dict(n_kept=int(len(kept)), mean_dice=float(np.mean(ds_)), n_below_080=int(sum(x < 0.8 for x in ds_)))
    res["risk_coverage"] = rc

    # robustness under noise shift
    levels = [0.03, 0.06, 0.10, 0.15, 0.20]
    shift = []
    for lv in levels:
        xn, yn, bn = make_dataset(40, seed=500 + int(lv * 100), noise_range=(lv, lv), tumor_prob=1.0)
        ds, tot, epi, ale, fl = [], [], [], [], []
        for i in range(len(xn)):
            o = eng.predict(xn[i], bn[i])
            ds.append(M.dice(o["mask"], yn[i]))
            tot.append(o["total"][bn[i]].mean()), epi.append(o["epistemic"][bn[i]].mean())
            ale.append(o["aleatoric"][bn[i]].mean()), fl.append(o["needs_review"])
        shift.append(dict(noise=lv, dice=float(np.mean(ds)), total=float(np.mean(tot)),
                          epistemic=float(np.mean(epi)), aleatoric=float(np.mean(ale)),
                          flagged=float(np.mean(fl))))
    res["noise_shift"] = shift

    Path(ROOT / "docs" / "results.json").write_text(json.dumps(res, indent=2))

    # ------------------------------------------------------------- figures
    # 1. qualitative examples
    pick = [i for i, r in enumerate(rows) if r["has_tumor"]]
    order = sorted(pick, key=lambda i: rows[i]["dice"])
    show = [order[-1], order[len(order) // 2], order[2]]
    nofree = [i for i, r in enumerate(rows) if not r["has_tumor"]]
    if nofree:
        show.append(nofree[0])
    cols = ["FLAIR input", "Ground truth", "Prediction", "Predictive entropy", "Epistemic (MI)", "Error map"]
    fig, ax = plt.subplots(len(show), 6, figsize=(11, 1.95 * len(show)))
    for r_, i in enumerate(show):
        o = outs[i]
        err = (o["mask"] != yt[i].astype(bool)) & bt[i]
        panels = [(xt[i][0], "gray", None), (yt[i], "gray", (0, 1)), (o["mask"], "gray", (0, 1)),
                  (o["total"], "magma", (0, 1)), (o["epistemic"], "magma", (0, 1)), (err, "gray", (0, 1))]
        for c_, (im, cm, lim) in enumerate(panels):
            a = ax[r_, c_]
            a.imshow(im, cmap=cm, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1])
            a.set_xticks([]), a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            if r_ == 0:
                a.set_title(cols[c_], fontsize=9)
        ax[r_, 0].set_ylabel(f"Dice {rows[i]['dice']:.2f}" if rows[i]["has_tumor"] else "tumour-free", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "fig_examples.png", bbox_inches="tight"), plt.close()

    # 2. reliability diagram
    cnt, cnt1 = np.sum([r["_cnt"] for r in rows], 0), np.sum([r["_cnt1"] for r in rows], 0)
    conf = np.sum([np.array(r["_conf"]) * np.array(r["_cnt"]) for r in rows], 0) / np.maximum(cnt, 1)
    acc = np.sum([np.array(r["_acc"]) * np.array(r["_cnt"]) for r in rows], 0) / np.maximum(cnt, 1)
    conf1 = np.sum([np.array(r["_conf1"]) * np.array(r["_cnt1"]) for r in rows], 0) / np.maximum(cnt1, 1)
    acc1 = np.sum([np.array(r["_acc1"]) * np.array(r["_cnt1"]) for r in rows], 0) / np.maximum(cnt1, 1)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], "--", c=C["grey"], label="perfect calibration")
    ax.plot(conf[cnt > 0], acc[cnt > 0], "o-", c=C["ens"], label=f"ensemble (ECE {res['calibration']['ece_ensemble']:.3f})")
    ax.plot(conf1[cnt1 > 0], acc1[cnt1 > 0], "s-", c=C["single"], label=f"single tree (ECE {res['calibration']['ece_single']:.3f})")
    ax.set_xlabel("predicted foreground probability"), ax.set_ylabel("observed tumour frequency")
    ax.legend(frameon=False, fontsize=8), plt.tight_layout()
    plt.savefig(FIG / "fig_reliability.png"), plt.close()

    # 3. retention curve
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.plot([f * 100 for f in FRACS], ret, "o-", c=C["ens"])
    ax.set_xlabel("most-uncertain pixels discarded (%)"), ax.set_ylabel("Dice on retained pixels")
    ax.grid(alpha=0.25), plt.tight_layout()
    plt.savefig(FIG / "fig_retention.png"), plt.close()

    # 4. noise shift
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4))
    n_ = [s["noise"] for s in shift]
    ax[0].plot(n_, [s["total"] for s in shift], "o-", c=C["ens"], label="total (entropy)")
    ax[0].plot(n_, [s["epistemic"] for s in shift], "s-", c=C["single"], label="epistemic (MI)")
    ax[0].plot(n_, [s["aleatoric"] for s in shift], "^-", c=C["grey"], label="aleatoric")
    ax[0].set_xlabel("Rician noise level σ"), ax[0].set_ylabel("mean uncertainty in brain (bits)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(n_, [s["dice"] for s in shift], "o-", c=C["ens"], label="Dice")
    ax[1].plot(n_, [s["flagged"] for s in shift], "s-", c=C["single"], label="fraction flagged for review")
    ax[1].set_xlabel("Rician noise level σ"), ax[1].legend(frameon=False, fontsize=8), ax[1].set_ylim(0, 1.02)
    plt.tight_layout(), plt.savefig(FIG / "fig_noise_shift.png"), plt.close()

    # 5. where is the uncertainty?
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    e = res["entropy_by_region"]
    ax.bar(["exterior", "interior", "boundary"], [e["exterior"], e["interior"], e["boundary"]],
           color=[C["grey"], C["single"], C["ens"]])
    ax.set_ylabel("mean predictive entropy (bits)"), plt.tight_layout()
    plt.savefig(FIG / "fig_regions.png"), plt.close()

    # 6. image-level review score vs Dice
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.scatter(rv, d, s=14, c=C["ens"], alpha=0.8, label="tumour slice")
    ax.axvline(thr, ls="--", c=C["grey"]), ax.text(thr, 0.05, " review\n threshold", fontsize=8, color=C["grey"])
    ax.set_xlabel("image-level review score (mean entropy)"), ax.set_ylabel("Dice")
    ax.set_title(f"Spearman ρ = {rho:.2f}", fontsize=9)
    fig.tight_layout()
    plt.savefig(FIG / "fig_review_vs_dice.png"), plt.close()

    print(json.dumps({k: res[k] for k in ("config", "selective_prediction", "risk_coverage", "failure_detection")}, indent=1))


if __name__ == "__main__":
    main()
