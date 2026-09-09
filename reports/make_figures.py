"""Generate the README figures from the committed result CSVs.

Every figure reads from reports/*.csv rather than hard-coded numbers, so a
figure can never drift from the table it illustrates. Run after the experiments:

    python reports/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

OUT = "reports/figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dedcd6"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
GREY = "#9a998f"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _style(ax, xlabel=None, title=None, subtitle=None):
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=16 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9,
                color=INK_2, va="bottom")


def fig_leakage():
    """The single most important number in the repo: what leakage buys you."""
    d = pd.read_csv("reports/leakage_experiment.csv").set_index("regime")["pr_auc"]
    labels = ["Naive\nevery available column", "Disciplined\npre-decision columns only"]
    vals = [d["naive"], d["clean"]]

    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    bars = ax.barh(labels[::-1], vals[::-1], height=0.5,
                   color=[BLUE, RED], zorder=3)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlim(0, 1.12)
    _style(ax, "PR-AUC  (base rate 0.096)",
           "A near-perfect credit model is a symptom",
           "Same model, same data, same split — trained twice")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/leakage_gap.png", dpi=200)
    plt.close(fig)


def fig_regimes():
    """What each tightening of the validation scheme costs."""
    d = pd.read_csv("reports/temporal_validation.csv")
    order = ["random", "random_matched", "temporal", "embargoed"]
    names = {"random": "Random 80/20",
             "random_matched": "Random, test set matched",
             "temporal": "Temporal",
             "embargoed": "Embargoed (the honest one)"}
    m = d.groupby("regime")["pr_auc"].agg(["mean", "std"]).reindex(order)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(order))[::-1]
    colors = [GREY, GREY, GREY, BLUE]
    ax.errorbar(m["mean"], y, xerr=m["std"], fmt="o", markersize=9,
                color=INK_2, ecolor=GRID, elinewidth=2.5, capsize=0, zorder=3,
                linestyle="none")
    for yi, c, mu in zip(y, colors, m["mean"]):
        ax.plot(mu, yi, "o", markersize=9, color=c, zorder=4)
    for yi, mu, sd in zip(y, m["mean"], m["std"]):
        ax.text(mu, yi + 0.28, f"{mu:.4f}  ±{sd:.4f}", fontsize=9,
                color=INK_2, ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels([names[o] for o in order])
    ax.set_ylim(-0.6, len(order) - 0.25)
    ax.margins(x=0.10)
    _style(ax, "PR-AUC  (whiskers = sd across cohort-year folds)",
           "Honest validation costs more than the split type",
           "Each row tightens what the model is allowed to know at fit time")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/validation_regimes.png", dpi=200)
    plt.close(fig)


def fig_models():
    """Two panels: where the models land, and whether the gap is real."""
    b = pd.read_csv("reports/baselines.csv")
    w = b.pivot(index="fold", columns="model", values="pr_auc")
    m = b.groupby("model")["pr_auc"].agg(["mean", "std"])

    order = ["trivial", "logistic_raw", "logistic_woe", "lightgbm"]
    names = {"trivial": "Trivial (majority class)",
             "logistic_raw": "Logistic, impute + one-hot",
             "logistic_woe": "Logistic, Weight-of-Evidence",
             "lightgbm": "LightGBM"}
    m = m.reindex(order)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    y = np.arange(len(order))[::-1]
    cols = [GREY, AQUA, ORANGE, BLUE]
    ax1.errorbar(m["mean"], y, xerr=m["std"], fmt="none",
                 ecolor=GRID, elinewidth=2.5, zorder=3)
    for yi, c, mu in zip(y, cols, m["mean"]):
        ax1.plot(mu, yi, "o", markersize=9, color=c, zorder=4)
    for yi, mu in zip(y, m["mean"]):
        ax1.text(mu, yi + 0.3, f"{mu:.4f}", fontsize=9, color=INK_2, ha="center")
    ax1.set_yticks(y); ax1.set_yticklabels([names[o] for o in order])
    ax1.set_ylim(-0.6, len(order) - 0.25)
    _style(ax1, "PR-AUC  (whiskers = sd across folds)",
           "The booster does not separate from the scorecard",
           "Embargoed walk-forward folds")
    ax1.tick_params(axis="y", length=0)

    # Panel 2: the paired per-fold differences, which is where the claim lives.
    tuned_g = pd.read_csv("reports/tuning.csv").set_index("fold")["pr_auc"]
    tuned_l = pd.read_csv("reports/logistic_tuning.csv").set_index("fold")["pr_auc"]
    comps = [("LightGBM − logistic\n(both untuned)", w["lightgbm"] - w["logistic_woe"], BLUE),
             ("LightGBM − logistic\n(both tuned)", tuned_g - tuned_l, ORANGE)]

    ax2.axvline(0, color=INK_2, linewidth=1.2, zorder=2)
    for i, (lab, d, col) in enumerate(comps):
        yy = np.full(len(d), len(comps) - 1 - i) + np.linspace(-0.13, 0.13, len(d))
        ax2.plot(d.values, yy, "o", markersize=8, color=col,
                 markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        ax2.plot([d.mean()], [len(comps) - 1 - i], "|", markersize=26,
                 color=INK, markeredgewidth=2.5, zorder=5)
    ax2.set_yticks(range(len(comps))[::-1])
    ax2.set_yticklabels([c[0] for c in comps], fontsize=9)
    ax2.set_ylim(-0.55, len(comps) - 0.45)
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    ax2.margins(x=0.16)
    _style(ax2, "Δ PR-AUC per fold  —  left of 0 favours logistic",
           "…and the gap changes sign",
           "One dot per cohort year")
    ax2.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(f"{OUT}/model_comparison.png", dpi=200)
    plt.close(fig)


def fig_calibration():
    """What ranking metrics hide: rebalancing wrecks the probability scale."""
    d = pd.read_csv("reports/imbalance.csv")
    order = ["none", "smote", "smote_rounded", "class_weight", "undersample"]
    names = {"none": "No treatment", "smote": "SMOTE-NC",
             "smote_rounded": "SMOTE-NC + rounding",
             "class_weight": "Class weights", "undersample": "Undersampling"}
    g = d.groupby("treatment")[["mean_pred", "true_rate", "pr_auc"]].mean().reindex(order)
    truth = float(d["true_rate"].mean())

    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    y = np.arange(len(order))[::-1]
    cols = [BLUE if o == "none" else RED for o in order]
    ax.barh(y, g["mean_pred"], height=0.5, color=cols, zorder=3)
    ax.axvline(truth, color=INK, linewidth=1.6, linestyle="--", zorder=5)
    ax.set_ylim(-1.15, len(order) - 0.45)
    ax.text(truth + 0.012, -0.85, f"true default rate  {truth:.3f}",
            fontsize=9, color=INK, va="center")
    for yi, v in zip(y, g["mean_pred"]):
        ax.text(v + 0.008, yi, f"{v:.3f}  ({v / truth:.1f}×)", va="center",
                fontsize=9, color=INK_2)
    ax.set_yticks(y); ax.set_yticklabels([names[o] for o in order])
    ax.set_xlim(0, 0.60)
    _style(ax, "Mean predicted probability of default",
           "Rebalancing leaves ranking alone and destroys calibration",
           "Which is exactly what cost-optimal thresholds need to be right")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/calibration_damage.png", dpi=200)
    plt.close(fig)


def fig_reliability():
    """Phase 4: do the predicted probabilities mean what they say?"""
    c = pd.read_csv("reports/reliability_curves.csv")
    c = c[c.model == "lightgbm"]
    styles = {"none": (RED, "Uncalibrated"),
              "platt": (BLUE, "Platt (2 parameters)"),
              "isotonic": (AQUA, "Isotonic (step function)")}

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    lim = 0.30
    ax.plot([0, lim], [0, lim], color=INK_2, linewidth=1.2, linestyle="--",
            zorder=2, label="perfect calibration")
    for cal, (col, lab) in styles.items():
        g = (c[c.calibrator == cal].groupby("bin")[["predicted", "observed"]]
             .mean().sort_values("predicted"))
        ax.plot(g["predicted"], g["observed"], "-o", color=col, linewidth=2,
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=lab, zorder=4)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_ylabel("Observed default rate", fontsize=9)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax, "Predicted probability of default",
           "Every version underpredicts risk on a later cohort",
           "Above the line = more defaults arrived than the model promised")
    fig.tight_layout()
    fig.savefig(f"{OUT}/reliability.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (fig_leakage, fig_regimes, fig_models, fig_calibration,
               fig_reliability):
        fn()
        print(f"wrote {fn.__name__}")
    print(f"\nfigures in {OUT}/")
