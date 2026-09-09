"""
Phase 3 audit fix: give the scorecard the same search budget as the booster.

WHY THIS MODULE EXISTS
----------------------
Phase 3 concluded that logistic-WoE matches LightGBM. That conclusion was
reached from an unfair comparison: LightGBM got an eight-configuration nested
hyperparameter search (models/tuning.py) while the logistic regression was
fitted once at scikit-learn's default `C=1.0` with whatever bins `WOEEncoder`
produces by default. Tuning one side of a comparison and not the other is
exactly the kind of thing this project is supposed to catch.

The bias happens to run *against* the reported conclusion -- an untuned
scorecard already matched a tuned booster -- so fixing it can only strengthen
the finding or overturn it. Either outcome is worth having.

WHAT IS SEARCHED
----------------
Eight configurations, deliberately the same count as the booster's grid so the
two models get an equal search budget, over the three knobs that actually move
a WoE scorecard:

  C                 -- L2 strength on the logistic regression.
  n_bins            -- how finely numeric features are cut before weighting.
  min_bin_fraction  -- the smallest bin allowed to keep its own weight.

The third matters more than it looks. At the Phase 3 default of 2%, `addr_state`
collapses from 50 levels to 18 bins and `purpose` from 14 to 7: every level
holding under 2% of rows is merged into a single `__rare__` bucket with one
shared weight. That is standard scorecard practice -- a bin needs enough
defaults to estimate a stable weight -- but it is also a real information loss
that LightGBM does not suffer, since it can split a categorical freely. If the
booster's edge comes from geography, relaxing this is where the scorecard would
claw it back.

The nesting is identical to models/tuning.py -- configurations ranked on an
inner temporal split of each fold's own training window, winner refit on the
full outer training window and scored once on the held-out cohort year -- so
the two searches are directly comparable and neither has seen its test data.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from data.dataset import FEATURE_COLS, load_modelling_frame
from features.woe import WOEEncoder
from models.baselines import ks_statistic
from models.temporal_validation import SEED
from models.tuning import outer_folds

OUT_PATH = "reports/logistic_tuning.csv"

PARAM_GRID = [
    dict(C=1.0,  n_bins=10, min_bin_fraction=0.02),   # the Phase 3 default
    dict(C=0.1,  n_bins=10, min_bin_fraction=0.02),
    dict(C=0.01, n_bins=10, min_bin_fraction=0.02),
    dict(C=10.0, n_bins=10, min_bin_fraction=0.02),
    dict(C=1.0,  n_bins=20, min_bin_fraction=0.01),
    dict(C=1.0,  n_bins=5,  min_bin_fraction=0.05),
    dict(C=0.1,  n_bins=20, min_bin_fraction=0.005),  # finest binning tested
    dict(C=0.1,  n_bins=10, min_bin_fraction=0.005),
]


def fit_logistic(train, valid, params):
    enc = WOEEncoder(n_bins=params["n_bins"],
                     min_bin_fraction=params["min_bin_fraction"])
    Xtr = enc.fit_transform(train[FEATURE_COLS], train["default_window"])
    Xva = enc.transform(valid[FEATURE_COLS])
    model = LogisticRegression(C=params["C"], max_iter=2000, random_state=SEED)
    model.fit(Xtr, train["default_window"])
    return model.predict_proba(Xva)[:, 1]


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    print("loading modelling frame...")
    df = load_modelling_frame()
    print(f"{len(df):,} loans\n")

    rows = []
    for year, outer_train, test, inner_train, inner_val, val_year in outer_folds(df):
        print(f"fold {year}: outer train {len(outer_train):,} -> test {len(test):,}   "
              f"| inner train {len(inner_train):,} -> val {val_year} ({len(inner_val):,})")

        scores = []
        for i, params in enumerate(PARAM_GRID):
            p = fit_logistic(inner_train, inner_val, params)
            ap = average_precision_score(inner_val["default_window"], p)
            scores.append(ap)
            print(f"   cfg {i}: C={params['C']:<5} bins={params['n_bins']:>2} "
                  f"minfrac={params['min_bin_fraction']:<5}  inner PR-AUC={ap:.4f}",
                  flush=True)

        best_i = int(np.argmax(scores))
        best = PARAM_GRID[best_i]
        spread = max(scores) - min(scores)
        print(f"   -> selected cfg {best_i} on INNER validation only "
              f"(grid spread {spread:.4f})")

        p_test = fit_logistic(outer_train, test, best)
        y = test["default_window"]
        rows.append({
            "fold": year, "chosen_cfg": best_i, "inner_pr_auc": scores[best_i],
            "inner_spread": spread,
            "pr_auc": average_precision_score(y, p_test),
            "roc_auc": roc_auc_score(y, p_test),
            "brier": brier_score_loss(y, p_test),
            "ks": ks_statistic(y.values, p_test),
            **{f"param_{k}": v for k, v in best.items()},
        })
        print(f"   OUTER test PR-AUC={rows[-1]['pr_auc']:.4f}  "
              f"ROC-AUC={rows[-1]['roc_auc']:.4f}\n", flush=True)
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

    res = pd.DataFrame(rows)
    print("=== tuned logistic-WoE, nested selection ===")
    print(res[["fold", "chosen_cfg", "inner_pr_auc", "pr_auc", "roc_auc", "ks"]]
          .round(4).to_string(index=False))
    print(f"\nmean outer PR-AUC {res['pr_auc'].mean():.4f} (sd {res['pr_auc'].std():.4f})")

    base = pd.read_csv("reports/baselines.csv")
    b = base.pivot(index="fold", columns="model", values="pr_auc")
    b.index = b.index.astype(int)
    t = res.set_index("fold")["pr_auc"]

    print("\npaired comparisons (identical test sets):")
    d = (t - b["logistic_woe"]).dropna()
    print(f"   tuned logistic - untuned logistic : "
          + ", ".join(f"{v:+.4f}" for v in d.values)
          + f"   mean {d.mean():+.4f}  paired sd {d.std():.4f}  "
            f"same sign: {bool((d > 0).all() or (d < 0).all())}")

    # The comparison this module exists to make fair: both models tuned, same
    # nesting, same search budget, same folds.
    gbm = pd.read_csv("reports/tuning.csv").set_index("fold")["pr_auc"]
    d2 = (gbm - t).dropna()
    print(f"   tuned LightGBM - tuned logistic   : "
          + ", ".join(f"{v:+.4f}" for v in d2.values)
          + f"   mean {d2.mean():+.4f}  paired sd {d2.std():.4f}  "
            f"same sign: {bool((d2 > 0).all() or (d2 < 0).all())}")
    print(f"\nwrote {OUT_PATH}")
