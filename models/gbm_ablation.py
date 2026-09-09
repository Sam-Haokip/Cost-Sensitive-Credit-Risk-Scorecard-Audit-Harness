"""
Phase 3: which knob actually moved the gradient booster?

The nested search (models/tuning.py) gained +0.0059 PR-AUC over the untuned
LightGBM, positive on all four folds. Left there, the finding is "tuning
helped", which explains nothing and is the kind of claim this project exists
to avoid.

Look at what the search chose. On every fold it picked a configuration with
`num_leaves=15` -- half the untuned default of 31 -- and `min_child_samples`
of 100 or 200 against a default of 20. It also had two knobs the untuned model
never had: `subsample=0.8` and `colsample_bytree=0.8`, which are on for every
configuration in the grid and so were never actually *selected*, merely
inherited.

So "tuned - untuned" is a confounded difference. It bundles:

    (a) row/column bagging, which no grid entry varied,
    (b) lower capacity: fewer leaves, larger minimum leaf size,
    (c) explicit L2 (reg_lambda 1-10 against a default of 0).

This module separates (a) from (b)+(c) with two extra fits per fold:

    baseline      leaves=31 lr=0.05 n=300 mcs=20  reg=0   no bagging   [Phase 3]
    bagging_only  identical, plus subsample=0.8 colsample=0.8
    capacity_only cfg 0 (leaves=15 mcs=100 reg=1.0), no bagging
    tuned         cfg 0 with bagging                                   [tuning.py]

If `bagging_only` recovers most of the gain, the search bought variance
reduction that a one-line default change would also have bought, and the
honest write-up says so. If `capacity_only` does, the story is overfitting:
the early training windows carry roughly 18,000 loans and, per Phase 1, 67 of
97 features are entirely null before 2016, so a 31-leaf tree has far more
capacity than the data supports.

Fold 2014 uses cfg 0 as the reference configuration because the search chose
it on the two earliest folds; cfg 4 (leaves=15, lr=0.03, n=800, mcs=200,
reg=10) is scored too, so the comparison does not rest on one arbitrary pick.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from data.dataset import CATEGORICAL, FEATURE_COLS, load_modelling_frame
from models.baselines import embargoed_folds, ks_statistic
from models.temporal_validation import SEED

OUT_PATH = "reports/gbm_ablation.csv"
CATEGORICAL_FEATURES = [c for c in FEATURE_COLS if c in CATEGORICAL]

BAGGING = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8)

# The four Phase 3 baseline knobs, held fixed except where a variant changes them.
BASE = dict(num_leaves=31, learning_rate=0.05, n_estimators=300,
            min_child_samples=20, reg_lambda=0.0)
CFG0 = dict(num_leaves=15, learning_rate=0.05, n_estimators=300,
            min_child_samples=100, reg_lambda=1.0)
CFG4 = dict(num_leaves=15, learning_rate=0.03, n_estimators=800,
            min_child_samples=200, reg_lambda=10.0)

VARIANTS = {
    "baseline":       {**BASE},                 # reproduces models/baselines.py
    "bagging_only":   {**BASE, **BAGGING},      # (a) alone
    "capacity_only":  {**CFG0},                 # (b)+(c) alone
    "tuned_cfg0":     {**CFG0, **BAGGING},      # both -- what tuning.py fitted
    "capacity_only4": {**CFG4},
    "tuned_cfg4":     {**CFG4, **BAGGING},
}


def fit_score(train, test, params):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(random_state=SEED, n_jobs=2, verbose=-1, **params)
    model.fit(train[FEATURE_COLS], train["default_window"],
              categorical_feature=CATEGORICAL_FEATURES)
    p = model.predict_proba(test[FEATURE_COLS])[:, 1]
    y = test["default_window"].values
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "ks": ks_statistic(y, p),
    }


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    print("loading modelling frame...")
    df = load_modelling_frame()
    print(f"{len(df):,} loans\n")

    rows = []
    for year, train, test in embargoed_folds(df):
        print(f"fold {year}  (train {len(train):,} -> test {len(test):,})")
        for name, params in VARIANTS.items():
            m = fit_score(train, test, params)
            rows.append({"variant": name, "fold": year, **m})
            print(f"   {name:15s} PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}",
                  flush=True)
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
        print()

    res = pd.DataFrame(rows)
    wide = res.pivot(index="fold", columns="variant", values="pr_auc")

    print("=== PR-AUC by fold ===")
    print(wide.round(4).to_string())
    print()
    print("=== paired against the untuned baseline (same folds, same test sets) ===")
    for v in ["bagging_only", "capacity_only", "tuned_cfg0", "capacity_only4", "tuned_cfg4"]:
        d = (wide[v] - wide["baseline"]).dropna()
        print(f"   {v:15s} " + ", ".join(f"{x:+.4f}" for x in d.values)
              + f"   mean {d.mean():+.4f}  paired sd {d.std():.4f}  "
                f"same sign: {bool((d > 0).all() or (d < 0).all())}")

    print()
    print("=== attribution ===")
    total = (wide["tuned_cfg0"] - wide["baseline"]).mean()
    bag = (wide["bagging_only"] - wide["baseline"]).mean()
    cap = (wide["capacity_only"] - wide["baseline"]).mean()
    print(f"   total gain, cfg 0 with bagging : {total:+.4f}")
    print(f"     of which bagging alone       : {bag:+.4f}  ({bag / total:.0%} of total)"
          if abs(total) > 1e-9 else "")
    print(f"     of which capacity/L2 alone   : {cap:+.4f}  ({cap / total:.0%} of total)"
          if abs(total) > 1e-9 else "")
    print(f"     interaction (non-additivity) : {total - bag - cap:+.4f}")
    print(f"\nwrote {OUT_PATH}")
