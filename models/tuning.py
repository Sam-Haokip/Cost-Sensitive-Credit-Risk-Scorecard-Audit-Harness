"""
Phase 3: hyperparameter search for LightGBM, on the temporal scheme.

WHY NESTED, AND WHY IT MATTERS HERE
-----------------------------------
The obvious way to tune is to try configurations on the walk-forward folds and
keep whichever scores best. That number is then optimistically biased: the folds
have been used to *choose* the model, so they are no longer a clean estimate of
how it generalises. With an effect this small -- the untuned LightGBM beat
logistic regression by +0.0011, a gap whose sign flips across folds -- a bias of
that size is easily enough to manufacture a winner that isn't one.

So the search is nested. Within each outer fold:

    outer test      = cohort year Y                      (never touched by tuning)
    outer train     = embargoed window (issued < Jan Y-2)
      inner val     = the latest cohort year inside outer train
      inner train   = everything earlier

Configurations are ranked on inner validation only. The winner is refit on the
full outer training window and scored once on the outer test. Tuning therefore
never sees the data the reported number comes from, and the inner split is
itself temporal rather than random, so the selection respects the same ordering
constraint the outer evaluation does.

The search is deliberately small (8 configurations). The question is whether
tuning changes the Phase 3 conclusion that gradient boosting fails to separate
from logistic regression -- not to squeeze out a leaderboard score.

ONE LIMITATION, STATED RATHER THAN HIDDEN
-----------------------------------------
The inner split respects temporal ordering but does *not* re-apply the 24-month
embargo. It cannot: the cohorts are 251 / 1,562 / 4,716 / 11,536 / 21,721 loans
for 2007-2011, so embargoing the inner split of the 2014 fold would leave 1,813
loans and roughly 175 defaults to choose 8 configurations on -- selection noise
would swamp the signal being selected on. The consequence is that configurations
are ranked under a slightly easier regime than the one they are then scored in,
which biases selection toward whatever suits a shorter distributional gap. The
direction of that bias is unknown; the alternative was a selection step with no
statistical power at all.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from data.dataset import CATEGORICAL, FEATURE_COLS, load_modelling_frame
from models.baselines import TEST_YEARS, ks_statistic
from models.temporal_validation import EMBARGO_MONTHS, SEED, TEST_CAP, TRAIN_CAP, _cap

OUT_PATH = "reports/tuning.csv"
CATEGORICAL_FEATURES = [c for c in FEATURE_COLS if c in CATEGORICAL]

# Small, hand-spread grid over the parameters that actually move a GBM on
# tabular data: capacity (num_leaves, min_child_samples), regularisation
# (reg_lambda, feature/bagging fraction), and the rate/length trade-off.
PARAM_GRID = [
    dict(num_leaves=15,  learning_rate=0.05, n_estimators=300, min_child_samples=100, reg_lambda=1.0),
    dict(num_leaves=31,  learning_rate=0.05, n_estimators=300, min_child_samples=50,  reg_lambda=1.0),
    dict(num_leaves=31,  learning_rate=0.03, n_estimators=600, min_child_samples=100, reg_lambda=5.0),
    dict(num_leaves=63,  learning_rate=0.03, n_estimators=600, min_child_samples=200, reg_lambda=5.0),
    dict(num_leaves=15,  learning_rate=0.03, n_estimators=800, min_child_samples=200, reg_lambda=10.0),
    dict(num_leaves=63,  learning_rate=0.05, n_estimators=300, min_child_samples=500, reg_lambda=10.0),
    dict(num_leaves=127, learning_rate=0.02, n_estimators=800, min_child_samples=500, reg_lambda=20.0),
    dict(num_leaves=31,  learning_rate=0.02, n_estimators=1000, min_child_samples=300, reg_lambda=20.0),
]


def fit_lgbm(train, valid, params):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        random_state=SEED, n_jobs=2, verbose=-1,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1, **params
    )
    model.fit(train[FEATURE_COLS], train["default_window"],
              categorical_feature=CATEGORICAL_FEATURES)
    p = model.predict_proba(valid[FEATURE_COLS])[:, 1]
    return model, p


def outer_folds(df):
    """Outer: embargoed train -> cohort-year test. Inner: latest year of the
    embargoed window held out, everything before it used for fitting."""
    for year in TEST_YEARS:
        test = _cap(df[df["issue_year"] == year], TEST_CAP)
        cutoff = pd.Timestamp(year=year - (EMBARGO_MONTHS // 12), month=1, day=1)
        outer_train = df[df["issue_d"] < cutoff]
        if len(outer_train) < 10_000 or len(test) < 5000:
            continue

        inner_val_year = int(outer_train["issue_year"].max())
        inner_train = outer_train[outer_train["issue_year"] < inner_val_year]
        inner_val = outer_train[outer_train["issue_year"] == inner_val_year]
        if len(inner_train) < 5000 or len(inner_val) < 2000:
            continue

        yield (year, _cap(outer_train, TRAIN_CAP), test,
               _cap(inner_train, TRAIN_CAP), _cap(inner_val, TEST_CAP), inner_val_year)


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
            _, p = fit_lgbm(inner_train, inner_val, params)
            ap = average_precision_score(inner_val["default_window"], p)
            scores.append(ap)
            print(f"   cfg {i}: leaves={params['num_leaves']:>3} lr={params['learning_rate']} "
                  f"n={params['n_estimators']:>4}  inner PR-AUC={ap:.4f}", flush=True)

        best_i = int(np.argmax(scores))
        best = PARAM_GRID[best_i]
        spread = max(scores) - min(scores)
        print(f"   -> selected cfg {best_i} on INNER validation only "
              f"(grid spread {spread:.4f} PR-AUC across 8 configurations)")

        _, p_test = fit_lgbm(outer_train, test, best)
        y = test["default_window"]
        row = {
            "fold": year, "chosen_cfg": best_i, "inner_pr_auc": scores[best_i],
            "inner_spread": spread,
            "pr_auc": average_precision_score(y, p_test),
            "roc_auc": roc_auc_score(y, p_test),
            "brier": brier_score_loss(y, p_test),
            "ks": ks_statistic(y.values, p_test),
            "n_inner_train": len(inner_train), "n_outer_train": len(outer_train),
            **{f"param_{k}": v for k, v in best.items()},
        }
        rows.append(row)
        print(f"   OUTER test PR-AUC={row['pr_auc']:.4f}  ROC-AUC={row['roc_auc']:.4f}  "
              f"KS={row['ks']:.4f}\n", flush=True)
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

    res = pd.DataFrame(rows)
    print("=== tuned LightGBM, nested selection ===")
    print(res[["fold", "chosen_cfg", "inner_pr_auc", "pr_auc", "roc_auc", "ks"]].round(4).to_string(index=False))
    print()
    print(f"mean outer PR-AUC {res['pr_auc'].mean():.4f} (sd {res['pr_auc'].std():.4f})")
    print(f"configurations chosen: {sorted(res['chosen_cfg'].tolist())} "
          f"-- {'stable' if res['chosen_cfg'].nunique() == 1 else 'they differ by fold, which is itself informative'}")

    # Paired against the untuned baselines, same folds, same test sets.
    base = pd.read_csv("reports/baselines.csv")
    b = base.pivot(index="fold", columns="model", values="pr_auc")
    t = res.set_index("fold")["pr_auc"]
    b.index = b.index.astype(int)
    print()
    print("paired against Phase 3 baselines (identical test sets):")
    for m in ["lightgbm", "logistic_woe"]:
        if m in b.columns:
            d = (t - b[m]).dropna()
            print(f"   tuned - {m:13s}: " + ", ".join(f"{v:+.4f}" for v in d.values)
                  + f"   mean {d.mean():+.4f}  paired sd {d.std():.4f}  "
                    f"same sign: {bool((d > 0).all() or (d < 0).all())}")
    print(f"\nwrote {OUT_PATH}")
