"""
Phase 2: walk-forward validation, and what random splitting was hiding.

Three regimes, same model, same metrics, same sample sizes -- so the only thing
varying is how the data is split:

  random     -- shuffle everything, 80/20. What most portfolio projects do.
                Lets the model learn from 2016 loans to predict 2014 ones.

  temporal   -- train on every cohort before the test year. The usual "time-based
                split", and a real improvement over random.

  embargoed  -- train only on cohorts whose OUTCOMES were already known at the
                moment the model would have been fitted. This is the honest one,
                and it is the regime most implementations skip.

Why the embargo matters. Suppose you are underwriting in January 2016. You do
not yet know the 18-month outcome of a loan issued in 2015 -- it has not matured,
and will not until mid-2017. So "train on everything up to last year" is still
using information you could not have had. A loan issued in month M only has a
known windowed outcome at M + WINDOW + LAG = M + 24 months. The embargoed regime
enforces that: to predict year Y, train only on loans issued before January of
year Y-2.

Sample sizes are capped for compute, identically across all three regimes, so
the comparison stays fair. Results are written incrementally to
reports/temporal_validation.csv.
"""
import os

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from data.dataset import FEATURE_COLS, load_modelling_frame
from data.vintage_target import CHARGEOFF_LAG_MONTHS, WINDOW_MONTHS

TEST_YEARS = [2014, 2015, 2016, 2017]
EMBARGO_MONTHS = WINDOW_MONTHS + CHARGEOFF_LAG_MONTHS  # 24

# Capped for memory/compute. Applied identically to every regime.
TRAIN_CAP = 150_000
TEST_CAP = 100_000
SEED = 42
OUT_PATH = "reports/temporal_validation.csv"


def _cap(df, n, seed=SEED):
    if len(df) <= n:
        return df
    keep, _ = train_test_split(df, train_size=n, random_state=seed,
                               stratify=df["default_window"])
    return keep


def fit_score(train, test):
    model = HistGradientBoostingClassifier(
        categorical_features="from_dtype", max_iter=150, random_state=SEED
    )
    model.fit(train[FEATURE_COLS], train["default_window"])
    p = model.predict_proba(test[FEATURE_COLS])[:, 1]
    y = test["default_window"]
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "n_train": len(train),
        "n_test": len(test),
        "test_base_rate": y.mean(),
    }


def run(df):
    rows = []

    def record(regime, fold, res):
        rows.append(dict(regime=regime, fold=fold, **res))
        print(f"  {regime:10s} {str(fold):>6}  PR-AUC={res['pr_auc']:.4f}  "
              f"ROC-AUC={res['roc_auc']:.4f}  Brier={res['brier']:.4f}  "
              f"n_train={res['n_train']:,}  base={res['test_base_rate']:.2%}", flush=True)
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)  # incremental save

    print("temporal + embargoed folds:")
    for year in TEST_YEARS:
        test = _cap(df[df["issue_year"] == year], TEST_CAP)
        if len(test) < 5000 or test["default_window"].nunique() < 2:
            print(f"  skipping {year}: too few loans")
            continue

        naive_train = _cap(df[df["issue_year"] < year], TRAIN_CAP)
        cutoff = pd.Timestamp(year=year - (EMBARGO_MONTHS // 12), month=1, day=1)
        emb_train = _cap(df[df["issue_d"] < cutoff], TRAIN_CAP)

        if len(naive_train) >= 5000:
            record("temporal", year, fit_score(naive_train, test))
        if len(emb_train) >= 5000:
            record("embargoed", year, fit_score(emb_train, test))

    # Random regime: same sizes, ignoring time entirely. One run per fold so the
    # spread is comparable to the temporal regimes' fold-to-fold spread.
    print("random folds (same sizes, time ignored):")
    for i, year in enumerate(TEST_YEARS):
        tr, te = train_test_split(df, train_size=TRAIN_CAP, test_size=TEST_CAP,
                                  random_state=100 + i, stratify=df["default_window"])
        record("random", f"r{i+1}", fit_score(tr, te))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    print("loading modelling frame...")
    df = load_modelling_frame()
    print(f"{len(df):,} eligible loans, base rate {df['default_window'].mean():.2%}\n")

    results = run(df)

    print()
    summary = (results.groupby("regime")["pr_auc"]
               .agg(folds="size", mean="mean", sd="std", lo="min", hi="max").round(4))
    print("PR-AUC by regime:")
    print(summary.to_string())

    if {"random", "temporal", "embargoed"} <= set(summary.index):
        print()
        print(f"random  -> temporal : {summary.loc['temporal','mean'] - summary.loc['random','mean']:+.4f}")
        print(f"temporal-> embargoed: {summary.loc['embargoed','mean'] - summary.loc['temporal','mean']:+.4f}")
        print(f"random  -> embargoed: {summary.loc['embargoed','mean'] - summary.loc['random','mean']:+.4f}  (total inflation)")
    print(f"\nwrote {OUT_PATH}")
