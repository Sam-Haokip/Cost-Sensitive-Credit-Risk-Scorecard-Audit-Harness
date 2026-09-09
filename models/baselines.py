"""
Phase 3: baselines before complexity, evaluated under the honest regime.

Four models, in deliberate order of increasing sophistication, each scored on
the same embargoed walk-forward folds Phase 2 established:

  trivial      -- predict the majority class. Exists to make the point that
                  accuracy is meaningless here: it scores ~90% while catching
                  exactly zero defaults, and its PR-AUC is the base rate.
  logistic_woe -- logistic regression on Weight-of-Evidence encoded features.
                  The credit-industry standard and this project's interpretable
                  reference model.
  logistic_raw -- logistic regression with the naive preprocessing most people
                  reach for: median-impute numerics, one-hot categoricals.
                  Included ONLY so the WoE choice is a measured decision rather
                  than an appeal to industry convention -- and because Phase 1
                  predicted this specific pipeline would damage the six
                  `mths_since_*` columns by imputing "never happened" as a
                  middling value.
  lightgbm     -- gradient boosting, untuned at this stage. Hyperparameter
                  search on the temporal scheme comes next.

WHY EMBARGOED FOLDS
-------------------
Phase 2 measured three ways to split and found the embargoed regime -- train
only on cohorts whose 18-month outcomes were already known when the model would
have been fitted -- costs -0.0131 PR-AUC against a matched random baseline.
That is the honest number, so it is the one baselines are held to. Scoring them
under a friendlier scheme would inflate every row of the table equally and
teach nothing.

Everything is fit inside the fold, including the WoE bins and the imputation
medians, because both are estimated from the target or the feature distribution
and would leak if fit on the full frame first.
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.dataset import CATEGORICAL, FEATURE_COLS, load_modelling_frame
from features.woe import WOEEncoder
from models.temporal_validation import EMBARGO_MONTHS, SEED, TEST_CAP, TRAIN_CAP, _cap

TEST_YEARS = [2014, 2015, 2016, 2017]
OUT_PATH = "reports/baselines.csv"

NUMERIC_FEATURES = [c for c in FEATURE_COLS if c not in CATEGORICAL]
CATEGORICAL_FEATURES = [c for c in FEATURE_COLS if c in CATEGORICAL]


def ks_statistic(y_true, scores) -> float:
    """Kolmogorov-Smirnov separation: the largest gap between the cumulative
    score distributions of defaults and non-defaults. Standard in credit
    scorecards, where it is often quoted ahead of AUC."""
    y_true = np.asarray(y_true)
    pos = np.sort(scores[y_true == 1])
    neg = np.sort(scores[y_true == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    grid = np.union1d(pos, neg)
    cdf_pos = np.searchsorted(pos, grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, grid, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def score(name, fold, y_true, proba, n_train):
    r = {
        "model": name,
        "fold": fold,
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "ks": ks_statistic(y_true, proba),
        "n_train": n_train,
        "base_rate": float(np.mean(y_true)),
    }
    print(f"  {name:14s} {fold}  PR-AUC={r['pr_auc']:.4f}  ROC-AUC={r['roc_auc']:.4f}  "
          f"Brier={r['brier']:.4f}  KS={r['ks']:.4f}", flush=True)
    return r


# ---- the four models -------------------------------------------------------
def fit_trivial(train, test):
    """Majority class for everyone. Constant score = the training base rate."""
    p = np.full(len(test), train["default_window"].mean())
    return p


def fit_logistic_woe(train, test):
    enc = WOEEncoder()
    Xtr = enc.fit_transform(train[FEATURE_COLS], train["default_window"])
    Xte = enc.transform(test[FEATURE_COLS])
    model = LogisticRegression(max_iter=2000, random_state=SEED)
    model.fit(Xtr, train["default_window"])
    return model.predict_proba(Xte)[:, 1]


def fit_logistic_raw(train, test):
    """The naive pipeline: median-impute numerics, one-hot categoricals."""
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))]),
         CATEGORICAL_FEATURES),
    ])
    model = Pipeline([("pre", pre),
                      ("lr", LogisticRegression(max_iter=2000, random_state=SEED))])
    tr = train[FEATURE_COLS].copy()
    te = test[FEATURE_COLS].copy()
    for c in CATEGORICAL_FEATURES:
        tr[c] = tr[c].astype("object")
        te[c] = te[c].astype("object")
    model.fit(tr, train["default_window"])
    return model.predict_proba(te)[:, 1]


def fit_lightgbm(train, test):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        random_state=SEED, n_jobs=2, verbose=-1,
    )
    model.fit(train[FEATURE_COLS], train["default_window"],
              categorical_feature=CATEGORICAL_FEATURES)
    return model.predict_proba(test[FEATURE_COLS])[:, 1]


MODELS = {
    "trivial": fit_trivial,
    "logistic_woe": fit_logistic_woe,
    "logistic_raw": fit_logistic_raw,
    "lightgbm": fit_lightgbm,
}


def embargoed_folds(df):
    for year in TEST_YEARS:
        test = _cap(df[df["issue_year"] == year], TEST_CAP)
        cutoff = pd.Timestamp(year=year - (EMBARGO_MONTHS // 12), month=1, day=1)
        train = _cap(df[df["issue_d"] < cutoff], TRAIN_CAP)
        if len(train) < 5000 or len(test) < 5000:
            continue
        yield year, train, test


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    print("loading modelling frame...")
    df = load_modelling_frame()
    print(f"{len(df):,} loans, base rate {df['default_window'].mean():.2%}\n")
    print("embargoed walk-forward folds (the honest regime from Phase 2):\n")

    rows = []
    for year, train, test in embargoed_folds(df):
        print(f"fold {year}  (train {len(train):,} -> test {len(test):,})")
        for name, fn in MODELS.items():
            proba = fn(train, test)
            rows.append(score(name, year, test["default_window"].values, proba, len(train)))
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
        print()

    res = pd.DataFrame(rows)
    summary = (res.groupby("model")[["pr_auc", "roc_auc", "brier", "ks"]]
               .agg(["mean", "std"]).round(4))
    print("=== across folds (mean / sd) ===")
    print(summary.to_string())

    print()
    print("=== why accuracy is the wrong metric ===")
    base = res[res.model == "trivial"]["base_rate"].mean()
    print(f"the trivial model predicts 'no default' for everyone:")
    print(f"   accuracy      : {1 - base:.1%}   <- looks excellent")
    print(f"   defaults caught: 0 of {base:.1%} of applicants   <- catches nothing")
    print(f"   PR-AUC        : {res[res.model=='trivial']['pr_auc'].mean():.4f}  "
          f"(= the base rate, the floor any real model must beat)")

    print(f"\nwrote {OUT_PATH}")
