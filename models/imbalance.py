"""
Phase 3: does anything you do about class imbalance actually help?

The default reflex on a 9.6% positive rate is to "fix the imbalance" -- class
weights, undersampling, or SMOTE. This module measures four treatments on the
same embargoed folds, holding the model fixed, so the answer is a number rather
than a habit.

    none          -- the model as fitted in Phase 3.
    class_weight  -- reweight the loss by inverse class frequency.
    undersample   -- randomly drop majority rows until the classes are balanced.
    smote         -- synthesise minority rows by interpolating between a point
                     and its k nearest minority neighbours.

WHAT TO EXPECT, AND WHY IT IS WORTH MEASURING ANYWAY
----------------------------------------------------
None of these should move a *ranking* metric much, and there is a reason.
PR-AUC, ROC-AUC and KS all depend only on the ORDER of the scores. Class
weighting and resampling change the base rate the model is fitted to, which
shifts the whole score distribution up or down -- a monotone transformation --
and a monotone transformation leaves the ranking untouched. What they can do is
change which splits a tree finds worth making, so the effect is second-order and
noisy, not zero.

They are not free, though, and the cost lands on the metric ranking hides:

  * Brier score and calibration. A model trained on artificially balanced data
    predicts as though defaults were 50% of the book. Those probabilities are
    wrong by construction, and Phase 4 (calibration) and Phase 5 (cost-based
    thresholds) both need probabilities that mean what they say. Brier is
    reported here precisely to catch that.
  * Data thrown away. Balanced undersampling on this book keeps every default
    and roughly an equal number of non-defaults -- discarding about 80% of the
    training rows, which is why it is expected to be the worst of the four.

SMOTE has a further problem specific to this data. It interpolates in feature
space, which presumes the space is continuous and metric. Of the 97 features,
10 are categorical and dozens are counts, zero-inflated, or -- per Phase 1 --
missing precisely *because an event never happened*. Interpolating halfway
between "no delinquency on record" and "delinquent 14 months ago" invents a
borrower who does not exist and cannot exist. The measurement below tests that
argument rather than asserting it.

To keep the comparison to one variable, every treatment is applied to the same
LightGBM configuration, and it is the untuned Phase 3 baseline rather than the
tuned one, so imbalance handling is never entangled with the tuning result.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from data.dataset import CATEGORICAL, FEATURE_COLS, load_modelling_frame
from models.baselines import embargoed_folds, ks_statistic
from models.temporal_validation import SEED

OUT_PATH = "reports/imbalance.csv"
CATEGORICAL_FEATURES = [c for c in FEATURE_COLS if c in CATEGORICAL]
NUMERIC_FEATURES = [c for c in FEATURE_COLS if c not in CATEGORICAL]

BASE = dict(num_leaves=31, learning_rate=0.05, n_estimators=300,
            min_child_samples=20, reg_lambda=0.0)


def _fit(train, test, sample_weight=None, class_weight=None):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(random_state=SEED, n_jobs=2, verbose=-1,
                               class_weight=class_weight, **BASE)
    model.fit(train[FEATURE_COLS], train["default_window"],
              sample_weight=sample_weight,
              categorical_feature=CATEGORICAL_FEATURES)
    return model.predict_proba(test[FEATURE_COLS])[:, 1]


def treat_none(train, test):
    return _fit(train, test), len(train)


def treat_class_weight(train, test):
    """Reweight the loss so the two classes contribute equally. No rows are
    created or destroyed -- only the gradient each row contributes."""
    return _fit(train, test, class_weight="balanced"), len(train)


def treat_undersample(train, test):
    """Keep every default, sample an equal number of non-defaults."""
    rng = np.random.default_rng(SEED)
    pos = train.index[train["default_window"] == 1]
    neg = train.index[train["default_window"] == 0]
    keep_neg = rng.choice(neg, size=min(len(pos), len(neg)), replace=False)
    sub = train.loc[np.concatenate([pos.values, keep_neg])]
    return _fit(sub, test), len(sub)


def treat_smote_rounded(train, test):
    """SMOTE-NC, then round every feature that is whole-numbered in real data.

    The obvious repair for the diagnostic in models/smote_diagnostic.py, which
    found synthetic rows perfectly separable (ROC-AUC 1.0000) because
    interpolation puts fractional values into 51 of 80 integer-valued features.
    Rounding removes that specific tell. It is included so the negative result
    survives the obvious objection rather than resting on an artefact anyone
    could have patched in one line.
    """
    return treat_smote(train, test, round_integers=True)


def treat_smote(train, test, round_integers=False):
    """SMOTE-NC: interpolate numeric features, majority-vote categoricals.

    Plain SMOTE cannot run here -- it interpolates every column, which is
    meaningless for the 10 categorical features. SMOTE-NC is the variant
    designed for mixed types and is the fair version of the comparison; using
    plain SMOTE would be attacking a weaker opponent than the one people
    actually reach for.
    """
    from imblearn.over_sampling import SMOTENC

    X = train[FEATURE_COLS].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("object").fillna("__missing__")
    # SMOTE-NC cannot handle NaN in the numeric block: it computes distances,
    # and a distance to an unknown value is undefined. So every missing value
    # must be replaced before the technique can run at all.
    #
    # This is the first cost, and it is not small. Phase 1 showed six
    # `mths_since_*` columns are missing precisely because the event never
    # happened; median-imputing them rewrites "never delinquent" as "delinquent
    # a middling time ago". Worse, between 29 and 66 of the 87 numeric features
    # are *entirely* null inside a given embargoed training window (the bureau
    # fields LendingClub only began reporting in 2016), so their median is
    # itself undefined and they must be filled with a flat constant.
    #
    # The tree models need none of this: LightGBM routes missing values down
    # their own branch and logistic-WoE gives them their own bin. SMOTE alone
    # forces the data to be invented before it can be resampled.
    med = X[NUMERIC_FEATURES].median()
    all_null = [c for c in NUMERIC_FEATURES if med[c] != med[c]]  # NaN median
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(med).fillna(0.0)
    print(f"      [smote] imputed {len(all_null)} of {len(NUMERIC_FEATURES)} numeric "
          f"features with a constant -- no observed value in this window",
          flush=True)

    idx = [X.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    sm = SMOTENC(categorical_features=idx, random_state=SEED, k_neighbors=5)
    Xr, yr = sm.fit_resample(X, train["default_window"])

    res = pd.DataFrame(Xr, columns=X.columns)
    for c in NUMERIC_FEATURES:
        res[c] = pd.to_numeric(res[c], errors="coerce")

    if round_integers:
        # Judged on the RAW column, not the imputed one. Post-imputation this
        # would count the all-null features (filled with a flat 0.0) as
        # integer-valued, and could miss a real integer column whose median
        # happens to be fractional.
        integral = [c for c in NUMERIC_FEATURES
                    if train[c].notna().any()
                    and pd.to_numeric(train[c], errors="coerce").dropna().mod(1).eq(0).all()]
        res[integral] = res[integral].round()
        print(f"      [smote_rounded] rounded {len(integral)} integer-valued features "
              f"(of {sum(train[c].notna().any() for c in NUMERIC_FEATURES)} observed "
              f"in this window)", flush=True)

    res["default_window"] = np.asarray(yr).astype(int)

    te = test[FEATURE_COLS].copy()
    for c in CATEGORICAL_FEATURES:
        te[c] = te[c].astype("object").fillna("__missing__")
    te[NUMERIC_FEATURES] = te[NUMERIC_FEATURES].fillna(med).fillna(0.0)

    # LightGBM reads pandas categoricals by their integer CODES, so train and
    # test must share one category list. Building them independently silently
    # maps different strings to the same code -- the same class of bug Phase 1
    # hit when concatenating frames with divergent category sets.
    for c in CATEGORICAL_FEATURES:
        dtype = pd.CategoricalDtype(
            categories=sorted(set(res[c].astype(str)) | set(te[c].astype(str))))
        res[c] = res[c].astype(str).astype(dtype)
        te[c] = te[c].astype(str).astype(dtype)

    import lightgbm as lgb
    model = lgb.LGBMClassifier(random_state=SEED, n_jobs=2, verbose=-1, **BASE)
    model.fit(res[FEATURE_COLS], res["default_window"])
    return model.predict_proba(te[FEATURE_COLS])[:, 1], len(res)


TREATMENTS = {
    "none": treat_none,
    "class_weight": treat_class_weight,
    "undersample": treat_undersample,
    "smote": treat_smote,
    "smote_rounded": treat_smote_rounded,
}


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    print("loading modelling frame...")
    df = load_modelling_frame()
    print(f"{len(df):,} loans, base rate {df['default_window'].mean():.2%}\n")

    rows = []
    for year, train, test in embargoed_folds(df):
        print(f"fold {year}  (train {len(train):,} -> test {len(test):,}, "
              f"train base rate {train['default_window'].mean():.2%})")
        for name, fn in TREATMENTS.items():
            try:
                proba, n_used = fn(train, test)
            except Exception as exc:      # keep the other treatments reportable
                print(f"   {name:13s} FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            y = test["default_window"].values
            r = {
                "treatment": name, "fold": year,
                "pr_auc": average_precision_score(y, proba),
                "roc_auc": roc_auc_score(y, proba),
                "brier": brier_score_loss(y, proba),
                "ks": ks_statistic(y, proba),
                "mean_pred": float(np.mean(proba)),
                "true_rate": float(np.mean(y)),
                "n_train_rows": n_used,
            }
            rows.append(r)
            print(f"   {name:13s} PR-AUC={r['pr_auc']:.4f}  Brier={r['brier']:.4f}  "
                  f"mean predicted p={r['mean_pred']:.3f}  rows={n_used:,}", flush=True)
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
        print()

    res = pd.DataFrame(rows)
    print("=== across folds (mean / sd) ===")
    print(res.groupby("treatment")[["pr_auc", "roc_auc", "brier", "ks", "mean_pred"]]
          .agg(["mean", "std"]).round(4).to_string())

    print()
    print("=== paired against no treatment (identical test sets) ===")
    wide = res.pivot(index="fold", columns="treatment", values="pr_auc")
    for t in [c for c in wide.columns if c != "none"]:
        d = (wide[t] - wide["none"]).dropna()
        print(f"   {t:13s} " + ", ".join(f"{x:+.4f}" for x in d.values)
              + f"   mean {d.mean():+.4f}  paired sd {d.std():.4f}  "
                f"same sign: {bool((d > 0).all() or (d < 0).all())}")

    print()
    print("=== the cost that ranking metrics hide ===")
    print("mean predicted probability against the fold's actual default rate.")
    print("a model fitted on rebalanced data believes defaults are far more")
    print("common than they are, which is exactly what Phases 4 and 5 need not")
    print("to be true.\n")
    truth = res.groupby("fold")["true_rate"].first()
    for t, g in res.groupby("treatment"):
        ratio = (g.set_index("fold")["mean_pred"] / truth).mean()
        print(f"   {t:13s} predicted {g['mean_pred'].mean():.3f} vs actual "
              f"{truth.mean():.3f}   ({ratio:.1f}x)   Brier {g['brier'].mean():.4f}")
    print(f"\nwrote {OUT_PATH}")
