"""
Phase 4: are the predicted probabilities worth anything as probabilities?

Everything so far has been scored on *ranking* -- PR-AUC, ROC-AUC, KS all depend
only on the order of the scores. Phase 5 cannot use a ranking. Choosing a
cut-off by expected cost needs a number that means what it says: if the model
says 8%, roughly 8 in 100 of those applicants must actually default, or the
threshold is optimising against a fiction.

Phase 3 already showed why this ordering matters. Three of the four imbalance
treatments left ranking almost untouched while destroying the probability scale
-- undersampling predicted 4.8x the true default rate. Had calibration been
measured first and ranking second, "undersampling barely hurts" would have been
the recorded conclusion.

WHAT IS MEASURED
----------------
Two models (the WoE scorecard and LightGBM), three treatments:

    none      -- the model's raw predict_proba output.
    platt     -- a logistic regression fitted to the log-odds of those
                 predictions. Two parameters, so it can shift and stretch the
                 scale but cannot change the ranking at all.
    isotonic  -- a monotone step function fitted to the predictions. Far more
                 flexible, and correspondingly easier to overfit; it can also
                 merge distinct scores into ties, which is the one way a
                 calibrator can move a ranking metric.

HOW THE SPLIT WORKS, AND WHY IT COSTS SOMETHING
-----------------------------------------------
A calibrator fitted on the data the model was fitted on learns the model's
training-set overconfidence rather than its true miscalibration. So the
embargoed training window is split temporally, exactly as the Phase 3
hyperparameter search splits it:

    model fit    = the embargoed window, minus its most recent cohort year
    calibrator   = that most recent cohort year
    test         = the held-out cohort year, two or more years later

All three treatments share the same underlying model, so differences between
them isolate the calibrator and nothing else. The cost is real and is reported:
the model sees less data than the Phase 3 baselines did, so these Brier scores
are not directly comparable to that table.

THE QUESTION THIS PROJECT CAN ASK THAT A TEXTBOOK CANNOT
--------------------------------------------------------
Calibration is normally taught as if the world were stationary. Here it is not.
The calibrator is fitted on one cohort and applied to a cohort at least two
years later, across a period where Phase 2 measured significant covariate drift
in 18 of 97 features and the default rate itself moves between cohorts. So the
same calibrator is scored twice -- on the cohort it was fitted on, and on the
test cohort -- and the gap between those two numbers is the part of calibration
that does not survive time.
"""
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from data.dataset import CATEGORICAL, FEATURE_COLS, load_modelling_frame
from features.woe import WOEEncoder
from models.temporal_validation import SEED
from models.tuning import outer_folds

OUT_PATH = "reports/calibration.csv"
CURVE_PATH = "reports/reliability_curves.csv"
CATEGORICAL_FEATURES = [c for c in FEATURE_COLS if c in CATEGORICAL]
N_BINS = 10
EPS = 1e-6


# ---- metrics ---------------------------------------------------------------
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expected_calibration_error(y, p, n_bins=N_BINS):
    """Mean |predicted - observed| across equal-frequency bins, weighted by bin
    size. Equal-frequency rather than equal-width because the predictions pile
    up near the base rate: equal-width bins would leave most of the range
    holding almost no loans and let a handful of rows dominate the average.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.unique(np.nanquantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return float("nan")
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    total, err = len(y), 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        err += m.sum() / total * abs(p[m].mean() - y[m].mean())
    return float(err)


def max_calibration_error(y, p, n_bins=N_BINS):
    """The worst bin, not the average. A model can look well calibrated on ECE
    while being badly wrong exactly where the decision threshold will sit."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.unique(np.nanquantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return float("nan")
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    worst = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 50:
            worst = max(worst, abs(p[m].mean() - y[m].mean()))
    return float(worst)


def calibration_slope(y, p):
    """Regress the outcome on the predicted log-odds. Slope 1 means the model's
    confidence is correctly scaled; below 1 means it is overconfident -- its
    high predictions are too high and its low ones too low."""
    z = _logit(p).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan")
    lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(z, y)
    return float(lr.coef_[0][0])


def reliability_curve(y, p, n_bins=N_BINS):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.unique(np.nanquantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "predicted": float(p[m].mean()),
                         "observed": float(y[m].mean()), "n": int(m.sum())})
    return rows


def score(y, p):
    y = np.asarray(y, dtype=float)
    return {
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "mce": max_calibration_error(y, p),
        "slope": calibration_slope(y, p),
        "mean_pred": float(np.mean(p)),
        "observed": float(np.mean(y)),
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
    }


# ---- models ----------------------------------------------------------------
def fit_logistic_woe(fit, frames):
    enc = WOEEncoder()
    X = enc.fit_transform(fit[FEATURE_COLS], fit["default_window"])
    model = LogisticRegression(max_iter=2000, random_state=SEED)
    model.fit(X, fit["default_window"])
    return [model.predict_proba(enc.transform(f[FEATURE_COLS]))[:, 1] for f in frames]


def fit_lightgbm(fit, frames):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               random_state=SEED, n_jobs=2, verbose=-1)
    model.fit(fit[FEATURE_COLS], fit["default_window"],
              categorical_feature=CATEGORICAL_FEATURES)
    return [model.predict_proba(f[FEATURE_COLS])[:, 1] for f in frames]


MODELS = {"logistic_woe": fit_logistic_woe, "lightgbm": fit_lightgbm}


# ---- calibrators -----------------------------------------------------------
def cal_none(p_cal, y_cal):
    return lambda p: p


def cal_platt(p_cal, y_cal):
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(_logit(p_cal).reshape(-1, 1), y_cal)
    return lambda p: lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def cal_isotonic(p_cal, y_cal):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_cal, y_cal)
    return lambda p: iso.predict(p)


CALIBRATORS = {"none": cal_none, "platt": cal_platt, "isotonic": cal_isotonic}


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    print("loading modelling frame...")
    df = load_modelling_frame()
    print(f"{len(df):,} loans\n")

    rows, curves = [], []
    for year, outer_train, test, fit, calib, cal_year in outer_folds(df):
        print(f"fold {year}: model fit {len(fit):,} (to {cal_year - 1}) | "
              f"calibrator {len(calib):,} ({cal_year}) | test {len(test):,} ({year})")
        print(f"   base rates -- calibration cohort {calib['default_window'].mean():.2%}, "
              f"test cohort {test['default_window'].mean():.2%}, "
              f"gap {test['default_window'].mean() - calib['default_window'].mean():+.2%}")

        for mname, fn in MODELS.items():
            p_cal, p_test = fn(fit, [calib, test])
            for cname, make in CALIBRATORS.items():
                f = make(p_cal, calib["default_window"].values)
                # Scored twice: on the cohort the calibrator was fitted on, and
                # on the test cohort years later. The gap is the drift.
                s_cal = score(calib["default_window"].values, np.clip(f(p_cal), EPS, 1 - EPS))
                s_test = score(test["default_window"].values, np.clip(f(p_test), EPS, 1 - EPS))
                rows.append({"model": mname, "calibrator": cname, "fold": year,
                             "cal_year": cal_year,
                             **{f"cal_{k}": v for k, v in s_cal.items()},
                             **s_test})
                for r in reliability_curve(test["default_window"].values, f(p_test)):
                    curves.append({"model": mname, "calibrator": cname,
                                   "fold": year, **r})
                print(f"   {mname:13s} {cname:9s} test  Brier={s_test['brier']:.5f}  "
                      f"ECE={s_test['ece']:.4f}  slope={s_test['slope']:.2f}  "
                      f"pred={s_test['mean_pred']:.3f} vs obs={s_test['observed']:.3f}",
                      flush=True)
                pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
                pd.DataFrame(curves).to_csv(CURVE_PATH, index=False)
        print()

    res = pd.DataFrame(rows)
    print("=== across folds (mean) ===")
    print(res.groupby(["model", "calibrator"])[
        ["brier", "ece", "mce", "slope", "mean_pred", "observed", "pr_auc"]
    ].mean().round(4).to_string())

    print("\n=== paired against no calibration (identical test sets) ===")
    for mname in MODELS:
        sub = res[res.model == mname]
        w = sub.pivot(index="fold", columns="calibrator", values="ece")
        wb = sub.pivot(index="fold", columns="calibrator", values="brier")
        wr = sub.pivot(index="fold", columns="calibrator", values="pr_auc")
        for c in ["platt", "isotonic"]:
            d, db, dr = w[c] - w["none"], wb[c] - wb["none"], wr[c] - wr["none"]
            print(f"  {mname:13s} {c:9s} ECE {d.mean():+.4f} (sd {d.std():.4f}, "
                  f"same sign {bool((d>0).all() or (d<0).all())})   "
                  f"Brier {db.mean():+.5f}   PR-AUC {dr.mean():+.5f}")

    print("\n=== what does not survive the gap between cohorts ===")
    print("ECE on the cohort the calibrator was fitted on, vs on the test cohort:")
    for (m, c), g in res.groupby(["model", "calibrator"]):
        print(f"  {m:13s} {c:9s} calibration cohort {g['cal_ece'].mean():.4f}"
              f"  ->  test cohort {g['ece'].mean():.4f}"
              f"   ({g['ece'].mean() - g['cal_ece'].mean():+.4f})")
    print(f"\nwrote {OUT_PATH} and {CURVE_PATH}")
