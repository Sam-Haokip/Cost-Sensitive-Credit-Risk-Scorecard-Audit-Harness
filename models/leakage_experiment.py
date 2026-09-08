"""
Measure the cost of leakage (and of the grade/int_rate circularity).

Trains the same quick model three ways on the same random split:

  "clean"      -- the disciplined allow-list only (feature_audit.py's "keep"
                  columns). No leakage, no grade/sub_grade/int_rate/installment.
  "grade_kept" -- clean + grade/sub_grade/int_rate/installment added back.
                  Isolates what those four columns alone are worth.
  "naive"      -- grade_kept + every leakage column too. What a project that
                  skipped the leakage audit entirely would ship.

Target is the fixed 18-month performance window (data/vintage_target.py), the
same one used everywhere else in the project, so these numbers sit on the same
footing as the temporal validation results.

This is a random 80/20 split, NOT the temporal validation in
models/temporal_validation.py. The only question it answers is "how big is the
mistake", not "how good is the model".

Note on using leakage columns here: this script asks the shared loader for the
excluded columns precisely so it can measure what including them would have
cost. That is the one legitimate reason to touch them.
"""
import os

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from data.dataset import FEATURE_COLS, load_modelling_frame
from data.feature_audit import FEATURE_AUDIT

REVIEW_COLS = sorted(c for c, (d, _) in FEATURE_AUDIT.items() if d == "review")
LEAKAGE_COLS = sorted(c for c, (d, _) in FEATURE_AUDIT.items() if d == "drop_leakage")

LEAKAGE_FLAG_COLS = {"hardship_flag", "debt_settlement_flag", "pymnt_plan"}
LEAKAGE_DATE_COLS = {
    "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d",
    "hardship_start_date", "hardship_end_date", "payment_plan_start_date",
    "debt_settlement_flag_date", "settlement_date",
}
LEAKAGE_CATEGORICAL_COLS = {
    "hardship_type", "hardship_reason", "hardship_status",
    "hardship_loan_status", "settlement_status",
}
LEAKAGE_NUMERIC_COLS = sorted(
    set(LEAKAGE_COLS) - LEAKAGE_FLAG_COLS - LEAKAGE_DATE_COLS - LEAKAGE_CATEGORICAL_COLS
)
DATE_PRESENCE_COLS = [f"{c}_present" for c in sorted(LEAKAGE_DATE_COLS)]

CATEGORICAL_REVIEW = {"grade", "sub_grade"}
SAMPLE_SIZE = 400_000
SEED = 42
OUT_PATH = "reports/leakage_experiment.csv"


def evaluate(name, X_train, X_test, y_train, y_test):
    model = HistGradientBoostingClassifier(
        categorical_features="from_dtype", max_iter=150, random_state=SEED
    )
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    r = dict(
        regime=name,
        n_features=X_train.shape[1],
        pr_auc=average_precision_score(y_test, p),
        roc_auc=roc_auc_score(y_test, p),
        brier=brier_score_loss(y_test, p),
    )
    print(f"{name:12s}  n_features={r['n_features']:3d}  PR-AUC={r['pr_auc']:.4f}  "
          f"ROC-AUC={r['roc_auc']:.4f}  Brier={r['brier']:.4f}", flush=True)
    return r


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)

    extra = (REVIEW_COLS + LEAKAGE_NUMERIC_COLS + sorted(LEAKAGE_FLAG_COLS)
             + sorted(LEAKAGE_CATEGORICAL_COLS) + sorted(LEAKAGE_DATE_COLS))
    extra_cat = (CATEGORICAL_REVIEW | LEAKAGE_FLAG_COLS | LEAKAGE_CATEGORICAL_COLS
                 | LEAKAGE_DATE_COLS)

    print("loading...")
    df = load_modelling_frame(extra_cols=extra, extra_categorical=extra_cat)

    # Leaky dates carry their signal mainly through mere presence -- a defaulted
    # loan has no scheduled next payment -- so a boolean is enough to make the
    # point without one-hot-encoding hundreds of month values.
    for c in sorted(LEAKAGE_DATE_COLS):
        df[f"{c}_present"] = df[c].notna().astype("int8")
    df = df.drop(columns=sorted(LEAKAGE_DATE_COLS))

    df, _ = train_test_split(df, train_size=SAMPLE_SIZE, random_state=SEED,
                             stratify=df["default_window"])
    y = df["default_window"]
    print(f"{len(df):,} loans sampled, base rate {y.mean():.2%}\n")

    variants = {
        "clean": FEATURE_COLS,
        "grade_kept": FEATURE_COLS + REVIEW_COLS,
        "naive": (FEATURE_COLS + REVIEW_COLS + LEAKAGE_NUMERIC_COLS
                  + sorted(LEAKAGE_FLAG_COLS) + sorted(LEAKAGE_CATEGORICAL_COLS)
                  + DATE_PRESENCE_COLS),
    }

    results = []
    for name, cols in variants.items():
        Xtr, Xte, ytr, yte = train_test_split(df[cols], y, test_size=0.2,
                                              random_state=SEED, stratify=y)
        results.append(evaluate(name, Xtr, Xte, ytr, yte))

    out = pd.DataFrame(results)
    out.to_csv(OUT_PATH, index=False)
    by = {r["regime"]: r for r in results}
    print()
    print(f"naive vs clean       PR-AUC {by['naive']['pr_auc'] - by['clean']['pr_auc']:+.4f}")
    print(f"grade_kept vs clean  PR-AUC {by['grade_kept']['pr_auc'] - by['clean']['pr_auc']:+.4f}")
    print(f"\nwrote {OUT_PATH}")
