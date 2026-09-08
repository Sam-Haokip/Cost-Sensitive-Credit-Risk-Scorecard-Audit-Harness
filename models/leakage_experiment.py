"""
Phase 1, deliverable 3: measure the cost of leakage (and, as a bonus, the cost
of the grade/int_rate circularity flagged in the leakage audit).

Trains the same quick model three ways on the same random 80/20 split:

  "clean"      -- the fully disciplined allow-list only (feature_audit.py's
                  "keep" columns). No leakage, no grade/sub_grade/int_rate/
                  installment.
  "grade_kept" -- clean + grade/sub_grade/int_rate/installment added back.
                  Isolates what those four columns alone are worth.
  "naive"      -- grade_kept + every leakage column too. This is what a
                  portfolio that skipped the leakage audit entirely would ship.

This is deliberately a quick, random 80/20 split, NOT the temporal validation
this project actually needs -- that's Phase 2. The only question this script
answers is "how big is the mistake," not "how good is the model."

Memory note: loading all ~140 candidate columns for 1.34M rows at pandas'
default object/string dtype exhausts memory on a modest machine. Fixed by
casting each column to a compact dtype (float32 for numerics, category for
categoricals) file-by-file before concatenating the 46 chunks, instead of
concatenating raw strings and casting afterward. Same lesson as the
CSV-to-Parquet conversion: convert to a cheap dtype BEFORE combining chunks,
not after.
"""
import glob
import os

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from data.feature_audit import FEATURE_AUDIT
from data.target import NEGATIVE_STATUSES, POSITIVE_STATUSES

PARQUET_DIR = os.environ.get("LC_PARQUET_DIR", "data/raw/parquet")

# not a leakage question -- just not feature-engineered yet (free text,
# high-cardinality raw zip, a secondary date field that's ~95% null anyway)
DEFERRED = {"emp_title", "desc", "title", "zip_code", "sec_app_earliest_cr_line"}

KEEP_COLS = sorted(c for c, (d, _) in FEATURE_AUDIT.items() if d == "keep" and c not in DEFERRED)
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

CATEGORICAL_KEEP = {
    "term", "home_ownership", "verification_status", "purpose", "addr_state",
    "application_type", "verification_status_joint", "initial_list_status",
    "disbursement_method", "emp_length",
}
CATEGORICAL_REVIEW = {"grade", "sub_grade"}

ALL_CATEGORICAL = CATEGORICAL_KEEP | CATEGORICAL_REVIEW | LEAKAGE_FLAG_COLS | LEAKAGE_CATEGORICAL_COLS
ALL_RAW_COLS = KEEP_COLS + REVIEW_COLS + LEAKAGE_NUMERIC_COLS + list(LEAKAGE_FLAG_COLS) + list(LEAKAGE_CATEGORICAL_COLS)
ALL_NUMERIC = sorted(set(ALL_RAW_COLS) - ALL_CATEGORICAL)


def load_master():
    """Load, label, and dtype-cast every needed column, chunk by chunk, so
    peak memory never holds more than one file's worth of raw strings."""
    files = sorted(glob.glob(f"{PARQUET_DIR}/part_*.parquet"))
    if not files:
        raise SystemExit(
            f"No Parquet files found in: {PARQUET_DIR}\n"
            "Run the conversion step first:  python -m data.convert_raw"
        )
    needed = sorted(set(ALL_RAW_COLS) | {"loan_status", "issue_d", "earliest_cr_line"} | LEAKAGE_DATE_COLS)

    chunks = []
    for f in files:
        chunk = pd.read_parquet(f, columns=needed)

        chunk["default"] = chunk["loan_status"].map(
            lambda s: 1 if s in POSITIVE_STATUSES else (0 if s in NEGATIVE_STATUSES else None)
        )
        chunk = chunk[chunk["default"].notna()]
        if chunk.empty:
            continue
        chunk = chunk.copy()

        issue = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        earliest = pd.to_datetime(chunk["earliest_cr_line"], format="%b-%Y", errors="coerce")
        chunk["credit_history_months"] = ((issue - earliest).dt.days / 30.44).astype("float32")

        for c in LEAKAGE_DATE_COLS:
            chunk[f"{c}_present"] = chunk[c].notna().astype("int8")

        for c in ALL_NUMERIC:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype("float32")
        # NOTE: categoricals are intentionally left as plain strings here, not
        # cast to 'category' yet -- see below.

        final_cols = ALL_RAW_COLS + ["credit_history_months"] + DATE_PRESENCE_COLS + ["default"]
        chunks.append(chunk[final_cols])

    master = pd.concat(chunks, ignore_index=True)
    master["default"] = master["default"].astype("int8")

    # Casting to 'category' must happen AFTER concatenation, not per-chunk.
    # pd.concat on Categorical columns whose chunks don't all observe the
    # exact same set of category values silently degrades the result back to
    # plain object/string dtype (no error, no warning) -- caught this because
    # HistGradientBoostingClassifier's from_dtype detection then treated those
    # columns as numeric and crashed trying to convert 'MO' to a float.
    for c in ALL_CATEGORICAL:
        master[c] = master[c].astype("category")

    return master


def evaluate(name, X_train, X_test, y_train, y_test):
    model = HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        max_iter=150,
        random_state=42,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    result = dict(
        name=name,
        n_features=X_train.shape[1],
        pr_auc=average_precision_score(y_test, proba),
        roc_auc=roc_auc_score(y_test, proba),
        brier=brier_score_loss(y_test, proba),
    )
    print(
        f"{name:12s}  n_features={result['n_features']:3d}  "
        f"PR-AUC={result['pr_auc']:.4f}  ROC-AUC={result['roc_auc']:.4f}  Brier={result['brier']:.4f}"
    )
    return result


if __name__ == "__main__":
    print("loading + typing data chunk by chunk...")
    master = load_master()
    print(f"labeled rows: {len(master)}")
    print(f"master frame memory: {master.memory_usage(deep=True).sum() / 1e6:.0f} MB")

    # scikit-learn upcasts to float64 internally regardless of input dtype,
    # so even an ~800MB frame can exhaust memory during fit. Subsampling to
    # 400k rows (still stratified, still a very solid sample size) is a
    # pragmatic fix for THIS quick sanity check -- it is not a Phase 3
    # decision about how the real baselines get trained, which will need its
    # own measured approach to the same constraint.
    SAMPLE_SIZE = 400_000
    master, _ = train_test_split(
        master, train_size=SAMPLE_SIZE, random_state=42, stratify=master["default"]
    )
    print(f"subsampled to {len(master)} rows for this quick experiment")

    y = master["default"]

    clean_cols = KEEP_COLS + ["credit_history_months"]
    grade_kept_cols = clean_cols + REVIEW_COLS
    naive_cols = grade_kept_cols + LEAKAGE_NUMERIC_COLS + list(LEAKAGE_FLAG_COLS) + list(LEAKAGE_CATEGORICAL_COLS) + DATE_PRESENCE_COLS

    variants = {"clean": clean_cols, "grade_kept": grade_kept_cols, "naive": naive_cols}

    print()
    print("training (quick random 80/20 split -- Phase 2 will redo this properly with temporal CV):")
    results = []
    for name, cols in variants.items():
        X_train, X_test, y_train, y_test = train_test_split(
            master[cols], y, test_size=0.2, random_state=42, stratify=y
        )
        results.append(evaluate(name, X_train, X_test, y_train, y_test))

    print()
    print("headline gap:")
    by_name = {r["name"]: r for r in results}
    print(f"  naive vs clean       PR-AUC +{by_name['naive']['pr_auc'] - by_name['clean']['pr_auc']:.4f}")
    print(f"  grade_kept vs clean  PR-AUC +{by_name['grade_kept']['pr_auc'] - by_name['clean']['pr_auc']:.4f}")
    print(f"  naive vs grade_kept  PR-AUC +{by_name['naive']['pr_auc'] - by_name['grade_kept']['pr_auc']:.4f}")
