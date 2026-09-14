"""
Single loader for the modelling frame, shared by every downstream script.

Combines the allow-list features (data/feature_audit.py) with the fixed-window
vintage target (data/vintage_target.py), and keeps issue_d so callers can build
time-based splits.

Memory: loads ~97 feature columns for ~1.4M rows. Columns are cast to compact
dtypes (float32 / category) file-by-file BEFORE the 46 chunks are concatenated,
because concatenating raw strings and casting afterward exhausts memory on a
modest machine. Categoricals are cast only after the concat -- pandas silently
degrades a Categorical column back to object dtype when the chunks don't all
observe the same category values.
"""
import glob
import os

import pandas as pd

from data.feature_audit import FEATURE_AUDIT
from data.vintage_target import (
    CHARGEOFF_LAG_MONTHS,
    DEFAULT_STATUSES,
    POLICY_EXCLUDED,
    WINDOW_MONTHS,
)

PARQUET_DIR = os.environ.get("LC_PARQUET_DIR", "data/raw/parquet")

# On the allow-list but not yet feature-engineered: free text and raw zip.
# earliest_cr_line joins them for a different reason -- see decision 13: it is
# a "keep" date field, but the date VALUE is already captured properly by the
# derived `credit_history_months` below, computed from a correctly-parsed
# copy before this column is (mis)treated as numeric. Left in KEEP_COLS, the
# raw "Aug-2003"-style string used to get silently zeroed out by the numeric
# coercion loop instead -- a 100%-null feature had been in FEATURE_COLS since
# Phase 3. See _coerce_numeric_or_raise, which now catches this class of bug
# instead of doing it quietly.
DEFERRED = {"emp_title", "desc", "title", "zip_code", "sec_app_earliest_cr_line",
            "earliest_cr_line"}

KEEP_COLS = sorted(c for c, (d, _) in FEATURE_AUDIT.items() if d == "keep" and c not in DEFERRED)

CATEGORICAL = {
    "term", "home_ownership", "verification_status", "purpose", "addr_state",
    "application_type", "verification_status_joint", "initial_list_status",
    "disbursement_method", "emp_length",
}
NUMERIC = sorted(set(KEEP_COLS) - CATEGORICAL)

# Outcome columns, used ONLY to construct the label -- never exported as features.
_OUTCOME_COLS = ["loan_status", "last_pymnt_d", "last_credit_pull_d"]

FEATURE_COLS = KEEP_COLS + ["credit_history_months"]

# How much extra nullness a numeric coercion is allowed to introduce before
# it's treated as "this column wasn't actually numeric" rather than "a few
# more values happened to be unparseable". Found via Phase 6: an extra_col
# not passed in extra_categorical is ASSUMED numeric and silently coerced --
# for a genuinely numeric column stored as text ("13.99%") that's correct and
# harmless, but for zip_code ("190xx") every single value fails to parse and
# the column becomes 100% null with no error at all. 5 points is generous
# enough for real numeric noise, tight enough to catch "this is not a number".
_MAX_COERCION_NULL_INCREASE = 0.05


def _coerce_numeric_or_raise(series: pd.Series, col_name: str) -> pd.Series:
    """pd.to_numeric(series, errors='coerce'), but fails loudly instead of
    quietly if that coercion destroys values that weren't already missing --
    the failure mode this guards is a caller passing a categorical/text
    extra_col without also listing it in `extra_categorical`."""
    before_null = series.isna().mean()
    coerced = pd.to_numeric(series, errors="coerce")
    after_null = coerced.isna().mean()
    if after_null > before_null + _MAX_COERCION_NULL_INCREASE:
        raise ValueError(
            f"load_modelling_frame: coercing {col_name!r} to numeric turned "
            f"{after_null - before_null:.1%} of its values null that weren't "
            f"null in the source (e.g. a non-numeric string like a zip "
            f"prefix). It probably isn't numeric -- pass it in "
            f"`extra_categorical` instead."
        )
    return coerced


def load_modelling_frame(parquet_dir: str = PARQUET_DIR, window: int = WINDOW_MONTHS,
                         extra_cols=None, extra_categorical=None) -> pd.DataFrame:
    """Returns eligible loans with allow-list features, `default_window`, and
    `issue_d` / `issue_year` for temporal splitting.

    `extra_cols` carries additional RAW columns through unchanged -- used only by
    the leakage experiment, which needs the excluded columns precisely so it can
    measure what including them would have cost. Nothing else should pass it."""
    extra_cols = list(extra_cols or [])
    extra_categorical = set(extra_categorical or [])
    files = sorted(glob.glob(os.path.join(parquet_dir, "part_*.parquet")))
    if not files:
        raise SystemExit(
            f"No Parquet files found in: {parquet_dir}\n"
            "Run the conversion step first:  python -m data.convert_raw"
        )

    needed = sorted(set(KEEP_COLS) | set(extra_cols) | set(_OUTCOME_COLS)
                    | {"issue_d", "earliest_cr_line"})

    # Snapshot date must be computed over the WHOLE file, before any filtering.
    snap = None
    for f in files:
        d = pd.read_parquet(f, columns=["last_pymnt_d", "last_credit_pull_d"])
        for c in ("last_pymnt_d", "last_credit_pull_d"):
            m = pd.to_datetime(d[c], format="%b-%Y", errors="coerce").max()
            snap = m if snap is None or (pd.notna(m) and m > snap) else snap

    chunks = []
    for f in files:
        c = pd.read_parquet(f, columns=needed)

        issue = pd.to_datetime(c["issue_d"], format="%b-%Y", errors="coerce")
        last_pymnt = pd.to_datetime(c["last_pymnt_d"], format="%b-%Y", errors="coerce")
        earliest = pd.to_datetime(c["earliest_cr_line"], format="%b-%Y", errors="coerce")

        months_observed = ((snap - issue).dt.days / 30.44).round()
        months_to_last = ((last_pymnt - issue).dt.days / 30.44).round()

        eligible = (
            (months_observed >= window + CHARGEOFF_LAG_MONTHS)
            & ~c["loan_status"].isin(POLICY_EXCLUDED)
            & c["loan_status"].notna()
            & issue.notna()
        )
        if not eligible.any():
            continue

        c = c[eligible].copy()
        c["default_window"] = (
            c["loan_status"].isin(DEFAULT_STATUSES) & (months_to_last[eligible] <= window)
        ).astype("int8")
        c["issue_d"] = issue[eligible]
        c["credit_history_months"] = (
            (issue[eligible] - earliest[eligible]).dt.days / 30.44
        ).astype("float32")

        extra_numeric = [x for x in extra_cols if x not in extra_categorical]
        for col in list(NUMERIC) + extra_numeric:
            c[col] = _coerce_numeric_or_raise(c[col], col).astype("float32")

        chunks.append(c[FEATURE_COLS + extra_cols + ["issue_d", "default_window"]])

    df = pd.concat(chunks, ignore_index=True)
    for col in set(CATEGORICAL) | extra_categorical:
        df[col] = df[col].astype("category")
    df["issue_year"] = df["issue_d"].dt.year.astype("int16")
    return df


if __name__ == "__main__":
    df = load_modelling_frame()
    print(f"rows: {len(df):,}   features: {len(FEATURE_COLS)}")
    print(f"memory: {df.memory_usage(deep=True).sum()/1e6:.0f} MB")
    print(f"default rate: {df['default_window'].mean():.2%}")
    print()
    print(df.groupby("issue_year")["default_window"].agg(loans="size", rate="mean").assign(
        rate=lambda t: (t["rate"] * 100).round(1)).to_string())
