"""
Build the binary default-risk target from Lending Club's `loan_status` field.

Decision (full rationale and the alternative it beat: see DECISIONS.md):
- Positive class (default=1): Charged Off, Default.
- Negative class (default=0): Fully Paid.
- Excluded entirely (no label assigned, row dropped from modeling):
    * Non-terminal statuses -- Current, In Grace Period, Late (16-30 days),
      Late (31-120 days). The loan hasn't finished yet, so there's no true
      outcome to give it. Guessing one would inject an error that correlates
      with how recently the loan was issued (recent loans are almost all
      still "Current" simply because not enough time has passed) -- which
      would corrupt the temporal validation this whole project depends on.
    * "Does not meet the credit policy" statuses -- known outcome, but these
      loans were approved under underwriting rules Lending Club no longer
      uses, so mixing them in risks learning patterns that don't transfer.
    * Rows with a null/missing loan_status -- corrupted footer rows, not
      real loans.
"""
import glob
import os

import pandas as pd

PARQUET_DIR = "data/raw/parquet"

POSITIVE_STATUSES = {"Charged Off", "Default"}
NEGATIVE_STATUSES = {"Fully Paid"}


def load_status_columns(parquet_dir: str = PARQUET_DIR) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(parquet_dir, "part_*.parquet")))
    frames = [pd.read_parquet(f, columns=["loan_status", "issue_d", "term"]) for f in files]
    return pd.concat(frames, ignore_index=True)


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["default"] = df["loan_status"].map(
        lambda s: 1 if s in POSITIVE_STATUSES else (0 if s in NEGATIVE_STATUSES else None)
    )
    return df


NON_TERMINAL_STATUSES = {
    "Current",
    "In Grace Period",
    "Late (16-30 days)",
    "Late (31-120 days)",
}
POLICY_EXCLUDED_STATUSES = {
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
}


def survivorship_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Break the excluded rows down by *why* they're excluded, not just whether.

    First pass at this table lumped "still active" and "different credit
    policy" into one "not resolved" bucket, which made 2007-2009 loans look
    like they were still unresolved -- impossible for 3-year loans issued
    that long ago. They're actually "does not meet the credit policy" loans
    (LendingClub retroactively tagged old loans that wouldn't pass a later,
    stricter policy). That's a real outcome we know, just a different
    exclusion reason than "hasn't finished yet." Separating the two makes
    the actual survivorship-bias claim -- which is specifically about
    *recent* loans not having had time to resolve -- honest.
    """
    d = df.copy()
    d["term"] = d["term"].str.strip()
    d["issue_year"] = pd.to_datetime(d["issue_d"], format="%b-%Y", errors="coerce").dt.year
    d = d.dropna(subset=["issue_year"])

    def bucket(status):
        if status in NON_TERMINAL_STATUSES:
            return "still_active"
        if status in POLICY_EXCLUDED_STATUSES:
            return "policy_excluded"
        if pd.isna(status):
            return "junk_row"
        return "labeled"

    d["bucket"] = d["loan_status"].map(bucket)
    table = (
        d.groupby(["issue_year", "term", "bucket"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["labeled", "still_active", "policy_excluded", "junk_row"]:
        if col not in table.columns:
            table[col] = 0
    table["n_loans"] = table[["labeled", "still_active", "policy_excluded", "junk_row"]].sum(axis=1)
    table["still_active_pct"] = (table["still_active"] / table["n_loans"] * 100).round(1)
    table["labeled_pct"] = (table["labeled"] / table["n_loans"] * 100).round(1)
    return table[["issue_year", "term", "n_loans", "labeled", "labeled_pct", "still_active", "still_active_pct", "policy_excluded"]]


if __name__ == "__main__":
    df = load_status_columns()
    df = build_target(df)

    n_total = len(df)
    n_included = int(df["default"].notna().sum())
    print(f"total rows: {n_total}")
    print(f"included (labeled) rows: {n_included} ({n_included/n_total:.1%})")
    print(f"excluded rows: {n_total - n_included}")
    print()
    print("excluded, by original status:")
    print(df.loc[df["default"].isna(), "loan_status"].value_counts(dropna=False).to_string())
    print()
    print("class balance among included rows:")
    print(df["default"].value_counts(normalize=True).rename("share").to_string())
    print()
    print("resolution rate by issue year x term (the survivorship-bias table):")
    print(survivorship_table(df).to_string(index=False))
