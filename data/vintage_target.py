"""
Phase 2: a fixed-performance-window (vintage) target.

WHY THIS REPLACES THE PHASE 1 TARGET
------------------------------------
Phase 1 labelled a loan by its final status: Charged Off = 1, Fully Paid = 0,
everything unresolved dropped. That is correct as far as it goes, but it makes
cohorts non-comparable, which breaks the temporal validation Phase 2 exists to
build.

The problem: defaults resolve fast, full repayments resolve slowly. A loan that
charges off is finished within a year or two; a loan that pays normally takes
the full 36 or 60 months. So when you look at a cohort that is only partly
resolved, you are looking at a pool enriched with whichever outcome happens to
finish quicker. Measured on the Phase 1 target, the 2016 and 2017 36-month
cohorts show ~20% default against ~13-15% for every fully-resolved year before
them. That is not a credit-quality shift, it is an artefact of cohort age -- and
a naive time-based split would have reported it as drift.

The fix is standard credit-risk practice: measure every cohort over an identical
window from origination. "Did this loan default within N months?" is comparable
across 2013 and 2017 in a way that "did this loan ultimately default?" is not.

TWO CONSEQUENCES WORTH NOTING
-----------------------------
1. Still-Current loans become usable. A 2016 loan still performing in 2019 did
   NOT default within 18 months -- that is a known outcome, not a missing one.
   The Phase 1 target threw away 913k such loans; this recovers most of them,
   which also mitigates (not just measures) the Phase 1 survivorship bias.
2. The target now means something narrower: "defaulted early", not "ever
   defaulted". An 18-month window captures ~56% of eventual charge-offs. The
   base rate falls accordingly, and the model answers a different -- but
   cleanly-posed and consistently-measured -- question.

ON USING LEAKAGE COLUMNS HERE
-----------------------------
This module reads last_pymnt_d, which the leakage audit classifies as
drop_leakage. That is deliberate and not a contradiction: outcome columns are
what you BUILD A LABEL from. The rule they violate is being used as model
inputs, and they are not exported as features anywhere. Label construction is
allowed to see the future; the model is not.
"""
import glob
import os

import pandas as pd

PARQUET_DIR = os.environ.get("LC_PARQUET_DIR", "data/raw/parquet")

# Performance window: default is judged over exactly this many months from
# origination, identically for every cohort.
WINDOW_MONTHS = 18

# Lending Club charges a loan off at roughly 120-150 days delinquent, so the
# charge-off is recorded several months after the borrower actually stops
# paying. A loan must therefore be observable for WINDOW + LAG months before we
# can be confident we would have SEEN a within-window default had one occurred.
CHARGEOFF_LAG_MONTHS = 6

DEFAULT_STATUSES = {"Charged Off", "Default"}
# Non-terminal but performing-or-recoverable at snapshot; they did not charge
# off, so within a closed window they count as survivals. See caveat below.
OPEN_STATUSES = {"Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"}
POLICY_EXCLUDED = {
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
}


def load_outcome_columns(parquet_dir: str = PARQUET_DIR) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(parquet_dir, "part_*.parquet")))
    if not files:
        raise SystemExit(
            f"No Parquet files found in: {parquet_dir}\n"
            "Run the conversion step first:  python -m data.convert_raw"
        )
    cols = ["issue_d", "last_pymnt_d", "last_credit_pull_d", "loan_status", "term"]
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def snapshot_date(df: pd.DataFrame) -> pd.Timestamp:
    """Latest date observable anywhere in the file -- when the data was pulled."""
    return max(df["last_pymnt_d"].max(), df["last_credit_pull_d"].max())


def build_vintage_target(df: pd.DataFrame, window: int = WINDOW_MONTHS) -> pd.DataFrame:
    d = df.copy()
    for c in ("issue_d", "last_pymnt_d", "last_credit_pull_d"):
        d[c] = pd.to_datetime(d[c], format="%b-%Y", errors="coerce")
    d["term"] = d["term"].str.strip()

    snap = snapshot_date(d)
    d["months_observed"] = ((snap - d["issue_d"]).dt.days / 30.44).round()
    d["months_to_last_pymnt"] = ((d["last_pymnt_d"] - d["issue_d"]).dt.days / 30.44).round()

    # Eligible = we have watched this loan long enough that a within-window
    # default would have had time to occur AND to be recorded as a charge-off.
    d["eligible"] = d["months_observed"] >= (window + CHARGEOFF_LAG_MONTHS)

    # Excluded regardless of window: retired underwriting regime, junk rows.
    d.loc[d["loan_status"].isin(POLICY_EXCLUDED), "eligible"] = False
    d.loc[d["loan_status"].isna(), "eligible"] = False
    d.loc[d["issue_d"].isna(), "eligible"] = False

    charged_off = d["loan_status"].isin(DEFAULT_STATUSES)
    stopped_in_window = d["months_to_last_pymnt"] <= window

    # 1 = charged off having stopped paying inside the window.
    # 0 = everything else eligible: repaid, still performing, or charged off
    #     only AFTER surviving the window.
    d["default_window"] = (charged_off & stopped_in_window).astype("int8")
    d.loc[~d["eligible"], "default_window"] = pd.NA

    d["issue_year"] = d["issue_d"].dt.year
    return d


def cohort_table(d: pd.DataFrame) -> pd.DataFrame:
    """The proof the fix worked: default rate should now be comparable across
    cohorts instead of spiking wherever the cohort happens to be young."""
    e = d[d["eligible"]]
    t = (e.groupby(["issue_year", "term"])["default_window"]
           .agg(loans="size", default_rate="mean").reset_index())
    t["default_rate"] = (t["default_rate"] * 100).round(1)
    return t


if __name__ == "__main__":
    raw = load_outcome_columns()
    d = build_vintage_target(raw)
    snap = snapshot_date(
        d.assign(**{c: pd.to_datetime(raw[c], format="%b-%Y", errors="coerce")
                    for c in ("last_pymnt_d", "last_credit_pull_d")})
    )

    n_elig = int(d["eligible"].sum())
    print(f"data snapshot           : {snap.date()}")
    print(f"performance window      : {WINDOW_MONTHS} months (+{CHARGEOFF_LAG_MONTHS} lag)")
    print(f"eligible loans          : {n_elig:,} of {len(d):,} ({n_elig/len(d):.1%})")
    print(f"default rate in window  : {d.loc[d['eligible'], 'default_window'].mean():.2%}")
    print()
    print("Recovered from Phase 1's excluded pile (still-open loans that")
    print("demonstrably survived the window):")
    open_recovered = d[d["eligible"] & d["loan_status"].isin(OPEN_STATUSES)]
    print(f"   {len(open_recovered):,} loans")
    print()
    print("Default rate by cohort -- should now be FLAT where Phase 1's spiked:")
    print(cohort_table(d).to_string(index=False))
