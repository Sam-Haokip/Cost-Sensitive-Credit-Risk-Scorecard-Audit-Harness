"""
Phase 5: turn a probability into a decision, and price what that decision is worth.

Everything through Phase 4 asked "does the model know something." This phase asks
a different question: given what it knows, where should the approve/reject line
sit, and what does getting that line wrong actually cost in dollars?

WHY THIS NEEDS ITS OWN COST MODEL, NOT JUST A THRESHOLD
--------------------------------------------------------
Approving a loan that defaults costs roughly what's left unpaid at charge-off.
Rejecting a loan that would have performed costs the interest income that loan
would have earned. These are not the same currency as PR-AUC or Brier score, and
they are not symmetric -- a $30,000 default and a $30,000 good loan are not
opposite outcomes of equal size, because the bad outcome loses principal and the
good outcome only earns a spread on it. A 0.5 probability threshold has no
relationship to where these two costs cross.

UNIT ECONOMICS ARE MEASURED, NOT ASSUMED
-----------------------------------------
Loss-given-default and margin-per-dollar-lent are estimated from loans whose
lifetime outcome is fully known -- Charged Off / Default and Fully Paid,
regardless of the 18-month modelling window -- because these are population-level
business constants (how much does an average default cost, how much does an
average performing loan earn), not per-loan predictions. This deliberately
decouples them from `default_window`: the model predicts an early-warning proxy
("defaulted within 18 months"), while the unit economics describe the full
lifetime outcome. That gap is real and is stated here rather than glossed over --
see `unit_economics()`'s docstring.

WHICH MODEL AND CALIBRATOR, AND WHY THAT IS NOT RE-LITIGATED HERE
-------------------------------------------------------------------
Decision 10 already measured this: Platt over isotonic, specifically *because*
isotonic collapses ~100,000 distinct predictions onto ~98 tied values, and a
cost-optimal threshold can only ever sit at one of however many distinct values
survive calibration. That argument was made in the abstract in Phase 4; here it
is the entire mechanism the threshold search depends on, so Platt is the only
calibrator used. Both models from Decision 8/10 are still scored side by side --
logistic-WoE as the primary (interpretable, arrives near-calibrated on its own),
LightGBM as the comparison -- to check whether model choice matters for a cost
decision even though Phase 3 found it does not matter for ranking.

THE EXTENSION PAST THE BRIEF: DOES THE THRESHOLD ITSELF DRIFT?
------------------------------------------------------------------
Decision 10's sharpest finding was that a calibrator fitted on one cohort and
scored two-plus years later converges to the same error floor no matter how well
it fit its own data. A cost-optimal threshold is downstream of exactly those
probabilities, so the natural next question is whether the THRESHOLD decision
itself goes stale the same way. This is measured directly: pick the threshold
that would have looked optimal on the most recent cohort available at decision
time, apply it unchanged to the cohort 2+ years later, and compare that to the
threshold that actually would have been optimal on that later cohort (an
oracle no real deployment could have computed in advance). The gap between the
two is the dollar cost of not re-optimising the threshold as time passes.
"""
import glob
import os

import numpy as np
import pandas as pd

from data.dataset import load_modelling_frame
from evaluation.calibration import MODELS, cal_platt
from models.tuning import outer_folds

OUT_THRESH = "reports/decisioning_thresholds.csv"
OUT_SENS = "reports/decisioning_sensitivity.csv"
OUT_CURVE = "reports/decisioning_curves.csv"
OUT_ECON = "reports/unit_economics.md"

PARQUET_DIR = os.environ.get("LC_PARQUET_DIR", "data/raw/parquet")
EPS = 1e-6

# Same resolved-outcome definition as data/target.py's original (pre-window)
# target -- used ONLY to identify loans whose lifetime economic outcome is
# fully known, not as a modelling label.
RESOLVED_DEFAULT = {"Charged Off", "Default"}
RESOLVED_PAID = {"Fully Paid"}

SENSITIVITY_PCTS = [-0.30, 0.0, 0.30]
NAIVE_THRESHOLD = 0.5


# ---- unit economics ---------------------------------------------------------
def _load_econ_columns():
    files = sorted(glob.glob(os.path.join(PARQUET_DIR, "part_*.parquet")))
    cols = ["loan_status", "issue_d", "loan_amnt", "term", "int_rate",
            "total_rec_prncp", "total_rec_int", "recoveries", "collection_recovery_fee"]
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def unit_economics():
    """Dollar-weighted loss-given-default and margin rates from loans with a
    fully known lifetime outcome, plus a formula cross-check and a by-year
    stability check.

    LGD_RATE: for charged-off loans, net_loss = loan_amnt - (principal actually
    recovered + gross post-charge-off recoveries - the collections fee taken out
    of those recoveries). Dollar-weighted (sum of losses / sum of loan amounts,
    not a mean of per-loan ratios) so a portfolio of larger loans is not diluted
    by many small ones -- the standard way LGD is reported.

    MARGIN_RATE: for fully-paid loans, realised interest actually collected
    (`total_rec_int`) as a fraction of loan_amnt, dollar-weighted the same way.
    Compared against a naive full-term formula (int_rate x term-in-years) to
    quantify how much early payoff shrinks realised yield below the sticker
    rate -- a loan that pays off in month 20 of a 36-month term never earns the
    other 16 months of interest the formula assumes.
    """
    raw = _load_econ_columns()
    raw["issue_year"] = pd.to_datetime(raw["issue_d"], format="%b-%Y", errors="coerce").dt.year
    raw["term_years"] = raw["term"].str.extract(r"(\d+)").astype(float) / 12.0
    for c in ("loan_amnt", "total_rec_prncp", "total_rec_int", "recoveries",
              "collection_recovery_fee", "int_rate"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    bad = raw[raw["loan_status"].isin(RESOLVED_DEFAULT)].dropna(subset=["loan_amnt"])
    good = raw[raw["loan_status"].isin(RESOLVED_PAID)].dropna(subset=["loan_amnt"])

    net_recovered = (bad["total_rec_prncp"].fillna(0) + bad["recoveries"].fillna(0)
                     - bad["collection_recovery_fee"].fillna(0))
    net_loss = (bad["loan_amnt"] - net_recovered).clip(lower=0, upper=bad["loan_amnt"])
    lgd_rate = float(net_loss.sum() / bad["loan_amnt"].sum())

    margin_rate = float(good["total_rec_int"].sum() / good["loan_amnt"].sum())
    formula_yield = good["int_rate"] / 100 * good["term_years"]
    margin_rate_formula = float((formula_yield * good["loan_amnt"]).sum() / good["loan_amnt"].sum())

    by_year_lgd = (net_loss.groupby(bad["issue_year"]).sum()
                   / bad.groupby("issue_year")["loan_amnt"].sum()).dropna()
    by_year_margin = (good.groupby("issue_year")["total_rec_int"].sum()
                      / good.groupby("issue_year")["loan_amnt"].sum()).dropna()
    # Drop the most recent 2 cohort years: too few fully-resolved loans yet
    # (this is exactly the survivorship-bias mechanism from decision 5) to be a
    # meaningful rate, not a real drift signal.
    cutoff_year = int(raw["issue_year"].max()) - 2
    by_year_lgd = by_year_lgd[by_year_lgd.index <= cutoff_year]
    by_year_margin = by_year_margin[by_year_margin.index <= cutoff_year]

    return {
        "lgd_rate": lgd_rate,
        "margin_rate": margin_rate,
        "margin_rate_formula": margin_rate_formula,
        "n_charged_off": int(len(bad)),
        "n_fully_paid": int(len(good)),
        "lgd_rate_by_year": by_year_lgd,
        "margin_rate_by_year": by_year_margin,
    }


# ---- threshold search --------------------------------------------------------
def profit_curve(y, p, loan_amnt, lgd_rate, margin_rate):
    """Exact profit at every threshold that could possibly be optimal.

    Approving loan i earns +loan_amnt*margin_rate if it performs (y=0) and costs
    -loan_amnt*lgd_rate if it defaults in-window (y=1); rejecting always nets 0
    (no loan made). Profit is piecewise constant between consecutive sorted
    p-values, so sorting once and taking a cumulative sum gives the exact profit
    at every candidate cut-point in O(n log n) -- no grid search, no discretising
    error, and this is also the shape needed for the profit-vs-threshold figure.

    Returns arrays (thresholds, cumulative_profit_per_applicant) where
    thresholds[k] = "approve everyone with p <= thresholds[k]" and profit is
    the MEAN profit per applicant in the whole population (rejected loans
    included, at 0), so profit is comparable across folds of different size.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    amt = np.asarray(loan_amnt, dtype=float)
    order = np.argsort(p, kind="mergesort")
    p_sorted, y_sorted, amt_sorted = p[order], y[order], amt[order]

    per_loan_if_approved = np.where(y_sorted == 1, -lgd_rate * amt_sorted, margin_rate * amt_sorted)
    cum_profit = np.cumsum(per_loan_if_approved)
    n = len(y)
    # Prepend the "approve nobody" point (threshold below the minimum score).
    thresholds = np.concatenate([[p_sorted[0] - 1e-9], p_sorted])
    mean_profit = np.concatenate([[0.0], cum_profit]) / n
    return thresholds, mean_profit


def best_threshold(y, p, loan_amnt, lgd_rate, margin_rate):
    thr, profit = profit_curve(y, p, loan_amnt, lgd_rate, margin_rate)
    i = int(np.argmax(profit))
    return float(thr[i]), float(profit[i])


def profit_from_approve(y, approve, loan_amnt, lgd_rate, margin_rate):
    """Same accounting as profit_at, taking a precomputed approve mask
    directly instead of deriving one from a single scalar threshold. Added for
    Phase 6: a fairness mitigation approves each applicant against THEIR
    GROUP's own threshold, so there is no one scalar to hand profit_at -- but
    the dollar accounting per approved/rejected loan is identical either way,
    and it stays here so both call sites can never drift apart."""
    y, loan_amnt = np.asarray(y, dtype=float), np.asarray(loan_amnt, dtype=float)
    approve = np.asarray(approve, dtype=bool)
    per_loan = np.where(~approve, 0.0, np.where(y == 1, -lgd_rate * loan_amnt, margin_rate * loan_amnt))
    return float(per_loan.mean()), float(approve.mean())


def profit_at(y, p, loan_amnt, lgd_rate, margin_rate, threshold):
    p = np.asarray(p, dtype=float)
    return profit_from_approve(y, p <= threshold, loan_amnt, lgd_rate, margin_rate)


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)

    print("=== unit economics (loans with a fully known lifetime outcome) ===")
    econ = unit_economics()
    LGD_RATE, MARGIN_RATE = econ["lgd_rate"], econ["margin_rate"]
    print(f"charged-off/default loans : {econ['n_charged_off']:,}")
    print(f"fully paid loans          : {econ['n_fully_paid']:,}")
    print(f"LGD rate (dollar-weighted): {LGD_RATE:.3f}  "
          f"(a defaulted $10,000 loan loses ~${LGD_RATE*10000:,.0f} net of recoveries)")
    print(f"margin rate, realised     : {MARGIN_RATE:.3f}  "
          f"(a performing $10,000 loan earns ~${MARGIN_RATE*10000:,.0f} in collected interest)")
    print(f"margin rate, full-term formula (int_rate x term): {econ['margin_rate_formula']:.3f}  "
          f"-- realised is {(1 - MARGIN_RATE/econ['margin_rate_formula']):.1%} lower, "
          f"the early-payoff effect")
    print(f"\nLGD rate by cohort year (excl. last 2, not yet resolved):")
    print(econ["lgd_rate_by_year"].round(3).to_string())
    print(f"\nmargin rate by cohort year (excl. last 2, not yet resolved):")
    print(econ["margin_rate_by_year"].round(3).to_string())
    with open(OUT_ECON, "w") as f:
        f.write("# Unit economics for Phase 5 cost-sensitive decisioning\n\n")
        f.write(f"- Charged-off/default loans used: {econ['n_charged_off']:,}\n")
        f.write(f"- Fully paid loans used: {econ['n_fully_paid']:,}\n")
        f.write(f"- LGD rate (dollar-weighted): {LGD_RATE:.4f}\n")
        f.write(f"- Margin rate, realised (dollar-weighted): {MARGIN_RATE:.4f}\n")
        f.write(f"- Margin rate, full-term formula: {econ['margin_rate_formula']:.4f}\n\n")
        f.write("## LGD rate by cohort year\n\n")
        f.write(econ["lgd_rate_by_year"].round(4).to_string() + "\n\n")
        f.write("## Margin rate by cohort year\n\n")
        f.write(econ["margin_rate_by_year"].round(4).to_string() + "\n")
    print(f"\nwrote {OUT_ECON}")

    print("\n\n=== threshold search + drift test ===")
    df = load_modelling_frame()
    print(f"{len(df):,} loans\n")

    thresh_rows, sens_rows, curve_rows = [], [], []
    for year, outer_train, test, fit, calib, cal_year in outer_folds(df):
        print(f"fold {year}: model fit {len(fit):,} (to {cal_year - 1}) | "
              f"calibrator/threshold {len(calib):,} ({cal_year}) | test {len(test):,} ({year})")

        for mname, fitfn in MODELS.items():
            p_fit_calib, p_fit_test = fitfn(fit, [calib, test])
            platt = cal_platt(p_fit_calib, calib["default_window"].values)
            p_calib = np.clip(platt(p_fit_calib), EPS, 1 - EPS)
            p_test = np.clip(platt(p_fit_test), EPS, 1 - EPS)

            y_calib, amt_calib = calib["default_window"].values, calib["loan_amnt"].values
            y_test, amt_test = test["default_window"].values, test["loan_amnt"].values

            # The realistic threshold: the best a real decision-maker could have
            # computed, using only the cohort available before the test cohort.
            # profit_calib_own is that threshold's profit on its OWN fitting
            # cohort -- reported alongside the test-cohort number for the same
            # reason decision 10 reports both: the gap between them is drift,
            # not just noise.
            thr_realistic, profit_calib_own = best_threshold(y_calib, p_calib, amt_calib, LGD_RATE, MARGIN_RATE)
            # The oracle: the best threshold ON THE TEST COHORT ITSELF -- not
            # achievable in a real deployment, an upper bound for comparison.
            thr_oracle, profit_oracle = best_threshold(y_test, p_test, amt_test, LGD_RATE, MARGIN_RATE)

            profit_realistic_on_test, approve_rate_realistic = profit_at(
                y_test, p_test, amt_test, LGD_RATE, MARGIN_RATE, thr_realistic)
            profit_naive_on_test, approve_rate_naive = profit_at(
                y_test, p_test, amt_test, LGD_RATE, MARGIN_RATE, NAIVE_THRESHOLD)
            _, approve_rate_oracle = profit_at(
                y_test, p_test, amt_test, LGD_RATE, MARGIN_RATE, thr_oracle)

            drift_cost = profit_oracle - profit_realistic_on_test
            row = {
                "fold": year, "model": mname, "cal_year": cal_year,
                "thr_realistic": thr_realistic, "thr_oracle": thr_oracle,
                "approve_rate_realistic": approve_rate_realistic,
                "approve_rate_oracle": approve_rate_oracle,
                "approve_rate_naive": approve_rate_naive,
                "profit_realistic_on_own_cohort": profit_calib_own,
                "profit_realistic_on_test": profit_realistic_on_test,
                "profit_oracle_on_test": profit_oracle,
                "profit_naive_on_test": profit_naive_on_test,
                "drift_cost_per_applicant": drift_cost,
                "drift_cost_pct_of_oracle": drift_cost / profit_oracle if profit_oracle else float("nan"),
                "vs_naive_gain_per_applicant": profit_realistic_on_test - profit_naive_on_test,
            }
            thresh_rows.append(row)
            print(f"   {mname:13s} thr={thr_realistic:.3f} (approve {approve_rate_realistic:.1%})  "
                  f"profit/applicant on test=${profit_realistic_on_test:.2f}  "
                  f"oracle=${profit_oracle:.2f}  naive-0.5=${profit_naive_on_test:.2f}  "
                  f"drift cost=${drift_cost:.2f} ({row['drift_cost_pct_of_oracle']:.1%} of oracle)",
                  flush=True)

            # Profit curve for the figure -- test cohort, this fold, this model.
            thr_arr, profit_arr = profit_curve(y_test, p_test, amt_test, LGD_RATE, MARGIN_RATE)
            approve_arr = np.searchsorted(np.sort(p_test), thr_arr, side="right") / len(p_test)
            step = max(1, len(thr_arr) // 200)
            for i in range(0, len(thr_arr), step):
                curve_rows.append({"fold": year, "model": mname,
                                    "threshold": thr_arr[i], "approve_rate": approve_arr[i],
                                    "profit_per_applicant": profit_arr[i]})

            # Sensitivity: perturb LGD and margin +/-30% independently, re-derive
            # the threshold ON THE SAME calib cohort, see how far it moves.
            for param in ("lgd_rate", "margin_rate"):
                for pct in SENSITIVITY_PCTS:
                    lgd = LGD_RATE * (1 + pct) if param == "lgd_rate" else LGD_RATE
                    mgn = MARGIN_RATE * (1 + pct) if param == "margin_rate" else MARGIN_RATE
                    thr_p, profit_p = best_threshold(y_calib, p_calib, amt_calib, lgd, mgn)
                    sens_rows.append({
                        "fold": year, "model": mname, "param": param, "shift_pct": pct,
                        "threshold": thr_p, "approve_rate": (p_calib <= thr_p).mean(),
                        "lgd_rate": lgd, "margin_rate": mgn,
                    })
            pd.DataFrame(thresh_rows).to_csv(OUT_THRESH, index=False)
            pd.DataFrame(sens_rows).to_csv(OUT_SENS, index=False)
            pd.DataFrame(curve_rows).to_csv(OUT_CURVE, index=False)
        print()

    res = pd.DataFrame(thresh_rows)
    print("=== across folds (mean, sd) ===")
    for m in res["model"].unique():
        sub = res[res.model == m]
        print(f"\n{m}:")
        print(f"  realistic threshold   : {sub['thr_realistic'].mean():.3f} (sd {sub['thr_realistic'].std():.3f})")
        print(f"  approve rate          : {sub['approve_rate_realistic'].mean():.1%}")
        print(f"  profit/applicant      : ${sub['profit_realistic_on_test'].mean():.2f} "
              f"(sd {sub['profit_realistic_on_test'].std():.2f})")
        print(f"  vs naive 0.5 gain     : ${sub['vs_naive_gain_per_applicant'].mean():+.2f} "
              f"(sd {sub['vs_naive_gain_per_applicant'].std():.2f}), "
              f"same sign: {bool((sub['vs_naive_gain_per_applicant'] > 0).all() or (sub['vs_naive_gain_per_applicant'] < 0).all())}")
        print(f"  drift cost/applicant  : ${sub['drift_cost_per_applicant'].mean():.2f} "
              f"(sd {sub['drift_cost_per_applicant'].std():.2f}), "
              f"{sub['drift_cost_pct_of_oracle'].mean():.1%} of oracle profit left on the table")

    print("\n=== sensitivity: how far does the threshold move under +/-30% cost assumptions? ===")
    sens = pd.DataFrame(sens_rows)
    # NB: bracket access, deliberately -- "shift_pct" was chosen to avoid this,
    # but the original column name here was "pct_change", which silently
    # resolves via attribute access to DataFrame.pct_change (a real pandas
    # method) instead of the column, turning `sens.pct_change == 0.0` into
    # `<bound method> == 0.0` -> plain `False`, then `sens[False]` -> KeyError.
    # Caught by actually running this, not by reading the code.
    base = sens[sens["shift_pct"] == 0.0].groupby(["fold", "model"])["threshold"].mean()
    for param in ["lgd_rate", "margin_rate"]:
        for pct in [-0.30, 0.30]:
            sub = sens[(sens["param"] == param) & (sens["shift_pct"] == pct)].set_index(["fold", "model"])["threshold"]
            delta = (sub - base).dropna()
            print(f"  {param:12s} {pct:+.0%}: mean threshold shift {delta.mean():+.4f} "
                  f"(sd {delta.std():.4f}, range {delta.min():+.4f} to {delta.max():+.4f})")

    print(f"\nwrote {OUT_THRESH}, {OUT_SENS}, {OUT_CURVE}")
