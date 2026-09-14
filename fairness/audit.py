"""
Phase 6: fairness and bias audit.

WHICH PROXIES
----------------
The brief asks for two kinds of proxy: geography (a race proxy) and
income/employment length (class proxies). annual_inc (income quintiles) and
emp_length (employment-length category, including "missing" as its own
group) need nothing beyond what's already a model feature. Geography needs a
Census join built in `fairness/geo_proxy.py` -- see that module's docstring
for why it's a materially weaker proxy than real BISG (no borrower surname is
available in LC's public data) and for why the live Census fetch can't run
inside this project's own automated pipeline (it needs real, unproxied
internet access). This script picks up `data/processed/census_zip3_race_proxy.csv`
-- the committed OUTPUT of that separate, one-time fetch -- if present, and
runs with two proxies rather than three if it isn't, so the class-proxy
results below never depend on the Census step having been run yet.

WHY NO DIRECT PROTECTED ATTRIBUTE IS USED, EVER
--------------------------------------------------
Lending Club's public data contains no race, sex, age, or disability field.
Every group label in this module is a proxy for something the model was
never told and this dataset never recorded. That is stated once here and
should be assumed throughout: a finding like "the model's error rates differ
by income quintile" is not a finding about a protected class, only about a
CLASS proxy the brief specifically asked for -- see fairness/metrics.py's
docstring for the demographic-parity/equal-opportunity naming convention this
whole module depends on.

METHODOLOGY, REUSING PHASE 5'S OWN MACHINERY ON PURPOSE
-----------------------------------------------------------
Same fit/calibrate/test split as every other phase (models.tuning.outer_folds),
same two models (logistic-WoE the primary per decision 8, LightGBM the
comparison), same Platt calibration (decision 10), same realistic
cost-optimal threshold (decision 12) -- fairness numbers computed on top of a
DIFFERENT model or threshold than the one this project actually recommends
would be beside the point. The mitigation experiment reuses
evaluation.decisioning's exact profit accounting (`profit_from_approve`) so a
mitigation's cost is denominated in the same dollars as decision 12, not a
new metric invented for this section alone.

Two mitigations, both fitted on the CALIBRATION cohort and scored on the TEST
cohort (never fitted on the data they're scored on, same discipline as the
threshold itself):
  - demographic parity: a per-group threshold that pushes every group's
    approval rate to the population's approval rate under the shared threshold.
  - equal opportunity: a per-group threshold that pushes every group's TPR
    (the fraction of CREDITWORTHY applicants in that group who get approved)
    to the population's TPR under the shared threshold.
Each is scored on the OTHER criterion too, and against the total dollar cost
of moving away from the profit-maximising shared threshold -- expected to be
a real cost most of the time (the shared threshold IS the profit-maximiser
on the calibration cohort; a mitigation targets a fairness criterion, not
profit, so it has no reason to also be profit-optimal), and reported rather
than assumed away either direction. On this dataset the measured mean cost
across folds is negative (the mitigation makes MORE money) in 4 of 8
proxy/model/scenario combinations -- see the fold-by-fold breakdown before
reading anything into that sign: with n=4 folds and a cost sd 1-3x the size
of the mean, most of these means are not distinguishable from zero, and the
sign flips fold to fold. The more interesting real pattern behind it: the
per-group `calibration_gap` (already in group_rates' output) is NOT flat
across income quintiles -- Platt calibration is fit once, pooled, so nothing
guarantees it calibrates every subgroup equally well. A per-group threshold
incidentally corrects some of that miscalibration as a side effect of
matching approval rate or TPR, which is a plausible reason a fairness
mitigation could occasionally cost nothing, or less than nothing, on some
folds -- worth a follow-up if this project had another phase, not asserted
as proven here.

STATED LIMITATION: STATISTICAL POWER
----------------------------------------
Every other phase's paired comparisons already rest on n=4 folds. Splitting
each fold further into quintiles or 12 emp_length categories divides an
already-small sample again. A single test fold (~100,000 loans) still gives
tens of thousands of loans per income quintile, but the smallest emp_length
category or a group with a low base rate can see this vary considerably fold
to fold -- gaps and costs are reported per fold, not silently pooled, so that
variability is visible rather than averaged away.

geo_race_proxy's "insufficient_data" bucket is this limitation's clearest
example, and gets the strongest response: it's a Census DATA-QUALITY flag
(a zip3 too small, or entirely outside ACS's ZCTA universe -- e.g. APO/FPO
military zips), not a race/ethnicity group, and its N is a few dozen loans
per 100k-loan test fold. The first version of this script included it as a
group like any other; its baseline approval rate was a noisy 100% in most
folds, and a per-group mitigation threshold fit on an even smaller
calibration-cohort slice of it sometimes made the OVERALL demographic-parity
gap WORSE than doing nothing -- the opposite of what a mitigation is for.
It's excluded from the gap/mitigation analysis entirely now (see the
__main__ block); its exclusion count is printed and written to the summary
rather than silently dropped from the data.
"""
import os

import numpy as np
import pandas as pd

from data.dataset import load_modelling_frame
from evaluation.calibration import MODELS, cal_platt
from evaluation.decisioning import best_threshold, profit_from_approve, unit_economics
from fairness.geo_proxy import DEFAULT_CACHE_PATH as GEO_CACHE_PATH, load_zip3_race_proxy
from fairness.metrics import (
    apply_group_thresholds,
    demographic_parity_gap,
    equalized_odds_gap,
    group_rates,
    per_group_thresholds,
)
from models.tuning import outer_folds

OUT_RATES = "reports/fairness_group_rates.csv"
OUT_GAPS = "reports/fairness_gaps.csv"
OUT_SUMMARY = "reports/fairness_summary.md"

EPS = 1e-6
N_INCOME_QUANTILES = 5


def income_quantile_labels(fit_income, apply_income, n=N_INCOME_QUANTILES):
    """Quintile edges are fit on `fit_income` ALONE (the calibration cohort)
    and applied unchanged to `apply_income` (the test cohort) -- fitting bin
    edges on the test cohort would let the grouping itself see data a real
    decision-time system wouldn't have. The outer edges are widened to
    +/-inf so a test-cohort income outside the calibration cohort's observed
    range still lands in the nearest quintile instead of becoming NaN."""
    _, edges = pd.qcut(fit_income, n, labels=False, retbins=True, duplicates="drop")
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    fit_labels = pd.cut(fit_income, edges, labels=False, include_lowest=True) + 1
    apply_labels = pd.cut(apply_income, edges, labels=False, include_lowest=True) + 1
    return fit_labels.astype(int), apply_labels.astype(int)


def emp_length_labels(series):
    """emp_length as-is, with missing values given their own explicit label
    rather than silently becoming a NaN that breaks group comparisons --
    same reasoning decision 7 applied to WoE-encoding missingness: an absent
    employment-length answer is itself information, not an absence of it."""
    return series.astype(object).fillna("missing").astype(str)


def race_proxy_labels(zip_code_series, zip3_table):
    """Maps each loan's zip3 prefix (LC truncates to "190xx"-style strings
    for borrower privacy -- already exactly 3 real digits) to the Census-
    derived plurality race/ethnicity group for that zip3. A zip3 present in
    LC's data but absent from the fetched ACS table (e.g. a territory zip3
    ACS's ZCTA table doesn't cover the same way) falls back to
    "insufficient_data" -- the same label geo_proxy.py already uses for a
    zip3 Census itself couldn't confidently characterize, so downstream code
    treats both cases identically rather than needing a second special case."""
    zip3 = zip_code_series.astype(str).str.slice(0, 3)
    lookup = zip3_table.set_index("zip3")["race_proxy_group"]
    return zip3.map(lookup).fillna("insufficient_data")


def _fit_and_calibrate(fit, calib, test, model_name):
    fitfn = MODELS[model_name]
    p_fit_calib, p_fit_test = fitfn(fit, [calib, test])
    platt = cal_platt(p_fit_calib, calib["default_window"].values)
    p_calib = np.clip(platt(p_fit_calib), EPS, 1 - EPS)
    p_test = np.clip(platt(p_fit_test), EPS, 1 - EPS)
    return p_calib, p_test


def run_proxy(proxy_name, group_calib, group_test, y_calib, y_test, p_calib, p_test,
              amt_calib, amt_test, shared_thr, fold, model_name, lgd_rate, margin_rate):
    """One proxy dimension, one fold, one model: baseline rates + gaps, then
    both mitigations, each scored on the test cohort and costed in dollars
    against the shared-threshold baseline.

    lgd_rate/margin_rate are passed explicitly rather than read off a module
    global -- the first version of this function read LGD_RATE/MARGIN_RATE
    set only inside the `if __name__ == "__main__":` block below, which meant
    it NameError'd the moment anything tried to import and call it directly
    (exactly what tests/test_audit.py does). Passing them as arguments makes
    the function callable -- and testable -- on its own."""
    rate_rows, gap_rows = [], []

    baseline = group_rates(y_test, p_test, shared_thr, group_test)
    for g, row in baseline.iterrows():
        rate_rows.append({"fold": fold, "model": model_name, "proxy": proxy_name,
                           "scenario": "baseline", "group": g, **row.to_dict()})
    baseline_dp = demographic_parity_gap(baseline)
    baseline_eo = equalized_odds_gap(baseline)
    baseline_profit, baseline_approve_rate = profit_from_approve(
        y_test, p_test <= shared_thr, amt_test, lgd_rate, margin_rate)
    gap_rows.append({"fold": fold, "model": model_name, "proxy": proxy_name, "scenario": "baseline",
                      "dp_gap": baseline_dp["gap"], "disparate_impact_ratio": baseline_dp["disparate_impact_ratio"],
                      "eo_tpr_gap": baseline_eo["tpr_gap"], "eo_fpr_gap": baseline_eo["fpr_gap"],
                      "profit_per_applicant": baseline_profit, "approve_rate": baseline_approve_rate,
                      "cost_vs_baseline": 0.0})

    calib_rates_shared = group_rates(y_calib, p_calib, shared_thr, group_calib)
    pop_approval_rate = float((p_calib <= shared_thr).mean())
    pop_tpr = float(calib_rates_shared["tpr"].mean())  # unweighted across groups, on purpose:
    # weighting by group size would make "equal opportunity" chase whichever
    # group is largest rather than a genuinely shared target.

    for scenario, mode, target in [("demographic_parity", "approval_rate", pop_approval_rate),
                                    ("equal_opportunity", "tpr", pop_tpr)]:
        thresholds = per_group_thresholds(y_calib, p_calib, group_calib, mode=mode, target=target)
        approve_test = apply_group_thresholds(p_test, group_test, thresholds, shared_thr)
        rates = _rates_from_approve(y_test, approve_test, group_test, p_test)
        for g, row in rates.iterrows():
            rate_rows.append({"fold": fold, "model": model_name, "proxy": proxy_name,
                               "scenario": scenario, "group": g, **row.to_dict()})
        dp = demographic_parity_gap(rates)
        eo = equalized_odds_gap(rates)
        profit, approve_rate = profit_from_approve(y_test, approve_test, amt_test, lgd_rate, margin_rate)
        gap_rows.append({"fold": fold, "model": model_name, "proxy": proxy_name, "scenario": scenario,
                          "dp_gap": dp["gap"], "disparate_impact_ratio": dp["disparate_impact_ratio"],
                          "eo_tpr_gap": eo["tpr_gap"], "eo_fpr_gap": eo["fpr_gap"],
                          "profit_per_applicant": profit, "approve_rate": approve_rate,
                          "cost_vs_baseline": baseline_profit - profit})

    return rate_rows, gap_rows


def _rates_from_approve(y_default, approve, group, score):
    """group_rates()-shaped table built from a boolean approve mask instead of
    (score, shared_threshold) -- per-group thresholds don't reduce to one
    scalar group_rates can take directly. mean_predicted/calibration_gap are
    still reported from the real scores, only approve is taken as given."""
    y_default = np.asarray(y_default, dtype=float)
    score = np.asarray(score, dtype=float)
    approve = np.asarray(approve, dtype=bool)
    group = np.asarray(group)
    y_good = 1 - y_default
    rows = []
    for g in pd.unique(group):
        m = group == g
        good_m, bad_m = m & (y_good == 1), m & (y_good == 0)
        rows.append({
            "group": g, "n": int(m.sum()),
            "base_default_rate": float(y_default[m].mean()),
            "approval_rate": float(approve[m].mean()),
            "tpr": float(approve[good_m].mean()) if good_m.any() else float("nan"),
            "fpr": float(approve[bad_m].mean()) if bad_m.any() else float("nan"),
            "fnr": float(1 - approve[good_m].mean()) if good_m.any() else float("nan"),
            "mean_predicted": float(score[m].mean()),
        })
    out = pd.DataFrame(rows).set_index("group").sort_index()
    out["calibration_gap"] = out["mean_predicted"] - out["base_default_rate"]
    return out


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)

    print("=== unit economics (reused from Phase 5) ===")
    econ = unit_economics()
    LGD_RATE, MARGIN_RATE = econ["lgd_rate"], econ["margin_rate"]
    print(f"LGD rate: {LGD_RATE:.3f}   margin rate: {MARGIN_RATE:.3f}\n")

    geo_table = None
    if os.path.exists(GEO_CACHE_PATH):
        geo_table = load_zip3_race_proxy(GEO_CACHE_PATH)
        n_insufficient = (geo_table["race_proxy_group"] == "insufficient_data").sum()
        print(f"loaded geography race proxy: {len(geo_table):,} zip3 rows from {GEO_CACHE_PATH} "
              f"({n_insufficient} flagged insufficient_data)\n")
    else:
        print(f"no geography race proxy found at {GEO_CACHE_PATH} -- run "
              f"`python -m fairness.geo_proxy` (needs LC_CENSUS_API_KEY and real, "
              f"unproxied internet access) to add the geography dimension. "
              f"Continuing with the two class proxies only.\n")

    extra_cols = ["zip_code"] if geo_table is not None else None
    extra_categorical = {"zip_code"} if geo_table is not None else None
    df = load_modelling_frame(extra_cols=extra_cols, extra_categorical=extra_categorical)
    print(f"{len(df):,} loans\n")

    all_rates, all_gaps = [], []
    geo_excluded_test_total = 0
    for year, outer_train, test, fit, calib, cal_year in outer_folds(df):
        print(f"fold {year}: model fit {len(fit):,} (to {cal_year - 1}) | "
              f"calibrator/threshold {len(calib):,} ({cal_year}) | test {len(test):,} ({year})")

        income_calib, income_test = income_quantile_labels(calib["annual_inc"], test["annual_inc"])
        emp_calib, emp_test = emp_length_labels(calib["emp_length"]), emp_length_labels(test["emp_length"])
        keep_all_calib, keep_all_test = np.ones(len(calib), dtype=bool), np.ones(len(test), dtype=bool)
        proxies = [("income_quintile", income_calib.values, income_test.values, keep_all_calib, keep_all_test),
                   ("emp_length", emp_calib.values, emp_test.values, keep_all_calib, keep_all_test)]
        if geo_table is not None:
            geo_calib_raw = race_proxy_labels(calib["zip_code"], geo_table)
            geo_test_raw = race_proxy_labels(test["zip_code"], geo_table)
            # "insufficient_data" is a Census-side DATA-QUALITY flag (a zip3
            # too small, or entirely outside ACS's ZCTA universe -- e.g.
            # APO/FPO military zips), not a race/ethnicity label, and its N
            # here is a few dozen loans per 100k-loan test fold. Keeping it
            # as a "group" produced exactly the pathology this excludes: its
            # baseline approval rate was a noisy 100% in most folds (a tiny-N
            # artifact, not a finding), and a per-group mitigation threshold
            # fit on an even smaller calibration-cohort slice of it sometimes
            # made the OVERALL demographic-parity gap WORSE than doing
            # nothing -- the opposite of what a mitigation is for. Excluded
            # from the gap/mitigation analysis; the exclusion count is
            # printed and written to the summary so it's visible, not
            # silently dropped from the data.
            geo_keep_calib = (geo_calib_raw != "insufficient_data").values
            geo_keep_test = (geo_test_raw != "insufficient_data").values
            print(f"   geo_race_proxy: excluding {(~geo_keep_calib).sum()} calib / "
                  f"{(~geo_keep_test).sum()} test loans with insufficient_data zip3s "
                  f"from the fairness comparison (kept as a data-quality footnote, not a group)")
            geo_excluded_test_total += int((~geo_keep_test).sum())
            proxies.append(("geo_race_proxy", geo_calib_raw.values[geo_keep_calib],
                            geo_test_raw.values[geo_keep_test], geo_keep_calib, geo_keep_test))

        for mname in MODELS:
            p_calib, p_test = _fit_and_calibrate(fit, calib, test, mname)
            y_calib, amt_calib = calib["default_window"].values, calib["loan_amnt"].values
            y_test, amt_test = test["default_window"].values, test["loan_amnt"].values

            shared_thr, _ = best_threshold(y_calib, p_calib, amt_calib, LGD_RATE, MARGIN_RATE)

            for proxy_name, g_calib, g_test, mask_calib, mask_test in proxies:
                rates, gaps = run_proxy(proxy_name, g_calib, g_test,
                                        y_calib[mask_calib], y_test[mask_test],
                                        p_calib[mask_calib], p_test[mask_test],
                                        amt_calib[mask_calib], amt_test[mask_test],
                                        shared_thr, year, mname, LGD_RATE, MARGIN_RATE)
                all_rates.extend(rates)
                all_gaps.extend(gaps)

            gaps_df = pd.DataFrame(all_gaps)
            base = gaps_df[(gaps_df.fold == year) & (gaps_df.model == mname) & (gaps_df.scenario == "baseline")]
            for _, r in base.iterrows():
                print(f"   {mname:13s} {r['proxy']:16s} baseline: dp_gap={r['dp_gap']:.3f} "
                      f"(DI ratio {r['disparate_impact_ratio']:.2f})  eo_tpr_gap={r['eo_tpr_gap']:.3f}  "
                      f"eo_fpr_gap={r['eo_fpr_gap']:.3f}")
        print()

        pd.DataFrame(all_rates).to_csv(OUT_RATES, index=False)
        pd.DataFrame(all_gaps).to_csv(OUT_GAPS, index=False)

    gaps = pd.DataFrame(all_gaps)
    print("=== across folds (mean, sd) -- baseline gaps by proxy and model ===")
    for proxy in gaps["proxy"].unique():
        for m in gaps["model"].unique():
            b = gaps[(gaps.proxy == proxy) & (gaps.model == m) & (gaps.scenario == "baseline")]
            print(f"\n{proxy} / {m}:")
            print(f"  demographic parity gap : {b['dp_gap'].mean():.3f} (sd {b['dp_gap'].std():.3f})")
            print(f"  disparate impact ratio : {b['disparate_impact_ratio'].mean():.2f} "
                  f"(sd {b['disparate_impact_ratio'].std():.2f})")
            print(f"  equal-opportunity gap  : {b['eo_tpr_gap'].mean():.3f} (sd {b['eo_tpr_gap'].std():.3f})")

    print("\n=== mitigation cost (mean $/applicant vs. the profit-maximising shared threshold) ===")
    for proxy in gaps["proxy"].unique():
        for m in gaps["model"].unique():
            for scenario in ("demographic_parity", "equal_opportunity"):
                s = gaps[(gaps.proxy == proxy) & (gaps.model == m) & (gaps.scenario == scenario)]
                print(f"  {proxy:16s} {m:13s} {scenario:20s} "
                      f"cost=${s['cost_vs_baseline'].mean():.2f} (sd {s['cost_vs_baseline'].std():.2f})  "
                      f"resulting dp_gap={s['dp_gap'].mean():.3f}  eo_tpr_gap={s['eo_tpr_gap'].mean():.3f}")

    proxy_label = " / ".join(gaps["proxy"].unique())
    with open(OUT_SUMMARY, "w") as f:
        f.write(f"# Phase 6 fairness audit: {proxy_label}\n\n")
        f.write("Baseline gaps at the Phase 5 cost-optimal (\"realistic\") threshold, "
                "mean +/- sd across 4 folds:\n\n")
        for proxy in gaps["proxy"].unique():
            f.write(f"## {proxy}\n\n")
            for m in gaps["model"].unique():
                b = gaps[(gaps.proxy == proxy) & (gaps.model == m) & (gaps.scenario == "baseline")]
                f.write(f"- **{m}**: demographic parity gap {b['dp_gap'].mean():.3f} "
                        f"(sd {b['dp_gap'].std():.3f}), disparate impact ratio "
                        f"{b['disparate_impact_ratio'].mean():.2f}, equal-opportunity gap "
                        f"{b['eo_tpr_gap'].mean():.3f} (sd {b['eo_tpr_gap'].std():.3f})\n")
            f.write("\n")
        f.write("Mitigation cost ($/applicant vs. the profit-maximising shared threshold):\n\n")
        for proxy in gaps["proxy"].unique():
            for m in gaps["model"].unique():
                for scenario in ("demographic_parity", "equal_opportunity"):
                    s = gaps[(gaps.proxy == proxy) & (gaps.model == m) & (gaps.scenario == scenario)]
                    f.write(f"- {proxy} / {m} / {scenario}: "
                            f"${s['cost_vs_baseline'].mean():.2f} (sd ${s['cost_vs_baseline'].std():.2f})\n")
        f.write(
            "\nNote on the negative-cost rows above: with n=4 folds and a cost sd "
            "1-3x the size of the mean, a negative mean here means \"not "
            "distinguishable from zero given this much data\", not \"the "
            "mitigation is profitable\" -- the sign flips fold to fold (see "
            "fairness_gaps.csv). It is NOT evidence that fairness constraints are "
            "generally free; it is a reminder that a global-pooled Platt "
            "calibration is not guaranteed to be equally well-calibrated in every "
            "subgroup, and a per-group threshold can incidentally correct some of "
            "that miscalibration as a side effect of matching approval rate or TPR.\n"
        )
        if "geo_race_proxy" in gaps["proxy"].unique():
            n_insufficient = int((geo_table["race_proxy_group"] == "insufficient_data").sum())
            f.write(
                "\nNote on geo_race_proxy specifically: every number under this proxy is "
                "a statement about a ZIP3's Census-measured demographic composition, NOT "
                "about any individual borrower's race -- there is no borrower surname in "
                "LC's public data, so this is a geography-only proxy, weaker than the "
                "surname+geography BISG method regulators use, and it inherits the "
                "ecological fallacy: a borrower in a majority-White zip3 who is Black (or "
                "the reverse) is silently assigned the zip3's plurality group, not their "
                "own. See fairness/geo_proxy.py's docstring for the full reasoning. "
                f"{n_insufficient} of {len(geo_table):,} zip3 prefixes were flagged "
                "insufficient_data (population under 500 in the ACS estimate, or no "
                "matching ZCTA data at all). Loans in those zip3s are excluded from the "
                "gap/mitigation numbers above entirely, not just left unlabeled -- "
                f"{geo_excluded_test_total} test-cohort loans across all 4 folds "
                "(roughly 0.04% of the dataset). insufficient_data is a Census DATA-"
                "QUALITY flag, not a race/ethnicity group, and its N (a few dozen loans "
                "per 100k-loan fold) made it behave like one: a noisy ~100% baseline "
                "approval rate, and a per-group mitigation threshold fit on an even "
                "smaller calibration-cohort slice of it that sometimes made the OVERALL "
                "demographic-parity gap WORSE than doing nothing at all.\n"
            )

    print(f"\nwrote {OUT_RATES}, {OUT_GAPS}, {OUT_SUMMARY}")
