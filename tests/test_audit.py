"""Phase 6: audit.py's own logic (income_quantile_labels, emp_length_labels,
run_proxy's wiring) tested the same way fairness/metrics.py's functions were --
these are new code with their own failure modes (fit/apply cohort separation,
out-of-range values, duplicate bin edges, missing-category handling) that
fairness/metrics.py's tests don't cover."""
import numpy as np
import pandas as pd
import pytest

from fairness.audit import emp_length_labels, income_quantile_labels, race_proxy_labels, run_proxy

SEED = 0


def test_income_quantile_labels_produces_n_groups_monotonic_in_income():
    rng = np.random.default_rng(SEED)
    fit_income = pd.Series(rng.uniform(20_000, 200_000, 5000))
    fit_labels, _ = income_quantile_labels(fit_income, fit_income, n=5)

    assert sorted(fit_labels.unique()) == [1, 2, 3, 4, 5]
    # Roughly equal-sized quantiles on a smooth distribution.
    counts = fit_labels.value_counts()
    assert counts.min() > 900  # ~1000 each, some slack for qcut boundary rounding

    # Monotonicity: sorting by income and taking the label must be non-decreasing.
    order = fit_income.sort_values().index
    labels_in_income_order = fit_labels.loc[order].values
    assert (np.diff(labels_in_income_order) >= 0).all()


def test_income_quantile_labels_fits_edges_on_fit_cohort_only():
    """The apply cohort's own distribution must NOT influence the edges -- only
    whether apply values fall inside bins fit on the fit cohort."""
    rng = np.random.default_rng(SEED + 1)
    fit_income = pd.Series(rng.uniform(20_000, 50_000, 2000))  # narrow range
    # Same values, repeated -- if edges were (wrongly) refit on apply_income
    # too, this would still pass; the real test is the out-of-range case below.
    _, apply_labels_same = income_quantile_labels(fit_income, fit_income, n=5)
    assert sorted(apply_labels_same.unique()) == [1, 2, 3, 4, 5]


def test_income_quantile_labels_widens_outer_edges_for_out_of_range_test_values():
    """A test-cohort income far outside the calibration cohort's observed range
    (later years, income drift) must land in the nearest quintile, not become
    NaN -- this is the entire point of widening the outer edges to +/-inf."""
    rng = np.random.default_rng(SEED + 2)
    fit_income = pd.Series(rng.uniform(30_000, 100_000, 3000))
    apply_income = pd.Series([1.0, 5_000.0, 500_000.0, 10_000_000.0])  # way outside fit range

    _, apply_labels = income_quantile_labels(fit_income, apply_income, n=5)

    assert not apply_labels.isna().any()
    assert apply_labels.iloc[0] == 1  # below everything -> lowest quintile
    assert apply_labels.iloc[1] == 1
    assert apply_labels.iloc[2] == 5  # above everything -> highest quintile
    assert apply_labels.iloc[3] == 5


def test_income_quantile_labels_tolerates_duplicate_edges():
    """Heavy ties in the fit cohort (e.g. many applicants reporting the exact
    same round-number income) can make qcut collapse bins via duplicates="drop"
    -- this must degrade to fewer groups, not crash or produce NaN."""
    fit_income = pd.Series([50_000.0] * 400 + [60_000.0] * 400 + list(range(70_000, 70_200)))
    apply_income = pd.Series([50_000.0, 60_000.0, 70_050.0, 45_000.0, 80_000.0])

    fit_labels, apply_labels = income_quantile_labels(fit_income, apply_income, n=5)

    assert not fit_labels.isna().any()
    assert not apply_labels.isna().any()
    assert fit_labels.nunique() <= 5
    assert fit_labels.min() >= 1


def test_emp_length_labels_fills_missing_with_explicit_string():
    s = pd.Series(["10+ years", None, "< 1 year", np.nan, "5 years"])
    out = emp_length_labels(s)
    assert out.tolist() == ["10+ years", "missing", "< 1 year", "missing", "5 years"]
    # dtype itself can be plain object or pandas' newer StringDtype depending on
    # the pandas version -- what matters is every element is a real python str.
    assert out.map(type).eq(str).all()


def test_emp_length_labels_is_idempotent_on_already_clean_input():
    s = pd.Series(["3 years", "7 years"])
    out = emp_length_labels(s)
    assert out.tolist() == ["3 years", "7 years"]


def test_race_proxy_labels_maps_zip3_prefix_to_census_group():
    zip3_table = pd.DataFrame({
        "zip3": ["190", "605"],
        "race_proxy_group": ["white_nonhispanic", "hispanic"],
    })
    zip_codes = pd.Series(["190xx", "605xx", "190xx"])
    out = race_proxy_labels(zip_codes, zip3_table)
    assert out.tolist() == ["white_nonhispanic", "hispanic", "white_nonhispanic"]


def test_race_proxy_labels_falls_back_for_null_zip_code():
    """Real data has exactly one loan with a null zip_code (out of 1.4M) --
    discovered while smoke-testing this function against the full dataset.
    A categorical Series' `.astype(str)` leaves a missing value as a real
    NaN rather than the string "nan", so this must not silently propagate
    NaN through .map() and out the other end uncaught."""
    zip3_table = pd.DataFrame({"zip3": ["190"], "race_proxy_group": ["white_nonhispanic"]})
    zip_codes = pd.Series(["190xx", None], dtype="category")
    out = race_proxy_labels(zip_codes, zip3_table)
    assert out.tolist() == ["white_nonhispanic", "insufficient_data"]
    assert not out.isna().any()


def test_race_proxy_labels_falls_back_for_zip3_missing_from_census_table():
    """A zip3 present in LC's data but absent from the fetched ACS table
    (can happen for territory zip3s) must not become NaN or crash -- it
    gets the same "insufficient_data" label geo_proxy.py uses for a zip3
    Census itself couldn't confidently characterize."""
    zip3_table = pd.DataFrame({"zip3": ["190"], "race_proxy_group": ["white_nonhispanic"]})
    zip_codes = pd.Series(["190xx", "999xx"])
    out = race_proxy_labels(zip_codes, zip3_table)
    assert out.tolist() == ["white_nonhispanic", "insufficient_data"]


# --- run_proxy wiring: an integration check on synthetic data with a known,
# engineered disparity. Reuses test_fairness.py's proven two-group construction
# (uniform score + y*separation) and threshold choice rather than inventing a
# new one -- a normal-noise version tried first produced a baseline with
# ~zero equal-opportunity gap by construction (identical score|y distribution
# across groups makes TPR/FPR group-invariant regardless of base rate), which
# is a real and correct possible outcome, just not the one this test wants to
# exercise: it was the test's synthetic data at fault, not run_proxy. ---

def _synthetic_two_group_scores(rng, n_per_group, base_rate_a, base_rate_b, separation=0.35):
    def make_group(n, base_rate):
        y = (rng.uniform(0, 1, n) < base_rate).astype(int)
        score = np.clip(rng.uniform(0, 1 - separation, n) + y * separation, 0, 1)
        return y, score

    y_a, p_a = make_group(n_per_group, base_rate_a)
    y_b, p_b = make_group(n_per_group, base_rate_b)
    y = np.concatenate([y_a, y_b])
    p = np.concatenate([p_a, p_b])
    group = np.array(["a"] * n_per_group + ["b"] * n_per_group)
    return y, p, group


def test_run_proxy_baseline_matches_group_rates_and_has_zero_cost():
    rng = np.random.default_rng(SEED + 3)
    n = 4000
    y_calib, p_calib, g_calib = _synthetic_two_group_scores(rng, n, 0.10, 0.30)
    y_test, p_test, g_test = _synthetic_two_group_scores(rng, n, 0.10, 0.30)
    amt_calib = np.full(2 * n, 10_000.0)
    amt_test = np.full(2 * n, 10_000.0)
    shared_thr = float(np.quantile(p_calib, 0.80))  # approve ~80% overall, like test_fairness.py

    rate_rows, gap_rows = run_proxy(
        "synthetic", g_calib, g_test, y_calib, y_test, p_calib, p_test,
        amt_calib, amt_test, shared_thr, fold=2099, model_name="test_model",
        lgd_rate=0.65, margin_rate=0.16,
    )

    baseline_gap = next(r for r in gap_rows if r["scenario"] == "baseline")
    assert baseline_gap["cost_vs_baseline"] == pytest.approx(0.0)
    assert baseline_gap["fold"] == 2099
    assert baseline_gap["model"] == "test_model"
    assert baseline_gap["proxy"] == "synthetic"

    baseline_rows = [r for r in rate_rows if r["scenario"] == "baseline"]
    assert {r["group"] for r in baseline_rows} == {"a", "b"}
    assert sum(r["n"] for r in baseline_rows) == 2 * n


def test_run_proxy_demographic_parity_mitigation_closes_the_approval_gap():
    rng = np.random.default_rng(SEED + 4)
    n = 4000
    y_calib, p_calib, g_calib = _synthetic_two_group_scores(rng, n, 0.10, 0.30)
    y_test, p_test, g_test = _synthetic_two_group_scores(rng, n, 0.10, 0.30)
    amt_calib = np.full(2 * n, 10_000.0)
    amt_test = np.full(2 * n, 10_000.0)
    shared_thr = float(np.quantile(p_calib, 0.80))

    _, gap_rows = run_proxy(
        "synthetic", g_calib, g_test, y_calib, y_test, p_calib, p_test,
        amt_calib, amt_test, shared_thr, fold=2099, model_name="test_model",
        lgd_rate=0.65, margin_rate=0.16,
    )
    by_scenario = {r["scenario"]: r for r in gap_rows}

    # With different base rates, a single shared threshold cannot land on
    # zero for both criteria at once -- at least one gap must be real (the
    # same "or", not "and", test_fairness.py's impossibility test uses, since
    # which gap shows up baseline depends on exactly where the threshold sits).
    assert by_scenario["baseline"]["dp_gap"] > 0.02 or by_scenario["baseline"]["eo_tpr_gap"] > 0.02
    # Demographic-parity mitigation must shrink the approval-rate gap close to
    # zero -- this follows directly from per_group_thresholds' definition
    # (each group's threshold is fit to hit the same target rate), so it's a
    # strong claim about the wiring, not a statistical one. The residual isn't
    # exactly zero: threshold_for_target_approval_rate is a finite-sample
    # quantile estimate (n=4000/group here), not an analytic inverse-CDF.
    assert by_scenario["demographic_parity"]["dp_gap"] < 0.03

    # Every mitigation row's cost is baseline profit minus its own profit --
    # check the arithmetic identity directly rather than trusting the formula.
    baseline_profit = by_scenario["baseline"]["profit_per_applicant"]
    for scenario in ("demographic_parity", "equal_opportunity"):
        row = by_scenario[scenario]
        assert row["cost_vs_baseline"] == pytest.approx(baseline_profit - row["profit_per_applicant"])


def test_run_proxy_equal_opportunity_mitigation_closes_the_tpr_gap():
    rng = np.random.default_rng(SEED + 5)
    n = 4000
    y_calib, p_calib, g_calib = _synthetic_two_group_scores(rng, n, 0.10, 0.30)
    y_test, p_test, g_test = _synthetic_two_group_scores(rng, n, 0.10, 0.30)
    amt_calib = np.full(2 * n, 10_000.0)
    amt_test = np.full(2 * n, 10_000.0)
    shared_thr = float(np.quantile(p_calib, 0.80))

    _, gap_rows = run_proxy(
        "synthetic", g_calib, g_test, y_calib, y_test, p_calib, p_test,
        amt_calib, amt_test, shared_thr, fold=2099, model_name="test_model",
        lgd_rate=0.65, margin_rate=0.16,
    )
    by_scenario = {r["scenario"]: r for r in gap_rows}

    # Same reasoning as above, applied to the TPR gap specifically: whichever
    # gap baseline leaves non-trivial, equal-opportunity mitigation must close
    # ITS gap close to zero by construction.
    assert by_scenario["equal_opportunity"]["eo_tpr_gap"] < 0.01
