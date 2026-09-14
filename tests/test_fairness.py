"""Phase 6: fairness metrics, and the demographic-parity / equal-opportunity
impossibility result demonstrated numerically rather than asserted.

The centerpiece test (test_impossibility_*) constructs two groups with
different base default rates and a classifier with real but imperfect
discrimination, then checks the textbook result (Chouldechova 2017; Kleinberg,
Mullainathan & Raghavan 2016): a single shared threshold cannot generally
equalize both approval rate and error rates across groups when base rates
differ, and forcing one criterion via per-group thresholds worsens the other.
A control case (equal base rates) checks the tension actually traces to the
base-rate difference and isn't an artifact of the metric code."""
import numpy as np
import pytest

from fairness.metrics import (
    apply_group_thresholds,
    demographic_parity_gap,
    equalized_odds_gap,
    group_rates,
    per_group_thresholds,
    threshold_for_target_approval_rate,
    threshold_for_target_tpr,
)

SEED = 0


def test_group_rates_hand_computed():
    """4 applicants, 2 groups, threshold=0.5 -- every rate checked by hand."""
    # group A: applicant 0 (good, score .2 -> approve), applicant 1 (bad, score .8 -> reject)
    # group B: applicant 2 (good, score .7 -> reject), applicant 3 (bad, score .3 -> approve)
    y_default = [0, 1, 0, 1]
    score = [0.2, 0.8, 0.7, 0.3]
    group = ["A", "A", "B", "B"]

    rates = group_rates(y_default, score, threshold=0.5, group=group)

    assert rates.loc["A", "n"] == 2
    assert rates.loc["A", "base_default_rate"] == pytest.approx(0.5)
    assert rates.loc["A", "approval_rate"] == pytest.approx(0.5)  # 1 of 2 approved
    assert rates.loc["A", "tpr"] == pytest.approx(1.0)   # the 1 good applicant WAS approved
    assert rates.loc["A", "fpr"] == pytest.approx(0.0)   # the 1 bad applicant was rejected
    assert rates.loc["A", "fnr"] == pytest.approx(0.0)

    assert rates.loc["B", "approval_rate"] == pytest.approx(0.5)  # 1 of 2 approved
    assert rates.loc["B", "tpr"] == pytest.approx(0.0)   # the 1 good applicant was REJECTED
    assert rates.loc["B", "fpr"] == pytest.approx(1.0)   # the 1 bad applicant was mistakenly approved
    assert rates.loc["B", "fnr"] == pytest.approx(1.0)

    assert rates.loc["A", "mean_predicted"] == pytest.approx(0.5)
    assert rates.loc["A", "calibration_gap"] == pytest.approx(0.5 - 0.5)


def test_demographic_parity_gap_and_disparate_impact_ratio():
    import pandas as pd
    rates = pd.DataFrame({
        "approval_rate": [0.90, 0.60, 0.75],
    }, index=["A", "B", "C"])
    out = demographic_parity_gap(rates)
    assert out["gap"] == pytest.approx(0.30)
    assert out["disparate_impact_ratio"] == pytest.approx(0.60 / 0.90)
    assert out["most_favored"] == "A"
    assert out["least_favored"] == "B"


def test_equalized_odds_gap():
    import pandas as pd
    rates = pd.DataFrame({
        "tpr": [0.95, 0.70],
        "fpr": [0.10, 0.30],
    }, index=["A", "B"])
    out = equalized_odds_gap(rates)
    assert out["tpr_gap"] == pytest.approx(0.25)
    assert out["fpr_gap"] == pytest.approx(0.20)
    assert out["worst_tpr_group"] == "B"
    assert out["worst_fpr_group"] == "B"


def test_threshold_for_target_approval_rate_hits_target():
    rng = np.random.default_rng(SEED)
    score = rng.uniform(0, 1, 5000)
    for target in (0.10, 0.5, 0.9):
        tau = threshold_for_target_approval_rate(score, target)
        achieved = (score <= tau).mean()
        assert achieved == pytest.approx(target, abs=0.01)


def test_threshold_for_target_tpr_hits_target_among_good_only():
    rng = np.random.default_rng(SEED + 1)
    n = 6000
    y_default = (rng.uniform(0, 1, n) < 0.2).astype(int)
    # score correlated with y_default but noisy, so tpr != 1 or 0 trivially
    score = np.clip(y_default * 0.5 + rng.uniform(0, 0.6, n), 0, 1)
    for target in (0.3, 0.7, 0.95):
        tau = threshold_for_target_tpr(y_default, score, target)
        good_scores = score[y_default == 0]
        achieved = (good_scores <= tau).mean()
        assert achieved == pytest.approx(target, abs=0.02)


def _two_group_population(rng, n_per_group, base_rate_a, base_rate_b, separation=0.35):
    """Two groups, each with real (imperfect) score/outcome correlation, but
    different base default rates. `separation` controls how much scores for
    defaulters are shifted up relative to non-defaulters -- 0 would make the
    classifier useless (uninformative test), 1 would make it near-perfect
    (base rates wouldn't matter, also an uninformative test). 0.35 gives a
    real but far-from-perfect classifier, the realistic case."""
    def make_group(n, base_rate):
        y = (rng.uniform(0, 1, n) < base_rate).astype(int)
        score = np.clip(rng.uniform(0, 1 - separation, n) + y * separation, 0, 1)
        return y, score

    y_a, s_a = make_group(n_per_group, base_rate_a)
    y_b, s_b = make_group(n_per_group, base_rate_b)
    y = np.concatenate([y_a, y_b])
    score = np.concatenate([s_a, s_b])
    group = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    return y, score, group


def test_impossibility_different_base_rates_creates_real_tradeoff():
    rng = np.random.default_rng(SEED + 2)
    n = 4000
    y, score, group = _two_group_population(rng, n, base_rate_a=0.10, base_rate_b=0.30)

    # Baseline: one shared threshold approving ~80% of the population overall.
    shared_thr = threshold_for_target_approval_rate(score, 0.80)
    baseline = group_rates(y, score, shared_thr, group)
    baseline_dp = demographic_parity_gap(baseline)
    baseline_eo = equalized_odds_gap(baseline)

    # With different base rates, a SINGLE threshold cannot land on zero for
    # both criteria at once -- at least one gap must be real.
    assert baseline_dp["gap"] > 0.02 or baseline_eo["tpr_gap"] > 0.02

    # --- Force demographic parity: per-group thresholds hitting the SAME
    # population-wide approval rate (0.80) in each group individually.
    # Per-group thresholds don't reduce to one (score, threshold) pair, so
    # rates are rebuilt from the resulting approve mask directly.
    dp_thresholds = per_group_thresholds(y, score, group, mode="approval_rate", target=0.80)
    dp_approve = apply_group_thresholds(score, group, dp_thresholds, shared_thr)
    dp_rates = _rates_from_approve_mask(y, dp_approve, group)
    dp_gap = demographic_parity_gap(dp_rates)
    dp_eo = equalized_odds_gap(dp_rates)
    assert dp_gap["gap"] < 0.01  # parity achieved by construction
    # Forcing equal approval rates across groups with different true risk
    # must widen the equal-opportunity (TPR) gap relative to just leaving
    # both groups on their own equal-opportunity-matched footing.
    assert dp_eo["tpr_gap"] > 0.03

    # --- Force equal opportunity: per-group thresholds hitting the SAME
    # population-wide TPR among each group's own creditworthy applicants.
    baseline_tpr = baseline["tpr"].mean()
    eo_thresholds = per_group_thresholds(y, score, group, mode="tpr", target=baseline_tpr)
    eo_approve = apply_group_thresholds(score, group, eo_thresholds, shared_thr)
    eo_rates = _rates_from_approve_mask(y, eo_approve, group)
    eo_gap = equalized_odds_gap(eo_rates)
    eo_dp = demographic_parity_gap(eo_rates)
    assert eo_gap["tpr_gap"] < 0.01  # equal opportunity achieved by construction
    # Forcing equal TPR across groups with different true risk must widen
    # the demographic-parity (approval-rate) gap.
    assert eo_dp["gap"] > 0.03


def test_impossibility_control_equal_base_rates_no_tradeoff():
    """Same setup, but both groups share the SAME base rate. Here a single
    threshold, and either mitigation, should already sit near both criteria
    at once -- confirming the tension above comes from the base-rate gap,
    not from a bug in the metric or threshold-fitting code."""
    rng = np.random.default_rng(SEED + 3)
    n = 4000
    y, score, group = _two_group_population(rng, n, base_rate_a=0.20, base_rate_b=0.20)

    shared_thr = threshold_for_target_approval_rate(score, 0.80)
    baseline = group_rates(y, score, shared_thr, group)
    assert demographic_parity_gap(baseline)["gap"] < 0.02
    assert equalized_odds_gap(baseline)["tpr_gap"] < 0.02

    dp_thresholds = per_group_thresholds(y, score, group, mode="approval_rate", target=0.80)
    dp_approve = apply_group_thresholds(score, group, dp_thresholds, shared_thr)
    dp_rates = _rates_from_approve_mask(y, dp_approve, group)
    assert equalized_odds_gap(dp_rates)["tpr_gap"] < 0.02  # nothing to trade off


def _rates_from_approve_mask(y_default, approve, group):
    """Rebuild a group_rates-shaped table from a boolean approve mask rather
    than a shared threshold, since per-group thresholds don't reduce to a
    single (score, threshold) pair group_rates can take directly."""
    import pandas as pd
    y_default = np.asarray(y_default, dtype=float)
    y_good = 1 - y_default
    rows = []
    for g in pd.unique(group):
        m = np.asarray(group) == g
        good_m = m & (y_good == 1)
        bad_m = m & (y_good == 0)
        rows.append({
            "group": g,
            "n": int(m.sum()),
            "base_default_rate": float(y_default[m].mean()),
            "approval_rate": float(approve[m].mean()),
            "tpr": float(approve[good_m].mean()) if good_m.any() else float("nan"),
            "fpr": float(approve[bad_m].mean()) if bad_m.any() else float("nan"),
            "fnr": float(1 - approve[good_m].mean()) if good_m.any() else float("nan"),
        })
    return pd.DataFrame(rows).set_index("group").sort_index()
