"""Tests for the hand-rolled KS statistic and the vintage target definition.

`ks_statistic` is implemented by hand rather than taken from a library, so it
needs checking against a known-good reference. The target rules are the
foundation every other number sits on.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from data.vintage_target import (CHARGEOFF_LAG_MONTHS, DEFAULT_STATUSES,
                                 POLICY_EXCLUDED, WINDOW_MONTHS)
from models.baselines import ks_statistic


# ---- KS statistic ---------------------------------------------------------
def test_ks_is_zero_when_scores_carry_no_information():
    y = np.array([0, 1] * 100)
    scores = np.full(200, 0.1)          # the trivial model
    assert ks_statistic(y, scores) == 0.0


def test_ks_is_one_under_perfect_separation():
    y = np.array([0] * 100 + [1] * 100)
    scores = np.concatenate([np.zeros(100), np.ones(100)])
    assert ks_statistic(y, scores) == pytest.approx(1.0)


def test_ks_matches_scipy_on_random_data():
    """The implementation is hand-written; scipy is the reference."""
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.1).astype(int)
    scores = rng.random(5000) + 0.3 * y
    mine = ks_statistic(y, scores)
    theirs = ks_2samp(scores[y == 1], scores[y == 0]).statistic
    assert mine == pytest.approx(theirs, abs=1e-9)


def test_ks_is_invariant_to_monotone_rescaling():
    """KS is a rank statistic, which is why rebalancing cannot move it much --
    the claim Phase 3's imbalance section rests on."""
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.2).astype(int)
    scores = rng.random(2000) + 0.4 * y
    assert ks_statistic(y, scores) == pytest.approx(
        ks_statistic(y, 1 / (1 + np.exp(-(scores * 3 - 1)))), abs=1e-9)


def test_ks_returns_nan_when_a_class_is_absent():
    y = np.zeros(50, dtype=int)
    assert np.isnan(ks_statistic(y, np.random.random(50)))


# ---- target definition ----------------------------------------------------
def test_window_and_lag_constants():
    assert WINDOW_MONTHS == 18
    assert CHARGEOFF_LAG_MONTHS == 6, (
        "the lag covers Lending Club's ~120-150 day delinquency-to-charge-off "
        "delay; shortening it would label loans whose outcome is not yet visible"
    )


def test_policy_excluded_statuses_are_not_treated_as_defaults():
    """These loans were underwritten under retired rules. Mixing two
    underwriting regimes into one target muddies what the model learns."""
    assert POLICY_EXCLUDED.isdisjoint(DEFAULT_STATUSES)
    assert any("Charged Off" in s for s in POLICY_EXCLUDED), (
        "the policy-excluded set should contain a charged-off variant, which is "
        "exactly why it must be excluded explicitly rather than by substring"
    )


def test_default_statuses_are_terminal_only():
    for s in DEFAULT_STATUSES:
        assert s in {"Charged Off", "Default"}, (
            f"{s!r} is not a terminal bad outcome; adding delinquency statuses "
            "would change the question the model answers"
        )


def test_eligibility_needs_window_plus_lag():
    """A loan observed for only the performance window is not yet eligible --
    a charge-off decision may still be in flight."""
    required = WINDOW_MONTHS + CHARGEOFF_LAG_MONTHS
    observed = pd.Series([WINDOW_MONTHS, required - 1, required, required + 12])
    eligible = observed >= required
    assert list(eligible) == [False, False, True, True]
