"""Phase 5: the threshold search is exact (sorted cumulative sum), not a grid
search, so it needs to be checked against brute force -- an off-by-one in the
cumulative sum would be easy to miss by eye and would silently pick the wrong
threshold on every fold."""
import numpy as np
import pytest

from evaluation.decisioning import best_threshold, profit_at, profit_curve

SEED = 0


def _brute_force_best(y, p, amt, lgd_rate, margin_rate, grid):
    best_t, best_profit = None, -np.inf
    for t in grid:
        profit, _ = profit_at(y, p, amt, lgd_rate, margin_rate, t)
        if profit > best_profit:
            best_t, best_profit = t, profit
    return best_t, best_profit


def test_best_threshold_matches_brute_force_search():
    rng = np.random.default_rng(SEED)
    n = 2000
    p = rng.uniform(0, 1, n)
    # Outcomes correlated with p, not independent of it -- an unrelated y and p
    # would make every threshold roughly equally (un)profitable and the test
    # would pass by accident regardless of whether the search is correct.
    y = (rng.uniform(0, 1, n) < p).astype(int)
    amt = rng.uniform(1000, 40000, n)
    lgd_rate, margin_rate = 0.65, 0.16

    thr_exact, profit_exact = best_threshold(y, p, amt, lgd_rate, margin_rate)
    thr_grid, profit_grid = _brute_force_best(y, p, amt, lgd_rate, margin_rate,
                                              np.linspace(0, 1, 4001))

    # The exact search must be at least as good as any grid point (it checks
    # every real cut-point, the grid only checks 4001 of them).
    assert profit_exact >= profit_grid - 1e-6
    # And the grid's best guess should land close to the exact answer.
    assert abs(thr_exact - thr_grid) < 0.01


def test_profit_curve_cumulative_sum_matches_direct_evaluation():
    """profit_curve's cumulative-sum trick must agree with independently
    evaluating profit_at() at the same threshold -- this is exactly the kind
    of two-implementations-of-the-same-idea check that catches an off-by-one
    the fast path could hide."""
    rng = np.random.default_rng(SEED + 1)
    n = 500
    p = rng.uniform(0, 1, n)
    y = rng.integers(0, 2, n)
    amt = rng.uniform(1000, 20000, n)
    lgd_rate, margin_rate = 0.7, 0.15

    thresholds, curve_profit = profit_curve(y, p, amt, lgd_rate, margin_rate)
    for i in [0, n // 4, n // 2, 3 * n // 4, n - 1, n]:  # includes "approve nobody" and "approve all"
        direct_profit, _ = profit_at(y, p, amt, lgd_rate, margin_rate, thresholds[i])
        assert curve_profit[i] == pytest.approx(direct_profit, abs=1e-9)


def test_all_defaults_means_reject_everyone_is_optimal():
    """If every loan in the population defaults, there is no profitable
    threshold above the minimum score -- approving nobody must win."""
    rng = np.random.default_rng(SEED + 2)
    n = 300
    p = rng.uniform(0, 1, n)
    y = np.ones(n, dtype=int)
    amt = np.full(n, 10_000.0)

    thr, profit = best_threshold(y, p, amt, lgd_rate=0.65, margin_rate=0.16)
    approved, approve_rate = profit_at(y, p, amt, 0.65, 0.16, thr)
    assert approve_rate == pytest.approx(0.0)
    assert profit == pytest.approx(0.0)


def test_no_defaults_means_approve_everyone_is_optimal():
    """If nothing ever defaults, rejecting anyone only forfeits margin --
    approving the whole population must win."""
    rng = np.random.default_rng(SEED + 3)
    n = 300
    p = rng.uniform(0, 1, n)
    y = np.zeros(n, dtype=int)
    amt = np.full(n, 10_000.0)

    thr, profit = best_threshold(y, p, amt, lgd_rate=0.65, margin_rate=0.16)
    _, approve_rate = profit_at(y, p, amt, 0.65, 0.16, thr)
    assert approve_rate == pytest.approx(1.0)
    assert profit == pytest.approx(0.16 * 10_000.0)


def test_threshold_separates_a_profitable_block_from_an_unprofitable_one():
    """A sanity check with a hand-computable answer. At lgd_rate=0.65,
    margin_rate=0.16 the break-even probability is margin/(margin+lgd) =
    0.1975: a homogeneous block of loans at p=0.10 is profitable to approve in
    expectation (0.16*0.90 - 0.65*0.10 = +0.079/dollar), a block at p=0.30 is
    not (0.16*0.70 - 0.65*0.30 = -0.083/dollar). The chosen threshold should
    land strictly between them, approving the first block and rejecting the
    second -- not, for instance, approving everything or nothing."""
    n_safe, n_risky = 400, 400
    p = np.concatenate([np.full(n_safe, 0.10), np.full(n_risky, 0.30)])
    rng = np.random.default_rng(SEED + 4)
    y = (rng.uniform(0, 1, n_safe + n_risky) < p).astype(int)
    amt = np.full(n_safe + n_risky, 5_000.0)

    thr, _ = best_threshold(y, p, amt, lgd_rate=0.65, margin_rate=0.16)
    # "Approve p <= threshold" with every safe loan sharing p=0.10 exactly
    # means the true optimum sits AT 0.10, not strictly above it -- there's no
    # data point in between to distinguish it from a point just above 0.10.
    assert 0.10 <= thr < 0.30
