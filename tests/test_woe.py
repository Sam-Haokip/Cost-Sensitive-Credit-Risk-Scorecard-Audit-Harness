"""Tests for the Weight-of-Evidence encoder.

The first test in this file is a regression test for a bug that shipped: the
encoder silently turned 11 of 58 usable features into constants because quantile
binning collapses on zero-inflated columns. It was found by reading the code, not
by anything failing, which is the reason this file exists.
"""
import numpy as np
import pandas as pd
import pytest

from features.woe import MISSING_LABEL, WOEEncoder


def _frame(n=4000, seed=0):
    return np.random.default_rng(seed), n


def test_zero_inflated_column_does_not_collapse():
    """REGRESSION: a column where one value holds most of the mass must still
    be binned, not folded into a single bucket.

    Quantile binning puts every quantile on the modal value, the edges
    deduplicate below three, and the pre-fix encoder returned None -- which
    routed every non-null value into one label, making the feature constant.
    On the real data this destroyed pub_rec, tax_liens, acc_now_delinq and
    eight others.
    """
    rng, n = _frame()
    # 95% zeros, 5% ones-and-up: exactly the shape of a public-records count.
    x = np.where(rng.random(n) < 0.95, 0, rng.integers(1, 5, n))
    # Non-zero rows default at 3x the rate, so the split carries real signal.
    y = rng.random(n) < np.where(x > 0, 0.30, 0.10)

    enc = WOEEncoder().fit(pd.DataFrame({"pub_rec": x}), pd.Series(y.astype(int)))

    assert enc.edges_["pub_rec"] is not None, "zero-inflated column was discarded"
    assert len(enc.woe_["pub_rec"]) >= 2, "column collapsed to a single bin"

    out = enc.transform(pd.DataFrame({"pub_rec": [0, 0, 3, 4]}))["pub_rec"]
    assert out.iloc[0] == out.iloc[1], "identical inputs must encode identically"
    assert out.iloc[0] != out.iloc[2], "zero and non-zero must not share a weight"
    # Higher risk must carry the lower weight: WoE is log(non-events / events).
    assert out.iloc[2] < out.iloc[0]


def test_all_null_column_is_handled_not_crashed():
    rng, n = _frame()
    df = pd.DataFrame({"never_reported": [np.nan] * n})
    y = pd.Series((rng.random(n) < 0.1).astype(int))
    enc = WOEEncoder().fit(df, y)
    out = enc.transform(df)["never_reported"]
    assert np.isfinite(out).all(), "all-null column must not produce inf/nan"


def test_missing_gets_its_own_learned_weight():
    """Phase 1's core finding: absence is informative, so it must not be
    imputed into the middle of the distribution."""
    rng, n = _frame(6000)
    observed = rng.normal(50, 10, n)
    is_missing = rng.random(n) < 0.4
    x = np.where(is_missing, np.nan, observed)
    # Missing rows are the SAFE ones here (event never happened).
    y = (rng.random(n) < np.where(is_missing, 0.03, 0.20)).astype(int)

    enc = WOEEncoder().fit(pd.DataFrame({"mths_since": x}), pd.Series(y))
    assert MISSING_LABEL in enc.woe_["mths_since"], "missing has no bin of its own"

    w_missing = enc.woe_["mths_since"][MISSING_LABEL]
    others = [v for k, v in enc.woe_["mths_since"].items() if k != MISSING_LABEL]
    assert w_missing > max(others), (
        "missing rows default less often, so their weight must be the highest; "
        "a median-imputing pipeline would place them mid-distribution instead"
    )


def test_unseen_category_encodes_as_neutral():
    train = pd.DataFrame({"state": ["CA"] * 300 + ["NY"] * 300})
    y = pd.Series([0, 1] * 300)
    enc = WOEEncoder().fit(train, y)
    out = enc.transform(pd.DataFrame({"state": ["WY"]}))["state"]
    assert out.iloc[0] == 0.0, "a category never seen in training must be neutral"


def test_transform_is_pure_and_repeatable():
    """The encoder must be fit on training data only. Calling transform must
    not update any learned state, or the fold discipline is a fiction."""
    rng, n = _frame()
    df = pd.DataFrame({"x": rng.normal(0, 1, n)})
    y = pd.Series((rng.random(n) < 0.2).astype(int))
    enc = WOEEncoder().fit(df, y)

    before = {k: dict(v) for k, v in enc.woe_.items()}
    first = enc.transform(df)
    second = enc.transform(df)
    assert enc.woe_ == before, "transform mutated the fitted weights"
    pd.testing.assert_frame_equal(first, second)


def test_no_infinite_weights_when_a_bin_has_no_events():
    """Smoothing exists so a pure bin cannot produce +/-inf, which would make
    the downstream logistic regression unfittable."""
    x = np.concatenate([np.zeros(500), np.ones(500)])
    y = np.concatenate([np.zeros(500), np.ones(500)])   # perfectly separable
    enc = WOEEncoder().fit(pd.DataFrame({"x": x}), pd.Series(y.astype(int)))
    assert np.isfinite(list(enc.woe_["x"].values())).all()


def test_information_value_ranks_a_real_signal_above_noise():
    rng, n = _frame(6000)
    signal = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + 1.8 * signal)))).astype(int)
    df = pd.DataFrame({"signal": signal, "noise": rng.normal(0, 1, n)})
    enc = WOEEncoder().fit(df, pd.Series(y))
    iv = enc.information_values()
    assert iv["signal"] > iv["noise"]
    assert iv.index[0] == "signal"


@pytest.mark.parametrize("n_bins", [5, 10, 20])
def test_bin_count_is_respected_on_a_continuous_column(n_bins):
    rng, n = _frame(8000)
    df = pd.DataFrame({"x": rng.normal(0, 1, n)})
    y = pd.Series((rng.random(n) < 0.2).astype(int))
    enc = WOEEncoder(n_bins=n_bins, min_bin_fraction=0.0).fit(df, y)
    assert len(enc.woe_["x"]) <= n_bins
