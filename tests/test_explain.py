"""Phase 7: explainability/explain.py's numbers are only trustworthy if the
two hand-rolled pieces it exists BECAUSE OF are actually correct -- the exact
linear-SHAP closed form (checked against the real shap library, not just
against itself) and the categorical-dtype-safe permutation shuffle (a
regression test for a bug that crashed LightGBM's predict). Everything else
here is the usual synthetic-data sanity check: does each small function do
the one thing its docstring says it does."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from explainability.explain import (
    CORRELATION_FLAG_THRESHOLD,
    _display_value,
    _shuffle_column,
    format_applicant_explanation,
    global_importance,
    linear_shap_values,
    nearest_correlated_neighbor,
    permutation_importance_manual,
    pick_representative_applicants,
    rank_table,
)

SEED = 0


# ---- linear_shap_values -----------------------------------------------------
def test_linear_shap_values_reconstructs_the_logit_exactly():
    """The whole point of linear_shap_values is that shap contributions plus
    the base value must sum EXACTLY to the model's own raw logit -- that's
    what "additively separable, no approximation needed" means. If this
    doesn't hold to floating-point precision, the closed form is wrong."""
    rng = np.random.default_rng(SEED)
    n, d = 500, 6
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    y = (rng.uniform(size=n) < 0.3).astype(int)
    model = LogisticRegression().fit(X, y)

    shap_values, base_value = linear_shap_values(model, X, X)
    logit = model.decision_function(X)
    reconstructed = shap_values.sum(axis=1) + base_value
    np.testing.assert_allclose(reconstructed, logit, atol=1e-10)


def test_linear_shap_values_matches_shap_linear_explainer_with_full_background():
    """This is the actual bug this project found: shap.LinearExplainer's
    default Independent masker silently subsamples the background to 100
    rows, which measurably disagrees with the exact closed form. Forcing
    max_samples=len(data) must reproduce the closed form -- if this test
    ever fails, either the shap library changed its default behaviour again,
    or linear_shap_values itself regressed."""
    shap = pytest.importorskip("shap")
    rng = np.random.default_rng(SEED + 1)
    n, d = 300, 5
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    y = (rng.uniform(size=n) < 0.4).astype(int)
    model = LogisticRegression().fit(X, y)

    manual_values, manual_base = linear_shap_values(model, X, X)

    masker = shap.maskers.Independent(X, max_samples=len(X))
    explainer = shap.LinearExplainer(model, masker)
    lib_values = np.asarray(explainer.shap_values(X))
    lib_base = float(np.asarray(explainer.expected_value).reshape(-1)[-1])

    np.testing.assert_allclose(manual_values, lib_values, atol=1e-8)
    assert manual_base == pytest.approx(lib_base, abs=1e-8)


def test_linear_shap_values_disagrees_with_shap_default_100_row_subsample():
    """The negative-space check: with enough background rows and a background
    mean that genuinely differs from its first 100 rows, the library's
    DEFAULT (no max_samples override) must disagree with the exact closed
    form -- documenting the bug this module's docstring describes, not just
    the fix for it. If this ever stops disagreeing, the library's default
    changed and the module docstring's justification needs revisiting."""
    shap = pytest.importorskip("shap")
    rng = np.random.default_rng(SEED + 2)
    n, d = 2000, 4
    # Construct background so the first 100 rows have a visibly different
    # mean than the full population -- otherwise subsampling wouldn't matter.
    shift = np.zeros((n, d))
    shift[:100] += 5.0
    X = pd.DataFrame(rng.normal(size=(n, d)) + shift, columns=[f"f{i}" for i in range(d)])
    y = (rng.uniform(size=n) < 0.4).astype(int)
    model = LogisticRegression().fit(X, y)

    manual_values, _ = linear_shap_values(model, X, X.iloc[:50])
    explainer = shap.LinearExplainer(model, X)  # default masker, default max_samples=100
    lib_values = np.asarray(explainer.shap_values(X.iloc[:50]))

    assert np.abs(manual_values - lib_values).max() > 1e-3


# ---- tree_shap_values --------------------------------------------------------
def test_tree_shap_values_reconstructs_the_raw_margin():
    """Same additivity check as the linear case, but for LightGBM's exact
    TreeSHAP: base_value + sum(shap contributions) must equal the model's own
    raw_score margin for every row."""
    lgb = pytest.importorskip("lightgbm")
    pytest.importorskip("shap")
    from explainability.explain import tree_shap_values

    rng = np.random.default_rng(SEED + 3)
    n, d = 400, 5
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    y = (rng.uniform(size=n) < 0.35).astype(int)
    model = lgb.LGBMClassifier(n_estimators=20, num_leaves=7, random_state=SEED, verbose=-1)
    model.fit(X, y)

    shap_values, base_value = tree_shap_values(model, X)
    raw_margin = model.predict(X, raw_score=True)
    reconstructed = shap_values.sum(axis=1) + base_value
    np.testing.assert_allclose(reconstructed, raw_margin, atol=1e-6)


# ---- global_importance -------------------------------------------------------
def test_global_importance_sorts_descending_by_mean_abs_shap():
    shap_values = np.array([
        [1.0, -5.0, 0.0],
        [-1.0, 5.0, 0.0],
        [2.0, 0.0, 0.1],
    ])
    imp = global_importance(shap_values, ["a", "b", "c"])
    assert list(imp.index) == ["b", "a", "c"]
    assert imp.iloc[0] == pytest.approx((5.0 + 5.0 + 0.0) / 3)


# ---- _shuffle_column / permutation_importance_manual -------------------------
def test_shuffle_column_preserves_categorical_dtype():
    """The exact bug found in this project: rng.permutation(X[col].values)
    silently converts a pandas Categorical to a plain object array (via
    np.asarray internally), which then crashes LightGBM's predict with
    "categorical_feature do not match". _shuffle_column must come back with
    the SAME dtype (and the same category set) it started with."""
    X = pd.DataFrame({
        "cat_col": pd.Categorical(["a", "b", "c", "a", "b"], categories=["a", "b", "c"]),
        "num_col": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    rng = np.random.default_rng(SEED)
    out = _shuffle_column(X, "cat_col", rng)
    assert isinstance(out["cat_col"].dtype, pd.CategoricalDtype)
    assert list(out["cat_col"].cat.categories) == ["a", "b", "c"]
    # values are a permutation of the original, not corrupted
    assert sorted(out["cat_col"].tolist()) == sorted(X["cat_col"].tolist())
    # the numeric column is untouched
    pd.testing.assert_series_equal(out["num_col"], X["num_col"])


def test_shuffle_column_naive_numpy_permutation_would_strip_the_dtype():
    """Documents WHY _shuffle_column exists, not just that it works: the
    naive approach this module's docstring describes really does strip the
    categorical dtype on this pandas/numpy version combination."""
    col = pd.Categorical(["a", "b", "c"], categories=["a", "b", "c"])
    rng = np.random.default_rng(SEED)
    naive = rng.permutation(col)
    assert not isinstance(naive, pd.Categorical)


def test_permutation_importance_manual_ranks_informative_feature_above_noise():
    """A feature that actually determines y must show a bigger PR-AUC drop
    under shuffling than pure-noise features -- if this failed, the shuffling
    or scoring logic would be broken in a way that no synthetic edge case
    would catch."""
    rng = np.random.default_rng(SEED + 4)
    n = 3000
    signal = rng.normal(size=n)
    y = (signal + rng.normal(scale=0.3, size=n) > 0).astype(int)
    X = pd.DataFrame({
        "signal": signal,
        "noise1": rng.normal(size=n),
        "noise2": rng.normal(size=n),
    })

    def predict_fn(Xin):
        # oracle-ish predictor: only "signal" matters, matching how y was built
        return 1 / (1 + np.exp(-3 * Xin["signal"].values))

    imp = permutation_importance_manual(predict_fn, X, y, ["signal", "noise1", "noise2"],
                                         seed=SEED, n_repeats=5)
    assert imp.index[0] == "signal"
    assert imp["signal"] > imp["noise1"]
    assert imp["signal"] > imp["noise2"]


def test_permutation_importance_manual_handles_categorical_feature_without_crashing():
    """Integration-style regression test for the actual reported crash: a
    predict_fn that requires a categorical dtype (like LightGBM does) must
    not raise when that column is the one being shuffled."""
    rng = np.random.default_rng(SEED + 5)
    n = 500
    X = pd.DataFrame({
        "cat_col": pd.Categorical(rng.choice(["x", "y", "z"], size=n), categories=["x", "y", "z"]),
        "num_col": rng.normal(size=n),
    })
    y = (rng.uniform(size=n) < 0.3).astype(int)

    def predict_fn(Xin):
        if not isinstance(Xin["cat_col"].dtype, pd.CategoricalDtype):
            raise ValueError("train and valid dataset categorical_feature do not match.")
        return rng.uniform(size=len(Xin))  # score doesn't need to be meaningful for this check

    imp = permutation_importance_manual(predict_fn, X, y, ["cat_col", "num_col"], seed=SEED, n_repeats=2)
    assert set(imp.index) == {"cat_col", "num_col"}


# ---- rank_table / nearest_correlated_neighbor --------------------------------
def test_rank_table_flags_the_largest_rank_disagreement():
    shap_imp = pd.Series({"a": 10.0, "b": 8.0, "c": 1.0, "d": 0.5})
    perm_imp = pd.Series({"a": 0.10, "b": 0.01, "c": 0.09, "d": 0.02})
    rt = rank_table(shap_imp, perm_imp)
    # "b" is #2 by SHAP but #4 by permutation, and "c" is #3 by SHAP but #2 by
    # permutation -- both have the largest rank gap (2); the top row must be
    # one of them, sorted descending by rank_gap.
    assert rt.iloc[0]["rank_gap"] == 2
    assert rt.iloc[0].name in ("b", "c")


def test_nearest_correlated_neighbor_finds_the_correlated_partner():
    rng = np.random.default_rng(SEED + 6)
    n = 1000
    base = rng.normal(size=n)
    X = pd.DataFrame({
        "target": base,
        "partner": base * 0.9 + rng.normal(scale=0.1, size=n),  # highly correlated with target
        "unrelated": rng.normal(size=n),
    })
    neighbor, corr = nearest_correlated_neighbor("target", ["partner", "unrelated"], X)
    assert neighbor == "partner"
    assert abs(corr) >= CORRELATION_FLAG_THRESHOLD


def test_nearest_correlated_neighbor_returns_none_when_no_candidates():
    X = pd.DataFrame({"target": [1.0, 2.0, 3.0]})
    neighbor, corr = nearest_correlated_neighbor("target", [], X)
    assert neighbor is None
    assert corr == 0.0


def test_nearest_correlated_neighbor_excludes_the_feature_itself():
    rng = np.random.default_rng(SEED + 7)
    n = 200
    X = pd.DataFrame({"target": rng.normal(size=n), "other": rng.normal(size=n)})
    # top_features includes "target" itself -- must not be returned as its own neighbor
    neighbor, _ = nearest_correlated_neighbor("target", ["target", "other"], X)
    assert neighbor == "other"


# ---- pick_representative_applicants ------------------------------------------
def test_pick_representative_applicants_selects_correct_extremes_and_borderline():
    p_test = np.array([0.9, 0.05, 0.5, 0.51, 0.02, 0.95])
    threshold = 0.5
    picks = pick_representative_applicants(p_test, threshold)
    assert picks["clear_approve"] == 4  # lowest score, 0.02
    assert picks["clear_reject"] == 5  # highest score, 0.95
    assert picks["borderline"] == 2  # p=0.5, exactly at threshold


# ---- _display_value ----------------------------------------------------------
def test_display_value_rounds_float32_precision_artifacts():
    """The actual bug found running this on real data: a raw modelling-frame
    float32 column (e.g. dti stored as 30.17) prints as 30.170000076293945
    if passed straight through -- not something a credit adjudicator should
    ever see. _display_value must round it to something clean."""
    raw = np.float32(30.17)
    # numpy's own str() of a float32 is already clean -- the artifact only
    # shows up once the value is widened to a python float (float64), which
    # is what an f-string does to a numpy scalar pulled out of a DataFrame
    # row. Confirms the artifact this test guards against is real before
    # checking the fix.
    assert f"{float(raw)}" == "30.170000076293945"
    assert f"{_display_value(raw)}" == "30.17"


def test_display_value_leaves_whole_number_floats_and_non_floats_untouched():
    assert _display_value(0.0) == 0.0
    assert _display_value(113.0) == 113.0
    assert _display_value("credit_card") == "credit_card"
    assert _display_value(13) == 13


# ---- format_applicant_explanation --------------------------------------------
def test_format_applicant_explanation_sorts_by_absolute_contribution_and_respects_top_k():
    shap_row = np.array([0.1, -0.5, 0.3, -0.05, 0.2])
    feature_values = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    rows = format_applicant_explanation(shap_row, feature_values, base_value=0.0, top_k=3)
    assert len(rows) == 3
    assert [r["feature"] for r in rows] == ["b", "c", "e"]
    assert rows[0]["direction"] == "lowers risk"  # shap_row["b"] = -0.5, negative
    assert rows[1]["direction"] == "raises risk"  # shap_row["c"] = +0.3, positive


def test_format_applicant_explanation_direction_matches_sign():
    shap_row = np.array([-1.0, 1.0])
    feature_values = pd.Series([10, 20], index=["down", "up"])
    rows = format_applicant_explanation(shap_row, feature_values, base_value=0.0, top_k=2)
    by_feature = {r["feature"]: r for r in rows}
    assert by_feature["down"]["direction"] == "lowers risk"
    assert by_feature["up"]["direction"] == "raises risk"
