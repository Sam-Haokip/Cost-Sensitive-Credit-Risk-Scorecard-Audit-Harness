"""
Phase 7: which features actually drive the model, by two different
definitions of "drive" -- and where those two definitions disagree.

BOTH MODELS ARE EXPLAINED IN LOG-ODDS-OF-DEFAULT SPACE
-------------------------------------------------------
SHAP values are only additive in whatever space the model's own output lives
in. LightGBM's raw score (`predict(..., raw_score=True)`) is already a
log-odds margin; the logistic-WoE scorecard's linear predictor
(`coef . WoE(x) + intercept`) is *also* a log-odds quantity by construction
-- that is the entire reason WoE encoding exists (see features/woe.py).
Explaining both models in this shared unit means a feature's contribution is
directly comparable across models, and it matches how a real scorecard
communicates risk to a credit officer (points added or subtracted from a
base log-odds, not an opaque 0-1 probability shift).

THE LOGISTIC-WOE MODEL DOESN'T NEED AN APPROXIMATE SHAP ALGORITHM
--------------------------------------------------------------------
For any function that is *additively separable* -- f(x) = sum_i(w_i * x_i) + b,
which is exactly what a fitted linear model is -- the Shapley value of
feature i is EXACTLY w_i * (x_i - E[x_i]), with no approximation and no
independence assumption required, because the function has no interaction
terms to apportion credit for in the first place. `linear_shap_values` below
computes this closed form directly rather than calling `shap.LinearExplainer`.

This was not an assumption -- it was checked against the library. The first
attempt called `shap.LinearExplainer(model, background_df)` and got values
that disagreed with the closed form by up to 0.044 in log-odds units, which
is large enough to change a feature's rank. The cause: `LinearExplainer`'s
default `Independent` masker silently summarises the background data down to
**100 rows** (`shap.maskers.Independent(data)` with no `max_samples` argument
defaults to 100) rather than using the full ~18-100k row training fold that
was passed in -- so its baseline (and therefore every contribution) is
computed against a 100-row sample's mean, not the population mean. Forcing
`max_samples=len(data)` reproduces the closed form to floating-point
precision (see `tests/test_explain.py`). This is exactly the kind of
"library does something reasonable-sounding by default that isn't what you
assumed" trap this project has hit before (Census's `state` requirement,
`LGD_RATE`/`MARGIN_RATE` read off module globals) -- caught here by
cross-checking a fast library call against a hand-derivable exact answer,
not by trusting either one blind.

LightGBM has no such closed form -- a tree ensemble's prediction is not
additively separable in the raw features, so `tree_shap_values` uses
`shap.TreeExplainer`, which computes the exact (not sampled/approximated)
Shapley values for tree ensembles via the polynomial-time TreeSHAP
algorithm, using the trees' own structure to account for feature
interactions and correlations rather than assuming independence.

PERMUTATION IMPORTANCE: A SECOND, DIFFERENT DEFINITION OF "IMPORTANT"
------------------------------------------------------------------------
SHAP's global importance (mean |shap value| per feature) asks: across
predictions the model actually made, how much did this feature move the
output? Permutation importance asks a different question: if this feature
were replaced with noise, how much would the model's TEST-SET RANKING METRIC
(PR-AUC here, matching this project's primary metric everywhere else) get
worse? These can disagree, most commonly when two features are correlated:
shuffling one still leaves the other carrying similar information, so
permuting either ALONE barely hurts PR-AUC even though SHAP (which sees the
model's actual per-prediction reliance) may still credit both. Where the two
rankings disagree sharply, `rank_table` flags it and the `__main__` block
checks the flagged feature's correlation with its nearest top-20 neighbour
as a candidate explanation -- reported honestly as "found" or "not found by
this simple check," not asserted.

`permutation_importance_manual` exists instead of
`sklearn.inspection.permutation_importance` for a concrete, checked reason:
the naive approach --
    Xp[col] = rng.permutation(Xp[col].values)
-- crashes for a `category`-dtype column, because `numpy.random.Generator
.permutation` calls `np.asarray` on its input, which silently converts a
pandas Categorical into a plain object array and strips the dtype. LightGBM
then refuses to predict on it ("train and valid dataset categorical_feature
do not match"). `sklearn.inspection.permutation_importance` hit some
version of this same problem in testing (a multi-minute hang rather than a
clean crash, not fully diagnosed) and was dropped in favour of a hand-rolled
loop that shuffles by *positional reindexing* (`X[col].iloc[perm]`) instead
of raw-array permutation -- this preserves whatever dtype the column already
has, and is directly testable (see
`test_permutation_importance_manual_preserves_categorical_dtype`).

COMPUTATIONAL SCOPE, STATED RATHER THAN HIDDEN
------------------------------------------------
Permutation importance is run on a fixed 10,000-row subsample of each test
fold, not the full ~100k. Measured on this project's real data (one fold,
96 features, 5 repeats each): ~200s for logistic_woe, ~63s for lightgbm --
the gap is `WOEEncoder.transform` itself, called fresh inside every one of
the 480 shuffled predict_fn calls (~0.31s each on 10,000 rows x 96 columns);
lightgbm's predict_proba has no such per-call encoding cost. This was not
assumed to be symmetric across models -- it was timed per model, per the
same "measure, don't assume" rule this project applies everywhere else. SHAP
values run on the full test fold: the closed-form linear computation is
negligible (<0.1s for 100,000 rows, since it's one vectorised subtraction and
multiply), but TreeExplainer is not -- ~130s per fold for 100,000 rows x 96
features on a 300-tree LightGBM model, the single largest piece of one
fold's runtime. Total measured: ~7 minutes per fold, ~28 minutes for all 4
folds plus the one-time ~50s data load. Global importance and permutation
importance are both computed on ALL 4 outer folds and reported as a
mean/spread across them, same as every other phase -- local explanations
(the 3 representative applicants) use only the most recent fold (2017),
matching the convention `reports/figures/decisioning_curve.png` and
`reliability.png` already set: illustrative examples don't need
cross-fold aggregation, headline numbers do.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from data.dataset import CATEGORICAL, FEATURE_COLS, load_modelling_frame
from evaluation.decisioning import best_threshold, unit_economics
from features.woe import WOEEncoder
from models.temporal_validation import SEED
from models.tuning import outer_folds

CATEGORICAL_FEATURES = [c for c in FEATURE_COLS if c in CATEGORICAL]
PERM_SAMPLE_SIZE = 10_000
PERM_REPEATS = 5
TOP_K_LOCAL = 8
OUT_IMPORTANCE = "reports/explainability_importance.csv"
OUT_LOCAL = "reports/explainability_local_examples.md"
OUT_SUMMARY = "reports/explainability_summary.md"

# The 6 features Phase 3's WoE-encoder fix rescued from being silently
# collapsed to a single bin (decision 7 in DECISIONS.md) -- their measured
# information value was 0.0002-0.0011, essentially nothing. If they show up
# with non-trivial SHAP importance here, that would contradict a finding
# this project already made and reported; if they land near the bottom, it's
# an independent confirmation from a completely different method.
LOW_IV_RESCUED_FEATURES = [
    "acc_now_delinq", "tax_liens", "pub_rec_bankruptcies",
    "pub_rec", "num_tl_90g_dpd_24m", "tot_coll_amt",
]

# The only geography feature actually fed to either model (data/dataset.py
# defers raw zip_code entirely -- see DEFERRED there). Phase 6 found ZIP3 is
# a usable, if imperfect, proxy for race/ethnicity; addr_state is a coarser
# version of the same signal. High importance here would be worth flagging
# alongside that finding, not just noting as an interpretability curiosity.
GEOGRAPHY_FEATURE = "addr_state"


# ---- model fitting (mirrors evaluation/calibration.py's hyperparameters
# exactly, but returns the fitted objects themselves rather than just
# predictions, since SHAP needs to inspect the model, not just call it) ----
def fit_logistic_woe_model(fit: pd.DataFrame):
    enc = WOEEncoder()
    X = enc.fit_transform(fit[FEATURE_COLS], fit["default_window"])
    model = LogisticRegression(max_iter=2000, random_state=SEED)
    model.fit(X, fit["default_window"])
    return enc, model


def fit_lightgbm_model(fit: pd.DataFrame):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               random_state=SEED, n_jobs=2, verbose=-1)
    model.fit(fit[FEATURE_COLS], fit["default_window"],
              categorical_feature=CATEGORICAL_FEATURES)
    return model


# ---- SHAP -------------------------------------------------------------------
def linear_shap_values(model: LogisticRegression, X_background: pd.DataFrame,
                        X_explain: pd.DataFrame):
    """Exact Shapley values for a linear model: coef_i * (x_i - E[x_i]),
    with E[x_i] estimated from X_background (the same cohort the model was
    fitted on). No approximation -- see module docstring for why none is
    needed, and for why this is NOT simply `shap.LinearExplainer` with
    default arguments."""
    coef = model.coef_[0]
    mean_x = X_background.mean(axis=0).values
    shap_values = (X_explain.values - mean_x) * coef
    base_value = float((coef * mean_x).sum() + model.intercept_[0])
    return shap_values, base_value


def tree_shap_values(model, X_explain: pd.DataFrame):
    """Exact TreeSHAP values via the shap library. Asserts the output shape
    rather than assuming it -- shap's own LightGBM integration has changed
    its return format across versions (a single 2D array vs. a per-class
    list), so a silent shape change would otherwise misattribute every
    contribution without erroring."""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)
    shap_values = np.asarray(shap_values)
    if shap_values.shape != (len(X_explain), X_explain.shape[1]):
        raise RuntimeError(
            f"tree_shap_values: expected shape {(len(X_explain), X_explain.shape[1])}, "
            f"got {shap_values.shape} -- shap's TreeExplainer output format may have "
            f"changed (it has before, across versions, for binary LightGBM classifiers); "
            f"do not proceed without checking which axis is which."
        )
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.asarray(base_value).reshape(-1)[-1])
    return shap_values, float(base_value)


def global_importance(shap_values: np.ndarray, feature_names) -> pd.Series:
    """Mean |SHAP value| per feature, sorted descending -- the standard
    global-importance summary of a set of local (per-row) explanations."""
    return pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)


# ---- permutation importance --------------------------------------------------
def _shuffle_column(X: pd.DataFrame, col: str, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle one column's values across rows, preserving its dtype exactly
    -- see module docstring for why this can't be `rng.permutation(X[col]
    .values)` when col is categorical."""
    perm = rng.permutation(len(X))
    out = X.copy()
    out[col] = X[col].iloc[perm].reset_index(drop=True).values \
        if not isinstance(X[col].dtype, pd.CategoricalDtype) \
        else pd.Categorical(X[col].iloc[perm].reset_index(drop=True), categories=X[col].cat.categories)
    return out


def permutation_importance_manual(predict_fn, X: pd.DataFrame, y, feature_names,
                                   seed: int = SEED, n_repeats: int = PERM_REPEATS) -> pd.Series:
    """Mean drop in PR-AUC (average_precision_score) when each feature is
    independently shuffled, averaged over `n_repeats` shuffles. `predict_fn`
    takes a DataFrame shaped like X and returns P(default) for each row."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    base_score = average_precision_score(y, predict_fn(X))
    drops = {}
    for col in feature_names:
        col_drops = []
        for _ in range(n_repeats):
            Xp = _shuffle_column(X, col, rng)
            col_drops.append(base_score - average_precision_score(y, predict_fn(Xp)))
        drops[col] = float(np.mean(col_drops))
    return pd.Series(drops).sort_values(ascending=False)


# ---- comparing the two rankings ----------------------------------------------
def rank_table(shap_imp: pd.Series, perm_imp: pd.Series) -> pd.DataFrame:
    """One row per feature: both importance values, both ranks (1 = most
    important), and the gap between the ranks -- a large gap is a candidate
    for the correlated-feature or interaction-effect investigation the brief
    asks for, not a data quality problem to hide."""
    shap_rank = shap_imp.rank(ascending=False, method="min")
    perm_rank = perm_imp.rank(ascending=False, method="min")
    out = pd.DataFrame({
        "shap_importance": shap_imp, "perm_importance": perm_imp,
        "shap_rank": shap_rank, "perm_rank": perm_rank,
    })
    out["rank_gap"] = (out["shap_rank"] - out["perm_rank"]).abs()
    return out.sort_values("rank_gap", ascending=False)


CORRELATION_FLAG_THRESHOLD = 0.3


def nearest_correlated_neighbor(feature: str, top_features, X_numeric: pd.DataFrame):
    """Among `top_features` (excluding `feature` itself), the one with the
    highest |Pearson correlation| to `feature`, computed on `X_numeric` --
    the WoE-encoded frame is used for this regardless of which model is being
    checked, since WoE encoding gives every feature (categorical or not) a
    single comparable numeric column, and correlation is otherwise undefined
    for a raw categorical.

    This is the concrete check the module docstring promises for a feature
    where SHAP and permutation importance disagree sharply: a correlated
    partner already ranked highly is a candidate explanation (permuting
    `feature` alone may barely hurt PR-AUC if the partner covers for it, even
    though SHAP still credits both individually) -- reported as "found" or
    "not found by this simple check," never asserted as the cause, since
    correlation alone cannot distinguish that from an interaction effect or
    from permutation importance's own sampling noise.
    """
    candidates = [f for f in top_features if f != feature]
    if not candidates:
        return None, 0.0
    corrs = X_numeric[candidates].corrwith(X_numeric[feature])
    corrs = corrs.dropna()
    if corrs.empty:
        return None, 0.0
    best = corrs.abs().idxmax()
    return best, float(corrs[best])


# ---- local explanations -------------------------------------------------------
def pick_representative_applicants(p_test: np.ndarray, threshold: float) -> dict:
    """Three positional indices into the test set, chosen using only
    decision-time information (the calibrated score vs. Phase 5's
    cost-optimal shared threshold) -- never the hindsight outcome label,
    since an adjudicator would not have it either.

    clear_approve : lowest-scoring applicant (most confidently good)
    clear_reject  : highest-scoring applicant (most confidently bad)
    borderline    : whichever applicant's score sits closest to the
                    threshold in either direction
    """
    p_test = np.asarray(p_test)
    return {
        "clear_approve": int(np.argmin(p_test)),
        "clear_reject": int(np.argmax(p_test)),
        "borderline": int(np.argmin(np.abs(p_test - threshold))),
    }


def _display_value(v):
    """feature_values comes straight from the raw modelling frame, where
    numeric columns are float32 -- printing one directly leaks float32's
    binary-rounding artifacts (e.g. dti stored as 30.17 prints as
    30.170000076293945). Round floats for display; leave ints, strings and
    categoricals untouched, since rounding those would be meaningless or
    would silently coerce a category to a number."""
    if isinstance(v, (float, np.floating)) and not float(v).is_integer():
        return round(float(v), 2)
    return v


def format_applicant_explanation(shap_row: np.ndarray, feature_values: pd.Series,
                                  base_value: float, top_k: int = TOP_K_LOCAL) -> list:
    """Top-k features by |contribution|, as a credit adjudicator would want
    to see them: which features pushed risk up, which pushed it down, and by
    how much (log-odds units), not a raw SHAP array dump."""
    order = np.argsort(-np.abs(shap_row))[:top_k]
    rows = []
    for i in order:
        rows.append({
            "feature": feature_values.index[i],
            "value": _display_value(feature_values.iloc[i]),
            "contribution_log_odds": float(shap_row[i]),
            "direction": "raises risk" if shap_row[i] > 0 else "lowers risk",
        })
    return rows


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    print("loading modelling frame...")
    df = load_modelling_frame()
    print(f"{len(df):,} loans, {len(FEATURE_COLS)} features\n")

    econ = unit_economics()
    LGD_RATE, MARGIN_RATE = econ["lgd_rate"], econ["margin_rate"]

    importance_rows = []
    local_sections = []
    disagreement_checks = []
    fold_list = list(outer_folds(df))

    for year, outer_train, test, fit, calib, cal_year in fold_list:
        print(f"fold {year}: model fit {len(fit):,} (to {cal_year - 1}) | test {len(test):,} ({year})")
        rng = np.random.default_rng(SEED)
        perm_idx = rng.choice(len(test), size=min(PERM_SAMPLE_SIZE, len(test)), replace=False)
        test_perm = test.iloc[perm_idx]

        # ---- logistic-WoE ----
        enc, lr = fit_logistic_woe_model(fit)
        X_fit_woe = enc.transform(fit[FEATURE_COLS])
        X_test_woe = enc.transform(test[FEATURE_COLS])
        shap_lr, base_lr = linear_shap_values(lr, X_fit_woe, X_test_woe)
        imp_shap_lr = global_importance(shap_lr, FEATURE_COLS)

        def predict_lr(Xraw, _enc=enc, _lr=lr):
            return _lr.predict_proba(_enc.transform(Xraw[FEATURE_COLS]))[:, 1]

        imp_perm_lr = permutation_importance_manual(
            predict_lr, test_perm, test_perm["default_window"].values, FEATURE_COLS)

        # ---- LightGBM ----
        lgbm = fit_lightgbm_model(fit)
        shap_gbm, base_gbm = tree_shap_values(lgbm, test[FEATURE_COLS])
        imp_shap_gbm = global_importance(shap_gbm, FEATURE_COLS)

        def predict_gbm(Xraw, _m=lgbm):
            return _m.predict_proba(Xraw[FEATURE_COLS])[:, 1]

        imp_perm_gbm = permutation_importance_manual(
            predict_gbm, test_perm, test_perm["default_window"].values, FEATURE_COLS)

        for mname, shap_imp, perm_imp in [
            ("logistic_woe", imp_shap_lr, imp_perm_lr),
            ("lightgbm", imp_shap_gbm, imp_perm_gbm),
        ]:
            rt = rank_table(shap_imp, perm_imp).reset_index().rename(columns={"index": "feature"})
            rt["fold"], rt["model"] = year, mname
            importance_rows.append(rt)
            top_gap = rt.iloc[0]
            print(f"   {mname:13s} top-5 by SHAP: {list(shap_imp.index[:5])}")
            print(f"   {mname:13s} biggest SHAP-vs-permutation rank gap: "
                  f"{top_gap['feature']} (shap rank {int(top_gap['shap_rank'])}, "
                  f"perm rank {int(top_gap['perm_rank'])})")

            # Candidate explanation: is the disagreeing feature correlated
            # (in WoE-encoded space, shared across both models -- see
            # nearest_correlated_neighbor's docstring) with something already
            # ranked in the top 20 by SHAP?
            neighbor, corr = nearest_correlated_neighbor(
                top_gap["feature"], list(shap_imp.index[:20]), X_test_woe)
            flagged = neighbor is not None and abs(corr) >= CORRELATION_FLAG_THRESHOLD
            disagreement_checks.append({
                "fold": year, "model": mname, "feature": top_gap["feature"],
                "rank_gap": int(top_gap["rank_gap"]), "neighbor": neighbor,
                "correlation": corr, "flagged": flagged,
            })
            if neighbor is not None:
                verdict = "found" if flagged else "not found by this simple check"
                print(f"   {mname:13s} correlation check for {top_gap['feature']}: "
                      f"nearest top-20 neighbor {neighbor} (r={corr:+.2f}) -- {verdict}")

        # ---- local explanations, most recent fold only ----
        if year == fold_list[-1][0]:
            y_calib, amt_calib = calib["default_window"].values, calib["loan_amnt"].values
            p_calib_lr = lr.predict_proba(enc.transform(calib[FEATURE_COLS]))[:, 1]
            thr, _ = best_threshold(y_calib, p_calib_lr, amt_calib, LGD_RATE, MARGIN_RATE)
            picks = pick_representative_applicants(predict_lr(test), thr)
            print(f"\n   local explanations on fold {year} (threshold={thr:.3f}):")
            for label, i in picks.items():
                expl = format_applicant_explanation(shap_lr[i], test[FEATURE_COLS].iloc[i], base_lr)
                score = predict_lr(test)[i]
                local_sections.append((label, year, thr, score, expl))
                print(f"   {label}: score={score:.3f}  top feature: "
                      f"{expl[0]['feature']} ({expl[0]['direction']}, {expl[0]['contribution_log_odds']:+.3f})")
        print()

    importance = pd.concat(importance_rows, ignore_index=True)
    importance.to_csv(OUT_IMPORTANCE, index=False)

    print("=== across folds: mean SHAP importance rank, top 10 per model ===")
    agg = importance.groupby(["model", "feature"]).agg(
        mean_shap_importance=("shap_importance", "mean"),
        mean_perm_importance=("perm_importance", "mean"),
        mean_rank_gap=("rank_gap", "mean"),
    ).reset_index()
    for mname in ["logistic_woe", "lightgbm"]:
        top = agg[agg.model == mname].sort_values("mean_shap_importance", ascending=False).head(10)
        print(f"\n{mname}:")
        print(top[["feature", "mean_shap_importance", "mean_perm_importance", "mean_rank_gap"]]
              .to_string(index=False))

    print("\n=== sanity check: the 6 features Phase 3's WoE-encoder fix rescued ===")
    lr_agg = agg[agg.model == "logistic_woe"].set_index("feature")
    low_iv_results = []
    for feat in LOW_IV_RESCUED_FEATURES:
        if feat in lr_agg.index:
            rank = int((lr_agg["mean_shap_importance"] >= lr_agg.loc[feat, "mean_shap_importance"]).sum())
            mean_shap = float(lr_agg.loc[feat, "mean_shap_importance"])
            low_iv_results.append({"feature": feat, "mean_abs_shap": mean_shap, "rank": rank, "n_features": len(lr_agg)})
            print(f"   {feat:20s} mean |SHAP|={mean_shap:.5f}  rank {rank}/{len(lr_agg)}")

    print(f"\n=== sanity check: geography feature ({GEOGRAPHY_FEATURE}) ===")
    geo_results = []
    for mname in ["logistic_woe", "lightgbm"]:
        sub = agg[agg.model == mname].sort_values("mean_shap_importance", ascending=False).reset_index(drop=True)
        if GEOGRAPHY_FEATURE in sub["feature"].values:
            rank = int(sub.index[sub["feature"] == GEOGRAPHY_FEATURE][0]) + 1
            geo_results.append({"model": mname, "rank": rank, "n_features": len(sub)})
            print(f"   {mname:13s} {GEOGRAPHY_FEATURE} rank {rank}/{len(sub)} by mean SHAP importance")

    with open(OUT_LOCAL, "w") as f:
        f.write("# Phase 7: local explanations for 3 representative applicants\n\n")
        f.write("logistic-WoE scorecard, most recent fold, decision-time information only "
                "(never the hindsight outcome label).\n\n")
        for label, year, thr, score, expl in local_sections:
            f.write(f"## {label} (fold {year}, threshold {thr:.3f}, score {score:.3f})\n\n")
            f.write("| feature | value | contribution (log-odds) | direction |\n")
            f.write("|---|---|---:|---|\n")
            for row in expl:
                f.write(f"| {row['feature']} | {row['value']} | {row['contribution_log_odds']:+.3f} "
                        f"| {row['direction']} |\n")
            f.write("\n")

    flagged_checks = [c for c in disagreement_checks if c["flagged"]]
    with open(OUT_SUMMARY, "w") as f:
        f.write("# Phase 7 explainability: SHAP vs. permutation importance, "
                 "logistic-WoE vs. LightGBM\n\n")
        f.write("Global importance and permutation importance are both computed on all 4 outer "
                 "folds (mean across folds reported below); local explanations use the most "
                 "recent fold (2017) only -- see explainability/explain.py's module docstring "
                 "for why.\n\n")

        for mname in ["logistic_woe", "lightgbm"]:
            top = agg[agg.model == mname].sort_values("mean_shap_importance", ascending=False).head(10)
            f.write(f"## {mname}: top 10 features by mean |SHAP| across folds\n\n")
            f.write("| feature | mean \\|SHAP\\| (log-odds) | mean permutation drop (PR-AUC) | mean SHAP-vs-permutation rank gap |\n")
            f.write("|---|---:|---:|---:|\n")
            for _, row in top.iterrows():
                f.write(f"| {row['feature']} | {row['mean_shap_importance']:.4f} | "
                        f"{row['mean_perm_importance']:.5f} | {row['mean_rank_gap']:.1f} |\n")
            f.write("\n")

        f.write("## Where SHAP and permutation importance disagree most, per fold\n\n")
        f.write("The feature with the largest |SHAP rank - permutation rank| gap in each "
                "fold/model, and whether a correlated feature already in the SHAP top 20 "
                "(WoE-encoded Pearson correlation, |r| >= "
                f"{CORRELATION_FLAG_THRESHOLD}) is a plausible reason why shuffling it alone "
                "didn't move PR-AUC as much as SHAP's credit to it would suggest:\n\n")
        f.write("| fold | model | feature | rank gap | nearest top-20 neighbor | correlation | correlated-feature explanation |\n")
        f.write("|---|---|---|---:|---|---:|---|\n")
        for c in disagreement_checks:
            neighbor = c["neighbor"] if c["neighbor"] is not None else "n/a"
            corr_str = f"{c['correlation']:+.2f}" if c["neighbor"] is not None else "n/a"
            verdict = "found" if c["flagged"] else "not found by this simple check"
            f.write(f"| {c['fold']} | {c['model']} | {c['feature']} | {c['rank_gap']} | "
                    f"{neighbor} | {corr_str} | {verdict} |\n")
        f.write(f"\n{len(flagged_checks)}/{len(disagreement_checks)} fold/model disagreements "
                "had a correlated top-20 feature as a candidate explanation; the rest are "
                "reported as found, not asserted -- correlation alone can't rule out an "
                "interaction effect or permutation-importance sampling noise instead.\n\n")

        f.write("## Sanity check: the 6 features Phase 3's WoE-encoder fix rescued\n\n")
        f.write("Information value 0.0002-0.0011 at fit time (decision 7, DECISIONS.md) -- "
                "essentially nothing. If SHAP, an unrelated method, also ranks them near the "
                "bottom of logistic_woe's " + str(len(lr_agg)) + " features, that's an "
                "independent confirmation, not a re-derivation of the same number.\n\n")
        f.write("| feature | mean \\|SHAP\\| (log-odds) | rank |\n")
        f.write("|---|---:|---:|\n")
        for r in low_iv_results:
            f.write(f"| {r['feature']} | {r['mean_abs_shap']:.5f} | {r['rank']}/{r['n_features']} |\n")
        f.write("\n")

        f.write(f"## Sanity check: geography feature ({GEOGRAPHY_FEATURE})\n\n")
        f.write("Phase 6 found ZIP3-derived geography is a usable, if imperfect, proxy for "
                f"race/ethnicity; {GEOGRAPHY_FEATURE} is data/dataset.py's only geography "
                "feature actually fed to either model (raw zip_code is deferred entirely -- "
                "see DEFERRED there). Its rank here is reported for that reason, not because "
                "high SHAP importance would itself be a fairness problem on its own.\n\n")
        f.write("| model | rank |\n")
        f.write("|---|---:|\n")
        for r in geo_results:
            f.write(f"| {r['model']} | {r['rank']}/{r['n_features']} |\n")
        f.write("\n")

        f.write(f"See {OUT_LOCAL} for the 3 representative-applicant local explanations, and "
                f"{OUT_IMPORTANCE} for the full per-fold, per-feature importance table.\n")

    print(f"\nwrote {OUT_IMPORTANCE}, {OUT_LOCAL}, {OUT_SUMMARY}")
