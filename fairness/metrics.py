"""
Phase 6: group-fairness metrics, and the demographic-parity / equal-opportunity
tension made numeric rather than asserted.

CONVENTION -- READ THIS BEFORE THE NUMBERS BELOW
--------------------------------------------------
Every other phase treats `default_window` (1 = defaulted) as the positive
class, because that's what the model predicts. Fairness metrics need the
opposite convention to line up with the literature this section cites: Hardt,
Price & Srebro (2016), "Equality of Opportunity in Supervised Learning",
define "positive" as the *favorable* outcome and "equal opportunity" as equal
true-positive rate on it. Flipping the label here isn't cosmetic -- it decides
which error rate (FPR vs FNR) reads as "the one that costs the applicant" vs
"the one that costs the lender", and getting that backwards would make the
write-up say the opposite of what it means.

So throughout this module:
    y_good      = 1 - default_window   (1 = did NOT default -- the favorable outcome)
    approve     = score <= threshold   (score = calibrated P(default), same
                                         direction as evaluation/decisioning.py,
                                         so a Phase 5 threshold plugs in unchanged)
    TPR ("equal opportunity rate") = P(approve | y_good=1)
                                    = fraction of creditworthy applicants who get approved
    FPR                            = P(approve | y_good=0)
                                    = fraction of applicants who will default but get approved anyway
    FNR = 1 - TPR                  = fraction of creditworthy applicants wrongly REJECTED
                                    -- the opportunity cost to good applicants, and the
                                       error equal-opportunity mitigation targets directly.

Demographic parity is defined on `approve` alone and is unaffected by this
choice -- P(approve) doesn't care which class is "positive".
"""
import numpy as np
import pandas as pd


def group_rates(y_default, score, threshold, group):
    """Per-group confusion-matrix-derived rates at a single shared threshold.

    Returns a DataFrame indexed by group value with: n, base_default_rate,
    approval_rate, tpr (equal-opportunity rate), fpr, fnr, mean_predicted
    (mean calibrated score), calibration_gap (mean_predicted - base_default_rate,
    signed so positive = model overstates risk for that group).
    """
    y_default = np.asarray(y_default, dtype=float)
    score = np.asarray(score, dtype=float)
    group = np.asarray(group)
    y_good = 1.0 - y_default
    approve = score <= threshold

    rows = []
    for g in pd.unique(group):
        m = group == g
        n = int(m.sum())
        if n == 0:
            continue
        good_m = m & (y_good == 1)
        bad_m = m & (y_good == 0)
        rows.append({
            "group": g,
            "n": n,
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


def demographic_parity_gap(rates):
    """Max-min approval rate across groups, plus the disparate-impact ratio
    (min/max approval rate) -- the 4/5ths rule from EEOC adverse-impact
    analysis (a group approved at less than 80% of the most-favored group's
    rate is the conventional flag) is a real, if employment-law-derived,
    reference point cited in lending disparate-impact discussions too."""
    ar = rates["approval_rate"]
    return {
        "gap": float(ar.max() - ar.min()),
        "disparate_impact_ratio": float(ar.min() / ar.max()) if ar.max() > 0 else float("nan"),
        "most_favored": ar.idxmax(),
        "least_favored": ar.idxmin(),
    }


def equalized_odds_gap(rates):
    """Max-min TPR (equal-opportunity gap) and max-min FPR across groups."""
    return {
        "tpr_gap": float(rates["tpr"].max() - rates["tpr"].min()),
        "fpr_gap": float(rates["fpr"].max() - rates["fpr"].min()),
        "worst_tpr_group": rates["tpr"].idxmin(),
        "worst_fpr_group": rates["fpr"].idxmax(),
    }


def threshold_for_target_approval_rate(score, target_rate):
    """The threshold tau such that mean(score <= tau) ~= target_rate, i.e. the
    (target_rate)-quantile of the score distribution within this group."""
    target_rate = min(max(target_rate, 0.0), 1.0)
    return float(np.quantile(score, target_rate))


def threshold_for_target_tpr(y_default, score, target_tpr):
    """The threshold tau such that P(score <= tau | y_good=1) ~= target_tpr,
    found within the GROUP's own good-applicant score distribution -- this is
    what an equal-opportunity mitigation actually adjusts: not the group's
    overall approval volume, but specifically how creditworthy applicants in
    that group are treated."""
    y_default = np.asarray(y_default, dtype=float)
    score = np.asarray(score, dtype=float)
    good_scores = score[y_default == 0]
    target_tpr = min(max(target_tpr, 0.0), 1.0)
    return float(np.quantile(good_scores, target_tpr))


def per_group_thresholds(y_default, score, group, mode, target):
    """Fit one threshold per group on this (fitting) cohort, targeting either
    'approval_rate' (demographic parity) or 'tpr' (equal opportunity) at the
    given population-wide `target` value. Returns {group_value: threshold}.

    Fitted on a CALIBRATION cohort and applied to a separate TEST cohort by
    the caller -- exactly the fit/calibrate/test separation every other phase
    uses, so a mitigation's measured cost isn't inflated by picking thresholds
    on the same data they're scored on."""
    y_default = np.asarray(y_default, dtype=float)
    score = np.asarray(score, dtype=float)
    group = np.asarray(group)
    out = {}
    for g in pd.unique(group):
        m = group == g
        if mode == "approval_rate":
            out[g] = threshold_for_target_approval_rate(score[m], target)
        elif mode == "tpr":
            out[g] = threshold_for_target_tpr(y_default[m], score[m], target)
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return out


def apply_group_thresholds(score, group, thresholds, default_threshold):
    """approve[i] = score[i] <= thresholds[group[i]], falling back to
    `default_threshold` for a group seen at apply-time but not at fit-time
    (can happen with a rare state in a small test fold)."""
    score = np.asarray(score, dtype=float)
    group = np.asarray(group)
    tau = np.array([thresholds.get(g, default_threshold) for g in group])
    return score <= tau
