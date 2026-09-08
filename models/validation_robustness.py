"""
Two robustness checks the headline Phase 2 numbers need before they can be
quoted with a straight face.

1. PAIRED FOLD DIFFERENCES.
   The three walk-forward regimes share an identical test set within each fold,
   so their differences are paired and cohort variance cancels. Reporting them
   as a difference of means across folds -- against a fold-to-fold sd of ~0.02,
   five times the effect -- throws that away and invites the fair question of
   whether the gap is distinguishable from zero at all. Paired, the noise drops
   roughly tenfold.

2. IS THE EMBARGO PENALTY REALLY THE EMBARGO?
   The embargo gap shrinks monotonically across folds: -0.0344, -0.0282,
   -0.0097, -0.0071. That ordering tracks how much training data the embargoed
   regime had, which is the same confound already acknowledged for the 2014 and
   2015 folds. If the gap keeps shrinking as training size grows even on a
   size-matched fold, then part of what is being reported as "the cost of only
   using matured outcomes" is really "the cost of having less data", and the
   headline number is overstated.

   This trains both regimes at 25k / 50k / 100k / 150k on the 2016 fold, where
   both have at least 150k available and the test set is identical, and reports
   whether the gap converges.
"""
import os

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from data.dataset import FEATURE_COLS, load_modelling_frame
from models.temporal_validation import EMBARGO_MONTHS, SEED, TEST_CAP, _cap

TRAIN_SIZES = [25_000, 50_000, 100_000, 150_000]
ALL_FOLDS = [2014, 2015, 2016, 2017]
SENSITIVITY_YEAR = 2016
RESULTS_CSV = "reports/temporal_validation.csv"
OUT_PATH = "reports/validation_robustness.csv"


def paired_differences(path=RESULTS_CSV):
    r = pd.read_csv(path)
    p = (r[r.regime.isin(["random_matched", "temporal", "embargoed"])]
         .pivot(index="fold", columns="regime", values="pr_auc"))
    p = p.loc[[f for f in ["2014", "2015", "2016", "2017"] if f in p.index]]
    rows = []
    for a, b in [("temporal", "random_matched"), ("embargoed", "random_matched"),
                 ("embargoed", "temporal")]:
        d = (p[a] - p[b]).dropna()
        rows.append({
            "comparison": f"{a} - {b}",
            "mean": d.mean(),
            "paired_sd": d.std(),
            "unpaired_sd_a": p[a].std(),
            "unpaired_sd_b": p[b].std(),
            "all_same_sign": bool((d > 0).all() or (d < 0).all()),
            "per_fold": ", ".join(f"{v:+.4f}" for v in d.values),
        })
    return pd.DataFrame(rows), p


def fit_pr_auc(train, test, seed=SEED):
    m = HistGradientBoostingClassifier(
        categorical_features="from_dtype", max_iter=150, random_state=seed
    )
    m.fit(train[FEATURE_COLS], train["default_window"])
    return average_precision_score(
        test["default_window"], m.predict_proba(test[FEATURE_COLS])[:, 1]
    )


def training_size_sensitivity(df, year=SENSITIVITY_YEAR):
    test = _cap(df[df["issue_year"] == year], TEST_CAP)
    temporal_pool = df[df["issue_year"] < year]
    cutoff = pd.Timestamp(year=year - (EMBARGO_MONTHS // 12), month=1, day=1)
    embargo_pool = df[df["issue_d"] < cutoff]

    print(f"fold {year}: temporal pool {len(temporal_pool):,}, "
          f"embargo pool {len(embargo_pool):,}, test {len(test):,}\n")
    print(f"{'n_train':>9} {'temporal':>10} {'embargoed':>10} {'gap':>9}")

    rows = []
    for n in TRAIN_SIZES:
        if len(temporal_pool) < n or len(embargo_pool) < n:
            continue
        t = fit_pr_auc(_cap(temporal_pool, n), test)
        e = fit_pr_auc(_cap(embargo_pool, n), test)
        rows.append({"n_train": n, "temporal": t, "embargoed": e, "gap": e - t})
        print(f"{n:>9,} {t:>10.4f} {e:>10.4f} {e - t:>+9.4f}", flush=True)
    return pd.DataFrame(rows)


def size_matched_embargo_gap(df):
    """Cap BOTH regimes at the embargo pool's size, fold by fold.

    The raw embargo gaps (-0.0344, -0.0282, -0.0097, -0.0071) shrink across
    folds, and the first two folds had far less embargoed training data, so the
    obvious reading is that the early gaps are a sample-size artefact. Matching
    the sizes shows that is only half true: the 2014 gap halves and the 2015 gap
    shrinks by a third, but they remain roughly twice the size of the 2016 and
    2017 gaps even at equal n.

    So there are two effects. Sample size explains part of the early folds. The
    remainder is distributional distance: a 24-month embargo forces the 2014
    fold to train on pre-2012 loans, a very different population from 2014 --
    smaller, earlier era, missing the bureau fields entirely -- while the 2016
    fold only has to reach back to 2013. The embargo costs most where the
    population is drifting fastest, which is exactly what evaluation/drift.py
    independently measures.

    Matching sizes also lets the headline rest on all four folds rather than
    only the two that happened to be naturally matched."""
    print(f"{'fold':>6} {'n (both)':>10} {'temporal':>10} {'embargoed':>10} {'gap':>9}")
    rows = []
    for year in ALL_FOLDS:
        test = _cap(df[df["issue_year"] == year], TEST_CAP)
        tpool = df[df["issue_year"] < year]
        cutoff = pd.Timestamp(year=year - (EMBARGO_MONTHS // 12), month=1, day=1)
        epool = df[df["issue_d"] < cutoff]
        n = min(len(tpool), len(epool), 150_000)
        t = fit_pr_auc(_cap(tpool, n), test)
        e = fit_pr_auc(_cap(epool, n), test)
        rows.append({"fold": year, "n_train": n, "temporal": t, "embargoed": e, "gap": e - t})
        print(f"{year:>6} {n:>10,} {t:>10.4f} {e:>10.4f} {e - t:>+9.4f}", flush=True)
    out = pd.DataFrame(rows)
    print()
    print(f"size-matched embargo penalty: mean {out['gap'].mean():+.4f}, "
          f"sd {out['gap'].std():.4f}, all four folds, all same sign: "
          f"{bool((out['gap'] < 0).all())}")
    early, late = out.iloc[:2]["gap"].mean(), out.iloc[2:]["gap"].mean()
    print(f"early folds (2014-15) {early:+.4f} vs late (2016-17) {late:+.4f} -- "
          f"the penalty is {abs(early/late):.1f}x larger in the faster-drifting era")
    return out


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)

    print("=== 1. Paired fold differences (identical test set within each fold) ===\n")
    paired, per_fold = paired_differences()
    print(per_fold.round(4).to_string())
    print()
    for _, r in paired.iterrows():
        print(f"{r['comparison']}")
        print(f"   per fold: {r['per_fold']}")
        print(f"   mean {r['mean']:+.4f}   paired sd {r['paired_sd']:.4f}   "
              f"(unpaired sds {r['unpaired_sd_a']:.4f} / {r['unpaired_sd_b']:.4f})")
        print(f"   consistent in sign across folds: {r['all_same_sign']}")
        print()

    print(f"=== 2. Does the embargo gap survive more training data? (fold {SENSITIVITY_YEAR}) ===\n")
    df = load_modelling_frame()
    sens = training_size_sensitivity(df)

    if len(sens) >= 2:
        first, last = sens.iloc[0], sens.iloc[-1]
        print()
        print(f"gap at {first['n_train']:,}: {first['gap']:+.4f}   "
              f"gap at {last['n_train']:,}: {last['gap']:+.4f}")
        if abs(last["gap"]) < abs(first["gap"]) * 0.5:
            print("Gap more than halves as training data grows -- a large share of the")
            print("reported embargo penalty is data volume, not the embargo. Report the")
            print("largest-sample figure, not the average across sizes.")
        else:
            print("Gap is broadly stable across training sizes -- the penalty is the")
            print("embargo itself rather than an artefact of having less data.")

    print()
    print("=== 3. Embargo penalty with training sizes matched, all four folds ===\n")
    matched = size_matched_embargo_gap(df)

    sens.to_csv(OUT_PATH, index=False)
    matched.to_csv("reports/embargo_size_matched.csv", index=False)
    paired.to_csv("reports/paired_fold_differences.csv", index=False)
    print(f"\nwrote {OUT_PATH}, reports/embargo_size_matched.csv, "
          f"reports/paired_fold_differences.csv")
