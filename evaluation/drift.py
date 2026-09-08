"""
Phase 2: covariate drift across origination periods (PSI and KS).

A temporal validation scheme is only meaningful if you know WHICH features moved
between the periods you are training and testing on. This measures that.

Population Stability Index compares a feature's distribution in a later period
against a baseline: bin the baseline, then ask how much probability mass moved.

    PSI = sum over bins of (p_compare - p_base) * ln(p_compare / p_base)

Conventional reading, from credit scorecard practice:
    < 0.10  stable
    0.10 - 0.25  moderate shift, monitor
    > 0.25  significant shift, investigate before trusting the model across it

MISSINGNESS IS ITS OWN BIN, deliberately. Phase 1 found fourteen bureau columns
that are 100% null before 2016 and ~0% after, because Lending Club introduced
the fields partway through. If nulls are dropped before binning -- which is the
default in most PSI implementations -- that shift is invisible, and it is the
single largest distributional change in the dataset. Treating null as a bin is
what makes it show up.

Writes reports/drift.md.
"""
import os

import numpy as np
import pandas as pd

from data.dataset import CATEGORICAL, FEATURE_COLS, load_modelling_frame

BASELINE_YEARS = [2013, 2014]
COMPARE_YEARS = [2015, 2016, 2017]
N_BINS = 10
EPS = 1e-4  # floors zero-count bins so PSI stays finite
OUT_PATH = "reports/drift.md"


def _proportions(base: pd.Series, comp: pd.Series, is_categorical: bool):
    """Return aligned bin proportions for baseline and comparison, with null
    carried as an explicit bin."""
    base_null = base.isna().mean()
    comp_null = comp.isna().mean()

    b, c = base.dropna(), comp.dropna()

    # If one side has no observed values at all, the presence/absence of the
    # field IS the drift, and there is nothing to bin by value. This is not an
    # edge case to skip: it is exactly the fourteen bureau columns Lending Club
    # introduced mid-history, which are 100% null across the 2013-2014 baseline
    # and populated afterwards. An earlier version of this function returned
    # None here, which silently dropped the largest distributional shifts in
    # the dataset from the report.
    if len(b) == 0 or len(c) == 0:
        bp = np.array([1.0 - base_null, base_null])
        cp = np.array([1.0 - comp_null, comp_null])
        return np.clip(bp, EPS, None), np.clip(cp, EPS, None)

    # Quantile binning collapses on heavily zero-inflated count columns
    # (acc_now_delinq, tax_liens, num_tl_30dpd and similar are ~99% zeros, so
    # every decile edge lands on 0). Those are effectively discrete, so fall
    # back to binning on their observed values rather than returning nothing --
    # an earlier version dropped eight such features from the report entirely.
    discrete_fallback = False
    if not is_categorical:
        edges = np.unique(np.nanquantile(b, np.linspace(0, 1, N_BINS + 1)))
        discrete_fallback = len(edges) < 3

    if is_categorical or discrete_fallback:
        levels = b.astype(str).value_counts(normalize=True)
        levels = levels[levels > 0.001]  # ignore vanishingly rare categories
        if levels.empty:
            return None
        keys = list(levels.index)
        bp = [(b.astype(str) == k).mean() for k in keys]
        cp = [(c.astype(str) == k).mean() for k in keys]
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        bp = np.histogram(b, bins=edges)[0] / len(b)
        cp = np.histogram(c, bins=edges)[0] / len(c)
        bp, cp = list(bp), list(cp)

    # scale the non-null mass, then append the null bin
    bp = [p * (1 - base_null) for p in bp] + [base_null]
    cp = [p * (1 - comp_null) for p in cp] + [comp_null]
    return np.clip(np.array(bp), EPS, None), np.clip(np.array(cp), EPS, None)


def psi(base: pd.Series, comp: pd.Series, is_categorical: bool):
    parts = _proportions(base, comp, is_categorical)
    if parts is None:
        return np.nan
    bp, cp = parts
    return float(np.sum((cp - bp) * np.log(cp / bp)))


def ks(base: pd.Series, comp: pd.Series):
    """Two-sample KS on non-null values only -- deliberately complementary to
    PSI, which folds nulls in. A feature with high PSI but low KS has moved
    mainly in its missingness, not in the values it does report."""
    b, c = base.dropna().astype(float), comp.dropna().astype(float)
    if len(b) < 100 or len(c) < 100:
        return np.nan
    b = np.sort(b.sample(min(len(b), 50_000), random_state=0).values)
    c = np.sort(c.sample(min(len(c), 50_000), random_state=0).values)
    grid = np.union1d(b, c)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    cdf_c = np.searchsorted(c, grid, side="right") / len(c)
    return float(np.max(np.abs(cdf_b - cdf_c)))


def drift_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["issue_year"].isin(BASELINE_YEARS)]
    rows = []
    for col in FEATURE_COLS:
        is_cat = col in CATEGORICAL
        row = {"feature": col, "type": "categorical" if is_cat else "numeric"}
        for year in COMPARE_YEARS:
            comp = df[df["issue_year"] == year]
            row[f"psi_{year}"] = psi(base[col], comp[col], is_cat)
            if not is_cat:
                row[f"ks_{year}"] = ks(base[col], comp[col])
        rows.append(row)
    t = pd.DataFrame(rows)
    psi_cols = [f"psi_{y}" for y in COMPARE_YEARS]
    t["psi_max"] = t[psi_cols].max(axis=1)
    return t.sort_values("psi_max", ascending=False)


def band(v):
    if pd.isna(v):
        return "n/a"
    return "SIGNIFICANT" if v > 0.25 else ("moderate" if v > 0.10 else "stable")


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    print("loading modelling frame...")
    df = load_modelling_frame()
    print(f"{len(df):,} loans; baseline {BASELINE_YEARS} vs {COMPARE_YEARS}\n")

    t = drift_table(df)
    t["band"] = t["psi_max"].map(band)

    n_sig = (t["psi_max"] > 0.25).sum()
    n_mod = ((t["psi_max"] > 0.10) & (t["psi_max"] <= 0.25)).sum()
    print(f"significant drift (PSI > 0.25): {n_sig} features")
    print(f"moderate drift (0.10-0.25)    : {n_mod} features")
    print(f"stable                        : {(t['psi_max'] <= 0.10).sum()} features\n")
    print("Top 15 by PSI:")
    show = ["feature", "type", "psi_2015", "psi_2016", "psi_2017", "psi_max", "band"]
    print(t[show].head(15).round(3).to_string(index=False))

    with open(OUT_PATH, "w") as f:
        f.write("# Covariate drift across origination periods\n\n")
        f.write(f"Generated by `python -m evaluation.drift`. Baseline "
                f"{'-'.join(map(str, BASELINE_YEARS))} vs {', '.join(map(str, COMPARE_YEARS))}. "
                f"{len(df):,} eligible loans.\n\n")
        f.write("PSI bands: <0.10 stable, 0.10-0.25 moderate, >0.25 significant. "
                "Nulls are carried as an explicit bin, so a field that appears or "
                "disappears registers as drift rather than vanishing from the "
                "calculation. KS is computed on non-null values only, so a feature "
                "with high PSI but low KS has moved mainly in its missingness "
                "rather than in the values it does report.\n\n")
        f.write(f"**{n_sig} features show significant drift, {n_mod} moderate.**\n\n")
        f.write("## All features by PSI\n\n")
        f.write(t.round(4).to_markdown(index=False) + "\n")
    print(f"\nwrote {OUT_PATH}")
