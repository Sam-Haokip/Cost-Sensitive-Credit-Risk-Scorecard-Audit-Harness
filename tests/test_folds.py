"""Tests for the walk-forward fold construction.

Phase 2's entire argument rests on one property: a model is never fitted on a
loan whose outcome could not have been known when the model would have been
built. That property is enforced by arithmetic on dates in two places, and
nothing was checking it. These tests check it.
"""
import numpy as np
import pandas as pd

from models.baselines import embargoed_folds
from models.temporal_validation import EMBARGO_MONTHS
from models.tuning import outer_folds


def synthetic_book(per_year=30_000, seed=0):
    """A frame with the columns the fold builders need, spanning 2007-2017."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(2007, 2018):
        n = per_year
        months = rng.integers(1, 13, n)
        rows.append(pd.DataFrame({
            "issue_d": [pd.Timestamp(year=year, month=int(m), day=1) for m in months],
            "issue_year": year,
            "default_window": (rng.random(n) < 0.1).astype(int),
        }))
    return pd.concat(rows, ignore_index=True)


def test_embargo_is_twenty_four_months():
    assert EMBARGO_MONTHS == 24, (
        "the embargo must equal the 18-month performance window plus the "
        "6-month charge-off lag; changing it invalidates every Phase 2 number"
    )


def test_no_training_loan_is_issued_after_the_embargo_cutoff():
    """The property the whole design exists to guarantee."""
    df = synthetic_book()
    folds = list(embargoed_folds(df))
    assert folds, "no folds were produced"

    for year, train, test in folds:
        cutoff = pd.Timestamp(year=year - EMBARGO_MONTHS // 12, month=1, day=1)
        assert train["issue_d"].max() < cutoff, (
            f"fold {year}: training on a loan issued {train['issue_d'].max()}, "
            f"whose 18-month outcome was not known at the {cutoff.date()} cutoff"
        )


def test_train_and_test_never_share_a_cohort():
    df = synthetic_book()
    for year, train, test in embargoed_folds(df):
        assert (test["issue_year"] == year).all()
        assert year not in set(train["issue_year"]), "test cohort leaked into training"


def test_folds_are_deterministic():
    """Capping uses a stratified sample; an unseeded one would make every
    reported number irreproducible."""
    df = synthetic_book()
    a = [(y, len(tr), len(te), int(tr["default_window"].sum()))
         for y, tr, te in embargoed_folds(df)]
    b = [(y, len(tr), len(te), int(tr["default_window"].sum()))
         for y, tr, te in embargoed_folds(df)]
    assert a == b


def test_nested_inner_split_never_touches_the_outer_test_set():
    """The point of nesting: configurations are chosen without seeing the data
    the reported score comes from."""
    df = synthetic_book()
    folds = list(outer_folds(df))
    assert folds, "no nested folds were produced"

    for year, outer_train, test, inner_train, inner_val, val_year in folds:
        assert year not in set(inner_train["issue_year"])
        assert year not in set(inner_val["issue_year"])
        assert val_year != year


def test_nested_inner_split_respects_temporal_order():
    df = synthetic_book()
    for year, outer_train, test, inner_train, inner_val, val_year in outer_folds(df):
        assert inner_train["issue_year"].max() < val_year, (
            "inner training data must precede the inner validation cohort"
        )
        assert (inner_val["issue_year"] == val_year).all()
        # Both inner halves must lie inside the outer training window.
        cutoff = pd.Timestamp(year=year - EMBARGO_MONTHS // 12, month=1, day=1)
        assert inner_train["issue_d"].max() < cutoff
        assert inner_val["issue_d"].max() < cutoff


def test_folds_are_ordered_and_walk_forward():
    df = synthetic_book()
    years = [y for y, _, _ in embargoed_folds(df)]
    assert years == sorted(years), "folds must advance through time"
    assert len(years) == len(set(years))
