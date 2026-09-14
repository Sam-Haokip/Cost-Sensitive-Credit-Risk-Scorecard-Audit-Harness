"""Decision 13: `load_modelling_frame`'s extra_cols path assumed anything not
listed in `extra_categorical` was numeric and force-coerced it with
`errors="coerce"` -- silently turning a non-numeric column (a zip prefix like
"190xx", or -- it turned out -- the already-in-KEEP_COLS `earliest_cr_line`
date string) into 100% nulls with no error. `_coerce_numeric_or_raise` is the
fix: same coercion, but it raises if the coercion destroys values that
weren't already missing, rather than doing it quietly."""
import numpy as np
import pandas as pd
import pytest

from data.dataset import _coerce_numeric_or_raise


def test_coerce_numeric_passes_through_genuinely_numeric_strings():
    s = pd.Series(["13.99", "7.5", None, "0"])
    out = _coerce_numeric_or_raise(s, "int_rate")
    assert out.tolist() == pytest.approx([13.99, 7.5, np.nan, 0.0], nan_ok=True)


def test_coerce_numeric_raises_on_non_numeric_text():
    s = pd.Series(["190xx", "577xx", "605xx"])
    with pytest.raises(ValueError, match="zip_code"):
        _coerce_numeric_or_raise(s, "zip_code")


def test_coerce_numeric_tolerates_a_few_unparseable_values():
    """A handful of genuinely bad values in an otherwise-numeric column is the
    normal case this must NOT flag -- only near-total destruction should."""
    s = pd.Series([str(x) for x in range(100)] + ["garbage"])  # 1/101 unparseable
    out = _coerce_numeric_or_raise(s, "mostly_numeric")
    assert out.isna().sum() == 1


def test_coerce_numeric_does_not_flag_values_already_null_in_source():
    s = pd.Series(["1.0", None, None, "2.0"])  # 50% null already, but the rest is fine
    out = _coerce_numeric_or_raise(s, "sparse_numeric")
    assert out.isna().sum() == 2  # unchanged -- no new nulls introduced
