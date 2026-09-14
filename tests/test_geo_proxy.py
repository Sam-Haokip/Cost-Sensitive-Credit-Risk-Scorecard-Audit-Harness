"""Phase 6: fairness/geo_proxy.py's pure aggregation logic, tested with
synthetic ZCTA-shaped data -- no live Census call needed (and none possible
from this sandbox; see the module docstring). Three real failure modes this
guards against: (1) averaging per-ZCTA percentages instead of summing raw
counts first, which silently un-weights population; (2) trusting Census's
negative sentinel codes (-666666666 and friends) as real population counts;
(3) a low-population zip3 producing a confident-looking plurality label."""
import numpy as np
import pandas as pd
import pytest

from fairness.geo_proxy import (
    RACE_GROUPS,
    _parse_acs_response,
    aggregate_zcta_to_zip3,
    assign_race_proxy_group,
    build_zip3_race_proxy,
)


def _acs_row(zcta, total, white=0, black=0, aian=0, asian=0, nhpi=0, other_race=0, two_or_more=0, hispanic=0):
    """One synthetic Census API data row, in the exact string-typed,
    positional-list shape the real API returns (NAME first, geography id
    last) -- exercises _parse_acs_response's actual column handling rather
    than a pre-built DataFrame."""
    return ["Fake ZCTA", str(total), str(white), str(black), str(aian), str(asian),
            str(nhpi), str(other_race), str(two_or_more), str(hispanic), zcta]


_HEADER = ["NAME", "B03002_001E", "B03002_003E", "B03002_004E", "B03002_005E",
           "B03002_006E", "B03002_007E", "B03002_008E", "B03002_009E", "B03002_012E",
           "zip code tabulation area"]


def test_parse_acs_response_renames_and_zero_pads_zcta():
    rows = [_HEADER, _acs_row("1001", total=500, white=400, black=100)]
    df = _parse_acs_response(rows)
    assert df["zcta"].tolist() == ["01001"]  # zero-padded to 5 digits
    assert df["total"].iloc[0] == 500
    assert df["white_nonhispanic"].iloc[0] == 400
    assert df["black_nonhispanic"].iloc[0] == 100


def test_parse_acs_response_treats_negative_sentinels_as_missing():
    """-666666666 ("insufficient sample observations") and its siblings are
    real values the Census API returns in place of an estimate -- summing
    one into a population total would corrupt it by two-thirds of a billion
    "people". Any negative count must become NaN, not survive as a number."""
    row = _acs_row("10001", total=-666666666, white=-999999999, black=50, hispanic=10)
    df = _parse_acs_response([_HEADER, row])
    assert np.isnan(df["total"].iloc[0])
    assert np.isnan(df["white_nonhispanic"].iloc[0])
    assert df["black_nonhispanic"].iloc[0] == 50  # untouched, genuinely non-negative
    assert df["hispanic"].iloc[0] == 10


def test_aggregate_zcta_to_zip3_weights_by_population_not_by_zcta_count():
    """Two ZCTAs sharing zip3 '100': one has 1000 people (80% white, 20%
    black), the other has 200 people (0% white, 100% black). Population-
    weighted, the zip3 is (800+0)/1200 = 66.7% white -- the plurality.
    Naively AVERAGING the two ZCTAs' percentages instead (80% and 0%,
    unweighted) would give 40% white vs. a 55% average black share, flipping
    the plurality to black. This is exactly the bug population-weighting
    prevents, so the assertion checks the number, not just "some result"."""
    zcta_df = pd.DataFrame([
        {"zcta": "10001", "total": 1000, "white_nonhispanic": 800, "black_nonhispanic": 200,
         "aian_nonhispanic": 0, "asian_nonhispanic": 0, "nhpi_nonhispanic": 0,
         "other_race_nonhispanic": 0, "two_or_more_nonhispanic": 0, "hispanic": 0},
        {"zcta": "10099", "total": 200, "white_nonhispanic": 0, "black_nonhispanic": 200,
         "aian_nonhispanic": 0, "asian_nonhispanic": 0, "nhpi_nonhispanic": 0,
         "other_race_nonhispanic": 0, "two_or_more_nonhispanic": 0, "hispanic": 0},
    ])
    zip3_df = aggregate_zcta_to_zip3(zcta_df)
    row = zip3_df[zip3_df["zip3"] == "100"].iloc[0]
    assert row["total"] == 1200
    assert row["white_nonhispanic"] == 800
    assert row["black_nonhispanic"] == 400
    assert row["pct_white_nonhispanic"] == pytest.approx(800 / 1200)
    assert row["pct_black_nonhispanic"] == pytest.approx(400 / 1200)
    # The population-weighted plurality is white -- an unweighted average of
    # the two ZCTAs' percentages would have said black (see docstring above).
    assert row["pct_white_nonhispanic"] > row["pct_black_nonhispanic"]


def test_aggregate_zcta_to_zip3_rolls_small_categories_into_other():
    zcta_df = pd.DataFrame([{
        "zcta": "20001", "total": 1000, "white_nonhispanic": 500, "black_nonhispanic": 200,
        "aian_nonhispanic": 50, "asian_nonhispanic": 100, "nhpi_nonhispanic": 20,
        "other_race_nonhispanic": 30, "two_or_more_nonhispanic": 100, "hispanic": 0,
    }])
    zip3_df = aggregate_zcta_to_zip3(zcta_df)
    row = zip3_df.iloc[0]
    assert row["other_nonhispanic"] == 50 + 20 + 30 + 100  # aian + nhpi + other_race + two_or_more
    assert set(f"pct_{g}" for g in RACE_GROUPS) <= set(zip3_df.columns)


def test_assign_race_proxy_group_picks_the_plurality():
    zip3_df = pd.DataFrame([
        {"zip3": "100", "total": 10000, "pct_white_nonhispanic": 0.45, "pct_black_nonhispanic": 0.30,
         "pct_asian_nonhispanic": 0.10, "pct_hispanic": 0.10, "pct_other_nonhispanic": 0.05},
        {"zip3": "200", "total": 10000, "pct_white_nonhispanic": 0.20, "pct_black_nonhispanic": 0.55,
         "pct_asian_nonhispanic": 0.05, "pct_hispanic": 0.15, "pct_other_nonhispanic": 0.05},
    ])
    out = assign_race_proxy_group(zip3_df, min_population=500)
    assert out.set_index("zip3")["race_proxy_group"].to_dict() == {
        "100": "white_nonhispanic", "200": "black_nonhispanic",
    }


def test_assign_race_proxy_group_flags_low_population_zip3s():
    """A zip3 with real but tiny population (a PO-box cluster, a military
    installation) must not get a confident-looking plurality label."""
    zip3_df = pd.DataFrame([
        {"zip3": "300", "total": 42, "pct_white_nonhispanic": 0.9, "pct_black_nonhispanic": 0.1,
         "pct_asian_nonhispanic": 0.0, "pct_hispanic": 0.0, "pct_other_nonhispanic": 0.0},
    ])
    out = assign_race_proxy_group(zip3_df, min_population=500)
    assert out["race_proxy_group"].iloc[0] == "insufficient_data"


def test_assign_race_proxy_group_flags_zip3_with_no_real_data():
    """total=NaN happens when every ZCTA in a zip3 was entirely suppressed
    (all sentinel values) -- fillna(0) in the population check must catch
    this the same way as a genuinely tiny population, not crash on NaN < int."""
    zip3_df = pd.DataFrame([
        {"zip3": "400", "total": float("nan"), "pct_white_nonhispanic": float("nan"),
         "pct_black_nonhispanic": float("nan"), "pct_asian_nonhispanic": float("nan"),
         "pct_hispanic": float("nan"), "pct_other_nonhispanic": float("nan")},
    ])
    out = assign_race_proxy_group(zip3_df, min_population=500)
    assert out["race_proxy_group"].iloc[0] == "insufficient_data"


def test_build_zip3_race_proxy_chains_fetch_aggregate_and_assign(monkeypatch):
    """End-to-end wiring check: build_zip3_race_proxy must call the fetch
    function with the api_key it was given, then pipe its result through
    both aggregation steps -- verified by monkeypatching only the network
    call, so this still exercises the two real aggregation functions."""
    def fake_fetch(api_key, *args, **kwargs):
        assert api_key == "fake-key-123"
        return pd.DataFrame([
            {"zcta": "50001", "total": 5000, "white_nonhispanic": 4000, "black_nonhispanic": 500,
             "aian_nonhispanic": 0, "asian_nonhispanic": 300, "nhpi_nonhispanic": 0,
             "other_race_nonhispanic": 0, "two_or_more_nonhispanic": 0, "hispanic": 200},
        ])

    monkeypatch.setattr("fairness.geo_proxy.fetch_acs_zcta_race", fake_fetch)
    out = build_zip3_race_proxy("fake-key-123")
    assert out["zip3"].iloc[0] == "500"
    assert out["race_proxy_group"].iloc[0] == "white_nonhispanic"
