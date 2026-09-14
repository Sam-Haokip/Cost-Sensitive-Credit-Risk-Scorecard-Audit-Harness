"""
Phase 6: geography as a race/ethnicity proxy, built from Census ACS 5-year
estimates at ZCTA (5-digit ZIP Code Tabulation Area) granularity, aggregated
up to the zip3 prefix Lending Club's public data actually exposes.

WHY THIS IS A WEAKER PROXY THAN REAL BISG, AND WHY IT'S USED ANYWAY
----------------------------------------------------------------------
Bayesian Improved Surname Geocoding (BISG) -- the standard regulators (e.g.
the CFPB) use to proxy race/ethnicity in lending data that lacks self-
reported race -- combines a borrower's SURNAME with their GEOGRAPHY. Lending
Club's public dataset drops the borrower's name entirely, so only the
geography half is possible here: this module estimates, for each zip3
prefix, what fraction of its residents fall into each race/ethnicity
category, and assigns each loan the PLURALITY group of its zip3.

That is a materially weaker signal than BISG for three independent reasons:
  1. No surname half. Removing the strongest single BISG predictor leaves a
     proxy that is right on average across a geography, but not for any one
     applicant in it.
  2. The ECOLOGICAL FALLACY. Even a perfect zip3-level race distribution
     doesn't tell you the race of any individual borrower in it -- a zip3
     that is 40% Black and 60% White does not mean a White applicant there
     is "40% likely" to be Black; treating an area-level average as an
     individual-level probability is exactly the ecological fallacy this
     limitation is named for. Any Black or Hispanic borrower living in a
     majority-White zip3 (and vice versa) is silently assigned the WRONG
     group by a plurality rule -- there is no way around this with geography
     alone, at any level of Census data precision.
  3. LC's public zip is truncated to 3 digits ("190xx", not "19104") for
     borrower privacy, coarser than the 5-digit ZCTA Census publishes race
     data at, forcing a second aggregation step (population-WEIGHTED, by
     summing raw counts before dividing, never by averaging percentages) on
     top of the geography-vs-surname gap above.

None of this makes the exercise pointless -- regulators use geography-only
proxies too (the CFPB's own "Geographic Proxy Method" sits alongside full
BISG) precisely because it's the only signal available in delivered credit
data that lacks self-reported race. But every number downstream of this
module is a statement about ZIP3-LEVEL demographic composition, not a
statement about any individual borrower's race -- that distinction has to
survive into the write-up, not just live in this docstring.

DATA SOURCE
-----------
American Community Survey (ACS) 5-Year Estimates, table B03002 (Hispanic or
Latino Origin by Race) -- the standard table for a MUTUALLY EXCLUSIVE
race/ethnicity split, unlike a race-alone table (B02001) where a Hispanic
respondent of any race would double-count into both a race category and
"Hispanic". Vintage: 2018 5-year estimates (data collected 2014-2018), chosen
to sit inside this project's own test-fold window (2014-2017) rather than
reaching for the newest available vintage -- a zip3's racial composition
does shift over a decade, and matching the vintage to the outcomes being
measured is the smaller of the two approximations already being made here.

WHY THE LIVE FETCH CAN'T RUN INSIDE CLAUDE'S OWN TOOL CALLS
----------------------------------------------------------------
api.census.gov is not on this session's egress allowlist -- confirmed from
both the cloud sandbox and the connected Mac's shell (same policy, "do not
retry or route around it, report the blocked host" per the proxy's own
troubleshooting doc). `fetch_acs_zcta_race` still does a normal `requests`
call because that's the correct code either way: it just has to be run
somewhere with real, unproxied internet access -- a plain terminal on the
user's own machine, not a Claude-driven shell. Everything downstream of the
fetch (`aggregate_zcta_to_zip3`, `assign_race_proxy_group`) is pure and
fully covered by tests using synthetic ZCTA data, no network required.
"""
import os
import time

import pandas as pd

ACS_YEAR = 2018
ACS_BASE_URL = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"

# B03002: Hispanic or Latino Origin by Race -- gives mutually exclusive
# race/ethnicity categories directly (unlike B02001, which would double-count
# a Hispanic respondent of any race into a race category too).
_B03002_FIELDS = {
    "B03002_001E": "total",
    "B03002_003E": "white_nonhispanic",
    "B03002_004E": "black_nonhispanic",
    "B03002_005E": "aian_nonhispanic",        # American Indian / Alaska Native
    "B03002_006E": "asian_nonhispanic",
    "B03002_007E": "nhpi_nonhispanic",        # Native Hawaiian / Pacific Islander
    "B03002_008E": "other_race_nonhispanic",
    "B03002_009E": "two_or_more_nonhispanic",
    "B03002_012E": "hispanic",
}

# The 4 small non-Hispanic categories get rolled into one "other" bucket --
# keeping them separate would fragment an already geography-only proxy into
# groups too small per zip3 to say anything reliable about.
RACE_GROUPS = ["white_nonhispanic", "black_nonhispanic", "asian_nonhispanic", "hispanic", "other_nonhispanic"]

DEFAULT_CACHE_PATH = "data/processed/census_zip3_race_proxy.csv"

# Below this many people, a zip3's "plurality" group is a coin flip dressed
# up as a finding -- 500 is small for a zip3 (real ones run tens of
# thousands) and only screens out the handful that are almost entirely
# non-residential (PO-box clusters, military/government zips).
_MIN_ZIP3_POPULATION = 500


def fetch_acs_zcta_race(api_key, year=ACS_YEAR, base_url=None, timeout=90):
    """One live call to the Census ACS 5-year API for every ZCTA in the
    country. Requires real, unproxied internet access -- see the module
    docstring for why that means a plain terminal, not a Claude-driven shell.

    `in=state:*` is not optional: the dataset's own geography metadata
    (api.census.gov/data/2018/acs/acs5/geography.json) lists "zip code
    tabulation area" as `"requires": ["state"]` -- a query for ZCTAs alone,
    with no state qualifier at all, isn't a documented request shape. It
    still fetches every ZCTA nationally in one call (state is separately
    marked `"wildcard": ["state"]`, so `state:*` doesn't restrict anything);
    it just has to be present. Confirmed the hard way: omitting it doesn't
    fail with an HTTP error status -- the API answers 200 with a PLAIN-TEXT
    error message instead of JSON, which surfaces here as an opaque
    JSONDecodeError unless the response body is surfaced first."""
    import requests

    url = base_url or (f"https://api.census.gov/data/{year}/acs/acs5" if base_url is None else base_url)
    fields = ",".join(["NAME"] + list(_B03002_FIELDS))
    params = {"get": fields, "for": "zip code tabulation area:*", "in": "state:*", "key": api_key}
    resp = requests.get(url, params=params, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Census API returned HTTP {resp.status_code} for {resp.url!r}.\n"
            f"Body (first 2000 chars): {resp.text[:2000]!r}"
        )
    try:
        rows = resp.json()
    except ValueError as e:
        raise RuntimeError(
            "Census API returned HTTP 200 but the body isn't valid JSON -- this "
            "is how the API reports a malformed request (a plain-text message, "
            "not a JSON error object), not a sign the API itself is down.\n"
            f"URL: {resp.url!r}\nBody (first 2000 chars): {resp.text[:2000]!r}"
        ) from e
    return _parse_acs_response(rows)


def _parse_acs_response(rows):
    """Split out from fetch_acs_zcta_race so a test can hand it exactly what
    the Census API returns (a header row + data rows, all-string JSON)
    without needing a live call or a mocked `requests` at all.

    ACS estimates use negative sentinel codes for "not a real number", not
    NaN or null: -666666666 ("insufficient sample observations"),
    -999999999, -888888888, -222222222, -333333333 are all documented
    (census.gov/data/developers/data-sets/acs-1year/notes-on-acs-estimate-
    and-annotation-values.html). A population count can never legitimately
    be negative, so treating every negative value as missing catches all of
    them (and any future one Census adds) without hard-coding each constant
    -- silently summing -666666666 into a zip3's population would otherwise
    corrupt that zip3's aggregate by two-thirds of a billion people."""
    header, data = rows[0], rows[1:]
    df = pd.DataFrame(data, columns=header)
    df = df.rename(columns=_B03002_FIELDS)
    zcta_col = "zip code tabulation area"
    df["zcta"] = df[zcta_col].astype(str).str.zfill(5)
    numeric_cols = list(_B03002_FIELDS.values())
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        df.loc[df[col] < 0, col] = float("nan")
    df = df[["zcta"] + numeric_cols]
    # Querying with in=state:* returns one row per (zcta, state) pair. Census
    # assigns each ZCTA to a single state in its own crosswalk even though a
    # ZIP's real-world delivery area can cross a state line, so duplicate
    # zcta rows aren't expected -- but this is exactly the kind of assumption
    # to check against the real response rather than trust, so: drop exact
    # duplicates defensively and let the __main__ block's own printed row
    # count make a silent drop visible if this assumption turns out wrong.
    return df.drop_duplicates(subset="zcta", keep="first")


def aggregate_zcta_to_zip3(zcta_df):
    """Sum raw population counts (not percentages) across every ZCTA sharing
    a zip3 prefix, THEN divide -- summing counts before dividing IS the
    population weighting: a zip3 covering one huge ZCTA and nine tiny ones
    ends up close to the huge one's composition, not an unweighted average
    of ten equally-counted percentages."""
    df = zcta_df.copy()
    df["zip3"] = df["zcta"].str[:3]
    count_cols = list(_B03002_FIELDS.values())
    zip3 = df.groupby("zip3")[count_cols].sum(min_count=1)
    zip3["other_nonhispanic"] = (
        zip3["aian_nonhispanic"] + zip3["nhpi_nonhispanic"]
        + zip3["other_race_nonhispanic"] + zip3["two_or_more_nonhispanic"]
    )
    for g in RACE_GROUPS:
        zip3[f"pct_{g}"] = zip3[g] / zip3["total"]
    return zip3.reset_index()


def assign_race_proxy_group(zip3_df, min_population=_MIN_ZIP3_POPULATION):
    """The PLURALITY race/ethnicity group per zip3 -- the single largest
    share, not a majority (most US zip3s have no group over 50%). A zip3
    with total population under `min_population` (or with no matching ZCTA
    data at all -- total is NaN) gets "insufficient_data" rather than a
    confident-looking label built on almost nothing."""
    df = zip3_df.copy()
    pct_cols = [f"pct_{g}" for g in RACE_GROUPS]
    too_small = df["total"].fillna(0) < min_population
    # A zip3 with NO real ZCTA data has every pct_* column NaN -- idxmax
    # raises on an all-NaN row rather than returning NaN, so fill first with
    # a value that can never legitimately win (a real share is >= 0); the
    # arbitrary column idxmax then picks is immediately overwritten by the
    # too_small/no-data check below, since a NaN total always fails it too.
    df["race_proxy_group"] = df[pct_cols].fillna(-1).idxmax(axis=1).str.replace("pct_", "", regex=False)
    df.loc[too_small, "race_proxy_group"] = "insufficient_data"
    return df


def build_zip3_race_proxy(api_key, min_population=_MIN_ZIP3_POPULATION):
    zcta_df = fetch_acs_zcta_race(api_key)
    zip3_df = aggregate_zcta_to_zip3(zcta_df)
    return assign_race_proxy_group(zip3_df, min_population=min_population)


def load_zip3_race_proxy(path=DEFAULT_CACHE_PATH):
    """Loads the cached, committed table this module's __main__ block
    produces -- the audit script depends on THIS, not a live API call, so
    the repo stays reproducible for anyone cloning it without a Census key.
    Regenerating it (a new ACS vintage, say) is a deliberate, separate step:
    `python -m fairness.geo_proxy`, run somewhere with real internet access."""
    return pd.read_csv(path, dtype={"zip3": str})


if __name__ == "__main__":
    api_key = os.environ.get("LC_CENSUS_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set LC_CENSUS_API_KEY before running this (see fairness/geo_proxy.py's "
            "module docstring for why -- this script needs real internet access, "
            "which is why it isn't run automatically as part of the audit). Get a "
            "free key at https://api.census.gov/data/key_signup.html -- no "
            "password, just an email address; the key is emailed to you directly."
        )
    print(f"Fetching ACS {ACS_YEAR} 5-year estimates (table B03002) for every ZCTA...")
    t0 = time.time()
    zcta_df = fetch_acs_zcta_race(api_key)
    print(f"fetched {len(zcta_df):,} unique ZCTA rows in {time.time() - t0:.1f}s "
          f"(after dropping any (zcta, state) duplicates -- see _parse_acs_response's "
          f"docstring for why that check exists)")
    zip3_df = aggregate_zcta_to_zip3(zcta_df)
    result = assign_race_proxy_group(zip3_df)
    print(f"aggregated to {len(result):,} zip3 prefixes")
    print(result["race_proxy_group"].value_counts().to_string())
    os.makedirs(os.path.dirname(DEFAULT_CACHE_PATH), exist_ok=True)
    result.to_csv(DEFAULT_CACHE_PATH, index=False)
    print(f"wrote {DEFAULT_CACHE_PATH}")
