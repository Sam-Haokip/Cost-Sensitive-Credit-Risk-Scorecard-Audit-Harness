"""
Weight-of-Evidence encoding — the credit-industry standard for scorecards.

WHY WoE HERE, RATHER THAN IMPUTE-AND-ONE-HOT
--------------------------------------------
For each bin of a feature, WoE is

    ln( share of non-defaults in bin / share of defaults in bin )

which is a log-odds quantity. Feeding that to logistic regression means the
model works in the units it is already additive in, so a monotone-but-curved
relationship (default risk against income, say) becomes linear without
hand-crafted splines.

Three reasons it fits this project specifically:

1. **Missingness becomes a bin, not a guess.** Phase 1 found six `mths_since_*`
   columns that are missing precisely because the event never happened — a
   borrower with no delinquency has no "months since last delinquency".
   Median-imputing those replaces "never delinquent" with "delinquent a
   middling time ago" and inverts the signal. WoE gives missing its own bin and
   *learns its own log-odds*, which is exactly the right treatment and requires
   no special-casing.

2. **It handles the vintage-driven columns honestly.** The fourteen bureau
   fields that are 100% null before 2016 get a null bin whose WoE is estimated
   from whatever data the fold has, rather than silently becoming a median.

3. **It is auditable.** A regulator or credit officer can read a WoE table --
   bin, count, default rate, weight -- which is the point of keeping an
   interpretable reference model at all.

The cost is a fitted transform: bins and weights are estimated from the target,
so they MUST be fit on training data only and applied to validation, or the
encoding leaks. `fit` / `transform` are separate here for that reason, and
callers must not fit on anything the model will later be scored against.

Information Value is reported alongside, since it is the standard companion
statistic for feature screening in scorecard work.
"""
import numpy as np
import pandas as pd

DEFAULT_BINS = 10
MIN_BIN_FRACTION = 0.02   # merge bins holding under 2% of rows
SMOOTHING = 0.5           # keeps a bin with zero events from producing +/-inf
MISSING_LABEL = "__missing__"
RARE_LABEL = "__rare__"


class WOEEncoder:
    """Fit per-column bins and weights on training data; apply to any frame."""

    def __init__(self, n_bins=DEFAULT_BINS, min_bin_fraction=MIN_BIN_FRACTION,
                 smoothing=SMOOTHING):
        self.n_bins = n_bins
        self.min_bin_fraction = min_bin_fraction
        self.smoothing = smoothing
        self.edges_ = {}        # numeric column -> bin edges
        self.woe_ = {}          # column -> {bin label: woe}
        self.iv_ = {}           # column -> information value
        self.categorical_ = set()
        self.columns_ = []

    # ---- binning ---------------------------------------------------------
    def _numeric_bins(self, s: pd.Series):
        vals = s.dropna()
        if vals.empty:
            return None
        edges = np.unique(np.nanquantile(vals, np.linspace(0, 1, self.n_bins + 1)))
        if len(edges) < 3:
            return None  # effectively constant -> treated as categorical below
        edges[0], edges[-1] = -np.inf, np.inf
        return edges

    def _assign(self, s: pd.Series, col: str) -> pd.Series:
        """Map raw values to bin labels, with missing as its own explicit bin."""
        if col in self.categorical_:
            out = s.astype("object").where(s.notna(), MISSING_LABEL).astype(str)
            known = self.woe_.get(col, {})
            return out.where(out.isin(known), RARE_LABEL)
        edges = self.edges_.get(col)
        if edges is None:
            return pd.Series(np.where(s.isna(), MISSING_LABEL, RARE_LABEL), index=s.index)
        binned = pd.cut(s, bins=edges, labels=False, include_lowest=True)
        return pd.Series(
            np.where(s.isna(), MISSING_LABEL, pd.Series(binned, index=s.index).astype("Int64").astype(str)),
            index=s.index,
        )

    # ---- fit / transform -------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.columns_ = list(X.columns)
        self.categorical_ = {
            c for c in X.columns
            if str(X[c].dtype) in ("category", "object", "str", "string")
        }
        y = pd.Series(np.asarray(y), index=X.index)
        n_events, n_nonevents = float(y.sum()), float((1 - y).sum())

        for col in X.columns:
            s = X[col]
            if col not in self.categorical_:
                self.edges_[col] = self._numeric_bins(s)

            # Provisional labels, then merge anything too small to estimate on.
            labels = self._assign_for_fit(s, col)
            counts = labels.value_counts()
            rare = counts[counts < self.min_bin_fraction * len(labels)].index
            if len(rare):
                labels = labels.where(~labels.isin(rare), RARE_LABEL)

            grouped = pd.DataFrame({"bin": labels, "y": y.values}).groupby("bin")["y"]
            events, totals = grouped.sum(), grouped.size()
            nonevents = totals - events

            k = max(len(totals), 1)
            p_event = (events + self.smoothing) / (n_events + self.smoothing * k)
            p_nonevent = (nonevents + self.smoothing) / (n_nonevents + self.smoothing * k)
            woe = np.log(p_nonevent / p_event)

            self.woe_[col] = woe.to_dict()
            self.iv_[col] = float(((p_nonevent - p_event) * woe).sum())
        return self

    def _assign_for_fit(self, s: pd.Series, col: str) -> pd.Series:
        """Bin assignment during fit, before the woe table exists."""
        if col in self.categorical_:
            return s.astype("object").where(s.notna(), MISSING_LABEL).astype(str)
        edges = self.edges_.get(col)
        if edges is None:
            return pd.Series(np.where(s.isna(), MISSING_LABEL, RARE_LABEL), index=s.index)
        binned = pd.cut(s, bins=edges, labels=False, include_lowest=True)
        return pd.Series(
            np.where(s.isna(), MISSING_LABEL, pd.Series(binned, index=s.index).astype("Int64").astype(str)),
            index=s.index,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in self.columns_:
            table = self.woe_[col]
            labels = self._assign(X[col], col)
            # unseen bins fall back to 0 == neutral evidence
            out[col] = labels.map(table).astype("float32").fillna(0.0)
        return pd.DataFrame(out, index=X.index)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def information_values(self) -> pd.Series:
        return pd.Series(self.iv_).sort_values(ascending=False)
