# Phase 6 fairness audit: income_quintile / emp_length / geo_race_proxy

Baseline gaps at the Phase 5 cost-optimal ("realistic") threshold, mean +/- sd across 4 folds:

## income_quintile

- **logistic_woe**: demographic parity gap 0.024 (sd 0.015), disparate impact ratio 0.98, equal-opportunity gap 0.022 (sd 0.013)
- **lightgbm**: demographic parity gap 0.018 (sd 0.015), disparate impact ratio 0.98, equal-opportunity gap 0.017 (sd 0.014)

## emp_length

- **logistic_woe**: demographic parity gap 0.055 (sd 0.030), disparate impact ratio 0.94, equal-opportunity gap 0.052 (sd 0.029)
- **lightgbm**: demographic parity gap 0.019 (sd 0.010), disparate impact ratio 0.98, equal-opportunity gap 0.019 (sd 0.010)

## geo_race_proxy

- **logistic_woe**: demographic parity gap 0.011 (sd 0.005), disparate impact ratio 0.99, equal-opportunity gap 0.010 (sd 0.005)
- **lightgbm**: demographic parity gap 0.010 (sd 0.004), disparate impact ratio 0.99, equal-opportunity gap 0.011 (sd 0.008)

Mitigation cost ($/applicant vs. the profit-maximising shared threshold):

- income_quintile / logistic_woe / demographic_parity: $-1.08 (sd $2.36)
- income_quintile / logistic_woe / equal_opportunity: $-0.96 (sd $2.48)
- income_quintile / lightgbm / demographic_parity: $1.88 (sd $1.77)
- income_quintile / lightgbm / equal_opportunity: $1.33 (sd $1.69)
- emp_length / logistic_woe / demographic_parity: $-1.03 (sd $1.09)
- emp_length / logistic_woe / equal_opportunity: $-0.16 (sd $3.22)
- emp_length / lightgbm / demographic_parity: $-0.35 (sd $1.04)
- emp_length / lightgbm / equal_opportunity: $-0.95 (sd $1.67)
- geo_race_proxy / logistic_woe / demographic_parity: $-0.53 (sd $0.41)
- geo_race_proxy / logistic_woe / equal_opportunity: $-1.47 (sd $1.45)
- geo_race_proxy / lightgbm / demographic_parity: $0.28 (sd $0.65)
- geo_race_proxy / lightgbm / equal_opportunity: $-0.81 (sd $1.71)

Note on the negative-cost rows above: with n=4 folds and a cost sd 1-3x the size of the mean, a negative mean here means "not distinguishable from zero given this much data", not "the mitigation is profitable" -- the sign flips fold to fold (see fairness_gaps.csv). It is NOT evidence that fairness constraints are generally free; it is a reminder that a global-pooled Platt calibration is not guaranteed to be equally well-calibrated in every subgroup, and a per-group threshold can incidentally correct some of that miscalibration as a side effect of matching approval rate or TPR.

Note on geo_race_proxy specifically: every number under this proxy is a statement about a ZIP3's Census-measured demographic composition, NOT about any individual borrower's race -- there is no borrower surname in LC's public data, so this is a geography-only proxy, weaker than the surname+geography BISG method regulators use, and it inherits the ecological fallacy: a borrower in a majority-White zip3 who is Black (or the reverse) is silently assigned the zip3's plurality group, not their own. See fairness/geo_proxy.py's docstring for the full reasoning. 5 of 894 zip3 prefixes were flagged insufficient_data (population under 500 in the ACS estimate, or no matching ZCTA data at all). Loans in those zip3s are excluded from the gap/mitigation numbers above entirely, not just left unlabeled -- 162 test-cohort loans across all 4 folds (roughly 0.04% of the dataset). insufficient_data is a Census DATA-QUALITY flag, not a race/ethnicity group, and its N (a few dozen loans per 100k-loan fold) made it behave like one: a noisy ~100% baseline approval rate, and a per-group mitigation threshold fit on an even smaller calibration-cohort slice of it that sometimes made the OVERALL demographic-parity gap WORSE than doing nothing at all.
