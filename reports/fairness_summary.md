# Phase 6 fairness audit: income_quintile / emp_length

Baseline gaps at the Phase 5 cost-optimal ("realistic") threshold, mean +/- sd across 4 folds:

## income_quintile

- **logistic_woe**: demographic parity gap 0.024 (sd 0.015), disparate impact ratio 0.98, equal-opportunity gap 0.022 (sd 0.013)
- **lightgbm**: demographic parity gap 0.018 (sd 0.015), disparate impact ratio 0.98, equal-opportunity gap 0.017 (sd 0.014)

## emp_length

- **logistic_woe**: demographic parity gap 0.055 (sd 0.030), disparate impact ratio 0.94, equal-opportunity gap 0.052 (sd 0.029)
- **lightgbm**: demographic parity gap 0.019 (sd 0.010), disparate impact ratio 0.98, equal-opportunity gap 0.019 (sd 0.010)

Mitigation cost ($/applicant vs. the profit-maximising shared threshold):

- income_quintile / logistic_woe / demographic_parity: $-1.08 (sd $2.36)
- income_quintile / logistic_woe / equal_opportunity: $-0.96 (sd $2.48)
- income_quintile / lightgbm / demographic_parity: $1.88 (sd $1.77)
- income_quintile / lightgbm / equal_opportunity: $1.33 (sd $1.69)
- emp_length / logistic_woe / demographic_parity: $-1.03 (sd $1.09)
- emp_length / logistic_woe / equal_opportunity: $-0.16 (sd $3.22)
- emp_length / lightgbm / demographic_parity: $-0.35 (sd $1.04)
- emp_length / lightgbm / equal_opportunity: $-0.95 (sd $1.67)

Note on the negative-cost rows above: with n=4 folds and a cost sd 1-3x the size of the mean, a negative mean here means "not distinguishable from zero given this much data", not "the mitigation is profitable" -- the sign flips fold to fold (see fairness_gaps.csv). It is NOT evidence that fairness constraints are generally free; it is a reminder that a global-pooled Platt calibration is not guaranteed to be equally well-calibrated in every subgroup, and a per-group threshold can incidentally correct some of that miscalibration as a side effect of matching approval rate or TPR.
