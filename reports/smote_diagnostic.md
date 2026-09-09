# Why SMOTE has no effect on this problem

Fold 2016. 150,000 real training rows at 7.90% default, resampled to 276,314 rows at 50% default (126,314 synthetic).

## A. Synthetic defaults are perfectly identifiable

A LightGBM classifier trained to separate synthesised minority rows from real ones, using the minority class only, scores **ROC-AUC 1.0000** on a held-out half.

SMOTE's premise is that a synthetic minority point is as good as a real one. At this separability it is not: the booster can learn `is_synthetic`, which predicts the positive label perfectly in training and does not exist at scoring time. That explains the otherwise strange result in `reports/imbalance.csv` -- SMOTE trains on a 50/50 book yet predicts a test mean of 0.091, close to the true base rate, while undersampling on an equally balanced book predicts 0.463. The synthetic half was effectively partitioned away.

## B. The tell is integrality

51 of the 58 numeric features actually observed in this training window take only whole-number values among real borrowers -- counts of accounts, delinquencies, enquiries. (The remaining 29 of 87 are null throughout the window and carry no information either way.) Interpolating between two real borrowers produces fractional values in 51 of them, so a single split at a non-integer threshold isolates the synthetic population.

| feature               |   pct_synthetic_non_integer |
|:----------------------|----------------------------:|
| revol_bal             |                        99.7 |
| total_acc             |                        96.2 |
| loan_amnt             |                        92.9 |
| fico_range_low        |                        92   |
| fico_range_high       |                        92   |
| open_acc              |                        90.5 |
| bc_open_to_buy        |                        79.9 |
| total_bal_ex_mort     |                        79.9 |
| total_bc_limit        |                        79.3 |
| mths_since_recent_bc  |                        77.9 |
| mths_since_recent_inq |                        73.6 |
| acc_open_past_24mths  |                        70.4 |
| num_sats              |                        70.1 |
| inq_last_6mths        |                        69.5 |
| avg_cur_bal           |                        69.5 |

A borrower with 7.43 open accounts does not exist. SMOTE assumes the straight line between two minority points is itself a plausible minority point; in a feature space of counts, categorical codes, and Phase 1's *missing because the event never happened* columns, it is not.

## What this does not claim

That SMOTE is useless in general. On genuinely continuous feature spaces the interpolation assumption is reasonable. The claim is narrower and measured: on this data it produces a population the model can identify and ignore, which is why the PR-AUC difference is -0.0015 with the sign flipping across folds.
