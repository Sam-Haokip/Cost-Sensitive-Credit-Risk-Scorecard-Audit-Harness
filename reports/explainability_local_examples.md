# Phase 7: local explanations for 3 representative applicants

logistic-WoE scorecard, most recent fold, decision-time information only (never the hindsight outcome label).

## clear_approve (fold 2017, threshold 0.199, score 0.004)

| feature | value | contribution (log-odds) | direction |
|---|---|---:|---|
| annual_inc | 95000.0 | -0.283 | lowers risk |
| fico_range_low | 805.0 | -0.262 | lowers risk |
| fico_range_high | 809.0 | -0.262 | lowers risk |
| acc_open_past_24mths | 0.0 | -0.242 | lowers risk |
| purpose | credit_card | -0.222 | lowers risk |
| mths_since_recent_bc | 113.0 | -0.197 | lowers risk |
| bc_open_to_buy | 35689.0 | -0.187 | lowers risk |
| mo_sin_rcnt_tl | 39.0 | -0.180 | lowers risk |

## clear_reject (fold 2017, threshold 0.199, score 0.459)

| feature | value | contribution (log-odds) | direction |
|---|---|---:|---|
| purpose | small_business | +0.780 | raises risk |
| loan_amnt | 32000.0 | +0.307 | raises risk |
| term |  60 months | +0.300 | raises risk |
| acc_open_past_24mths | 13.0 | +0.295 | raises risk |
| mths_since_recent_inq | 0.0 | +0.201 | raises risk |
| mths_since_recent_bc | 0.0 | +0.186 | raises risk |
| num_tl_op_past_12m | 11.0 | +0.151 | raises risk |
| fico_range_low | 670.0 | +0.145 | raises risk |

## borderline (fold 2017, threshold 0.199, score 0.199)

| feature | value | contribution (log-odds) | direction |
|---|---|---:|---|
| annual_inc | 16000.0 | +0.334 | raises risk |
| fico_range_low | 660.0 | +0.153 | raises risk |
| fico_range_high | 664.0 | +0.153 | raises risk |
| acc_open_past_24mths | 6.0 | +0.151 | raises risk |
| dti | 30.17 | +0.139 | raises risk |
| mo_sin_old_rev_tl_op | 40.0 | +0.133 | raises risk |
| mo_sin_rcnt_tl | 1.0 | +0.118 | raises risk |
| tot_hi_cred_lim | 21818.0 | +0.102 | raises risk |

