# Phase 7 explainability: SHAP vs. permutation importance, logistic-WoE vs. LightGBM

Global importance and permutation importance are both computed on all 4 outer folds (mean across folds reported below); local explanations use the most recent fold (2017) only -- see explainability/explain.py's module docstring for why.

## logistic_woe: top 10 features by mean |SHAP| across folds

| feature | mean \|SHAP\| (log-odds) | mean permutation drop (PR-AUC) | mean SHAP-vs-permutation rank gap |
|---|---:|---:|---:|
| annual_inc | 0.1983 | 0.00608 | 2.2 |
| loan_amnt | 0.1523 | 0.00640 | 2.0 |
| purpose | 0.1503 | 0.00672 | 2.0 |
| term | 0.1380 | 0.00820 | 3.0 |
| fico_range_high | 0.1223 | 0.00244 | 5.2 |
| fico_range_low | 0.1223 | 0.00248 | 4.0 |
| addr_state | 0.1058 | -0.00057 | 46.8 |
| inq_last_6mths | 0.1050 | 0.00338 | 22.8 |
| acc_open_past_24mths | 0.0768 | 0.00570 | 3.5 |
| revol_util | 0.0753 | 0.00101 | 45.0 |

## lightgbm: top 10 features by mean |SHAP| across folds

| feature | mean \|SHAP\| (log-odds) | mean permutation drop (PR-AUC) | mean SHAP-vs-permutation rank gap |
|---|---:|---:|---:|
| fico_range_high | 0.3226 | 0.01050 | 2.0 |
| annual_inc | 0.2842 | 0.01215 | 1.2 |
| loan_amnt | 0.1954 | 0.01567 | 2.8 |
| addr_state | 0.1917 | 0.00147 | 31.2 |
| inq_last_6mths | 0.1615 | 0.00435 | 3.0 |
| purpose | 0.1218 | 0.00498 | 1.5 |
| term | 0.1052 | 0.00272 | 21.5 |
| revol_util | 0.0865 | 0.00144 | 3.0 |
| dti | 0.0822 | -0.00018 | 26.0 |
| acc_open_past_24mths | 0.0737 | 0.00648 | 3.2 |

## Where SHAP and permutation importance disagree most, per fold

The feature with the largest |SHAP rank - permutation rank| gap in each fold/model, and whether a correlated feature already in the SHAP top 20 (WoE-encoded Pearson correlation, |r| >= 0.3) is a plausible reason why shuffling it alone didn't move PR-AUC as much as SHAP's credit to it would suggest:

| fold | model | feature | rank gap | nearest top-20 neighbor | correlation | correlated-feature explanation |
|---|---|---|---:|---|---:|---|
| 2014 | logistic_woe | addr_state | 93 | fico_range_high | +0.02 | not found by this simple check |
| 2014 | lightgbm | dti | 89 | revol_util | +0.09 | not found by this simple check |
| 2015 | logistic_woe | revol_util | 87 | fico_range_low | +0.38 | found |
| 2015 | lightgbm | emp_length | 85 | annual_inc | +0.15 | not found by this simple check |
| 2016 | logistic_woe | total_il_high_credit_limit | 87 | annual_inc | +0.34 | found |
| 2016 | lightgbm | addr_state | 89 | total_il_high_credit_limit | +0.04 | not found by this simple check |
| 2017 | logistic_woe | inq_last_6mths | 81 | mths_since_recent_inq | +0.40 | found |
| 2017 | lightgbm | mo_sin_rcnt_tl | 75 | num_tl_op_past_12m | +0.57 | found |

4/8 fold/model disagreements had a correlated top-20 feature as a candidate explanation; the rest are reported as found, not asserted -- correlation alone can't rule out an interaction effect or permutation-importance sampling noise instead.

## Sanity check: the 6 features Phase 3's WoE-encoder fix rescued

Information value 0.0002-0.0011 at fit time (decision 7, DECISIONS.md) -- essentially nothing. If SHAP, an unrelated method, also ranks them near the bottom of logistic_woe's 96 features, that's an independent confirmation, not a re-derivation of the same number.

| feature | mean \|SHAP\| (log-odds) | rank |
|---|---:|---:|
| acc_now_delinq | 0.00000 | 63/96 |
| tax_liens | 0.00000 | 61/96 |
| pub_rec_bankruptcies | 0.02261 | 27/96 |
| pub_rec | 0.01277 | 37/96 |
| num_tl_90g_dpd_24m | 0.00397 | 50/96 |
| tot_coll_amt | 0.00088 | 56/96 |

## Sanity check: geography feature (addr_state)

Phase 6 found ZIP3-derived geography is a usable, if imperfect, proxy for race/ethnicity; addr_state is data/dataset.py's only geography feature actually fed to either model (raw zip_code is deferred entirely -- see DEFERRED there). Its rank here is reported for that reason, not because high SHAP importance would itself be a fairness problem on its own.

| model | rank |
|---|---:|
| logistic_woe | 7/96 |
| lightgbm | 4/96 |

See reports/explainability_local_examples.md for the 3 representative-applicant local explanations, and reports/explainability_importance.csv for the full per-fold, per-feature importance table.
