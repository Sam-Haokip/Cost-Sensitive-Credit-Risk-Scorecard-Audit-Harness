# Decision log

Every modelling decision recorded with the alternative it was chosen over, and — where one exists — the measured cost of choosing it. "I used X" is not a justification; "I used X over Y, and here is what that cost" is.

Decisions are listed in the order they were made.

---

## 1. Dataset: Lending Club over Home Credit Default Risk

**Decision.** Build on the Lending Club accepted-loans dataset (2007–2018), not Home Credit Default Risk.

**Alternative considered.** Home Credit — relational tables (application, bureau, previous applications, instalments) would exercise more join- and aggregation-heavy feature engineering, which a single flat table doesn't require.

**Why.** Temporal validation is the backbone of this project: time-based splits, a walk-forward backtest, and a measured random-vs-temporal gap. Home Credit's application data carries no true calendar origination date — only anonymised relative day-counts (`DAYS_BIRTH`, `DAYS_EMPLOYED` as negative offsets from application). A genuine multi-year walk-forward backtest isn't achievable on it. Lending Club has a real `issue_d` spanning June 2007 to December 2018 with no missing months. Relational complexity is real value, but not at the cost of calendar-time validity.

**Verified.** 2,260,701 rows, 151 columns, 139 distinct origination months, no gaps.

---

## 2. Target: resolved loans only, Charged Off vs Fully Paid

> **SUPERSEDED by decision 5.** Kept as the record of what was decided and why it was later overturned. The figures below describe this target, not the one in current use.

**Decision.** `default = 1` for Charged Off / Default, `0` for Fully Paid. All other statuses excluded from the modelling population entirely — not soft-labelled, not imputed, removed.

**Alternative considered.** Redefining the target as "ever 90+ days past due", a real industry practice that treats severe delinquency as a proxy for eventual charge-off. This would pull the 21,467 Late (31-120 days) loans into the positive class and shrink the excluded population.

**Why the simple version.** Non-terminal loans have no outcome, and no honest way to invent one. Labelling Current loans as "safe" injects an error that correlates with *how recently the loan was issued* — a loan originated two months before the data was pulled is Current because time hasn't passed, not because the borrower is sound. That error would flow directly into the temporal validation this project is built on, teaching the model that recent vintages are low-risk. Labelling Late loans as defaults has the mirror problem: many cure and finish Fully Paid.

The 90+ DPD alternative is defensible, but it changes the question the model answers and would need to be carried through calibration, cost analysis and fairness as a second parallel track. That cost isn't justified by the value of the comparison here.

**Also excluded.** "Does not meet the credit policy" statuses (2,749 loans) — outcome known, but underwritten under rules since retired; mixing two underwriting regimes into one target muddies what the model learns. And 33 rows with null `loan_status` and null `issue_d`, which are corrupted footer rows rather than loans.

**Measured cost.** Modelling population falls to 1,345,350 of 2,260,701 rows (59.5%), 19.97% default rate. The loss is not random: it skews toward older cohorts. By 2018, 88% of 36-month loans and 90% of 60-month loans had not yet resolved and are therefore absent. Quantified by issue year and term in `data/target.py`; this is measured and disclosed, not corrected.

---

## 3. Feature selection: a documented allow-list, not a drop-list

**Decision.** Classify all 151 raw columns individually with a stated reason each. A column is included only if it can be justified as available at the moment of the credit decision. Anything that cannot be is cut. Implemented in `data/feature_audit.py`; validated to cover exactly the 151 columns present, with no gaps or typos.

**Alternative considered.** A drop-list — remove the obviously suspicious columns (`recoveries`, `total_pymnt`) and keep the rest. This is faster and is what most public work on this dataset does.

**Why.** A drop-list is a statement about the columns you happened to think of. An allow-list forces a positive justification for every column that survives, which is how the less obvious leaks get caught — the hardship programme (15 columns) and debt settlement (7 columns) fields are not intuitively "payment data", but both only populate for loans that already went bad.

**Result.** 101 kept, 38 dropped as leakage, 8 dropped as identifiers/constants/redundant/label-source, 4 flagged for review (decision 4).

**Measured cost of getting this wrong.** Same model, same split, trained on all available columns versus the allow-list: PR-AUC 0.9972 vs 0.1944 on the current (decision 5) target. A +0.80 gap. A near-perfect credit risk model is a symptom, not an achievement — the naive version is reading the outcome, not predicting it.

---

## 4. Excluding Lending Club's own risk grade

**Decision.** Exclude `grade`, `sub_grade`, `int_rate` and `installment` from the model.

**The tension.** These are available before the loan is funded, so they are not leakage in the "happened afterwards" sense that decision 3 covers. But they are *outputs of Lending Club's own underwriting model*, not raw applicant characteristics: `grade` is computed from the applicant's risk profile, and `int_rate` (hence `installment`) follows from the grade. Training on them risks a model that re-derives Lending Club's decision rather than learning the underlying default relationship — strong reported performance that wouldn't transfer to a setting where their grade doesn't exist.

**Alternative considered.** Keep them and add policy vintage as a control. This is what most public work does, and it is not unreasonable — they genuinely are known pre-funding.

**Why measured rather than argued.** Both positions are defensible from reasoning alone, so reasoning alone shouldn't decide it.

**Measured cost — REVISED after decision 5.** Adding all four moves PR-AUC by **+0.0110 on average across five seeds** (sd 0.0026, range +0.0076 to +0.0143; `models/seed_stability.py`), positive on every seed, against a seed-to-seed spread in the baseline of only 0.0020. The effect is roughly five times its own measurement noise — about a 6% relative gain on a 0.1896 baseline.

**This overturns an earlier conclusion, and the correction is the point.** Measured on the retired decision-2 target, the same comparison gave +0.0037 against baseline noise of 0.0042 — smaller than the measurement error, which was the basis for calling the exclusion costless. That claim did not survive re-measurement on the current target. The columns are still excluded, on the circularity argument, which stands on its own: a model that re-derives Lending Club's grade has learned their decision, not the underlying risk, and would not transfer to a setting where their grade does not exist. But the decision now carries a real and stated price rather than a free one.

**Process note.** This was only caught because every experiment was re-run against the new target rather than leaving old numbers in place beside new ones. A README quoting two target definitions would have been confusing; one quoting a conclusion that its own current code contradicts would have been worse.

---

## 5. Replacing the target with a fixed 18-month performance window

**Decision.** Supersede decision 2. A loan is labelled `1` if it charged off having stopped paying within **18 months of origination**, `0` if it survived that window (repaid, still performing, or charged off only afterwards). A loan is eligible only if observed for 18 + 6 months, the extra six covering Lending Club's ~120-150 day delinquency-to-charge-off lag. Implemented in `data/vintage_target.py`.

**What forced the revision.** Decision 2's "final status" target makes cohorts non-comparable, and Phase 2 is built on comparing cohorts. Defaults resolve fast and full repayments resolve slowly, so a partly-resolved cohort is enriched with whichever outcome finishes sooner. Measured that way, the 2016 and 2017 36-month cohorts show ~20% default against 12-15% for every fully-resolved year before them — and the 2018 cohort drops back to 13%, because at under twelve months of age the pool is dominated by early full repayments instead. The bias is not merely present, it reverses direction with cohort age. A naive time-based split would have reported all of that as credit drift.

**Alternatives considered.** (a) Restrict to fully-resolved cohorts only — clean, but discards 2016 onward and leaves too few cohorts for a walk-forward backtest. (b) Proceed naively and document the bias — what most public work does, and now demonstrably wrong.

**Why 18 months, measured rather than assumed.** Time from origination to final payment on charged-off 36-month loans (2012-2014 cohorts, fully resolved so the timing itself isn't censored) has a median of 17 months. Cumulative share of eventual charge-offs caught: 34% by month 12, **56% by month 18**, 76% by 24, 92% by 30. Longer windows capture more defaults but cost usable cohorts, since every loan needs window + lag months of observation. 18 captures a majority while preserving 2012 to early-2017 for folds. Set by `WINDOW_MONTHS`; changing it is one line.

**Measured effect.** Cohort default rate for 36-month loans goes from a 12.3%-20.0% swing under decision 2 to 6.9%-9.8% under the window — a gentle rise peaking in 2016 and easing in 2017, consistent with a real credit cycle rather than an artefact. 60-month loans sit consistently above 36-month at every vintage (13.3% vs 9.8% in 2016), a sanity check that real risk is still being captured.

**Cost, and an unexpected gain.** The target now means "defaulted early" rather than "ever defaulted", so it answers a narrower question and the base rate falls from 19.97% to 9.63%. But the eligible population *grows*, from 1,345,350 to 1,445,560, because 265,871 still-open loans become usable: a loan still performing well past the window demonstrably did not default inside it. Decision 2 was discarding known outcomes as unknown, so this partly mitigates the Phase 1 survivorship bias rather than only measuring it.

**Known caveat.** Loans sitting delinquent-but-not-yet-charged-off at snapshot are counted as survivals. They are ~1.5% of the file and some will eventually charge off, so the measured rate is very slightly conservative.

---

## 6. Walk-forward validation with an outcome-availability embargo

**Decision.** Evaluate with four walk-forward folds (test years 2014–2017) under three regimes at identical sample sizes: random, temporal, and embargoed. The embargoed regime is the one reported as honest — it trains only on cohorts whose 18-month outcomes were already known at the moment the model would have been fitted, i.e. to predict year Y, train on loans issued before January of Y−2. Implemented in `models/temporal_validation.py`.

**Alternative considered.** A plain time-based split — train on everything before the test year. This is the standard "temporal validation" and a real improvement over random shuffling, but it is still optimistic: in January 2016 you do not know the 18-month outcome of a 2015 loan, because it has not matured. Training on it uses information you could not have had.

**Measured result, and it was not the expected one.** Letting training see the future costs almost nothing: −0.0037 PR-AUC, measured against `random_matched` (same test set, training drawn randomly from all cohorts). The conventional claim that random splitting massively inflates performance does not hold here.

**A fourth regime was added after an audit found the original comparison confounded.** The first version compared plain `random` against the temporal folds, but those differ in *both* the training selection and the test population — plain random tests on a mixed pool of all cohorts, the temporal folds test on one cohort year each. `random_matched` holds the test set fixed so only the training selection varies, and it is the regime the headline gap should be read from.

**That audit also overturned a claim.** The original write-up read the sd difference (0.0020 plain-random vs 0.0191 temporal) as "random splitting erases the variance". It does not: with the test set held fixed, random training shows sd 0.0200, essentially identical to temporal's 0.0191. The variance difference is caused by the *evaluation* scheme, not the training scheme — plain random repeatedly scores the same mixed population, so of course it looks stable. The practical warning survives in corrected form: a practitioner running one random 80/20 reports a single confident number and never discovers that per-cohort performance ranges 0.16 to 0.20. But the cause is evaluating on a pooled test set, not shuffling the training data.

**Embargo cost, isolated properly.** The raw gaps against `temporal` (−0.0344, −0.0282, −0.0097, −0.0071) track how much embargoed training data each fold had, so they cannot be averaged as they stand. Capping both regimes at the embargo pool's size per fold gives **−0.0131 (sd 0.0055) across all four folds, same sign throughout** — roughly 3.5× the cost of temporal splitting alone. An earlier version reported −0.0084 from only the two naturally size-matched folds; matching sizes explicitly lets the number rest on n=4 rather than n=2.

**A second effect the size-matching exposed.** Sample size explains about half the 2014 gap and a third of 2015's, but the early folds still cost 2.1× more than the late ones at equal n (−0.0178 vs −0.0084). The remainder is distributional distance: a 24-month embargo forces the 2014 fold back to pre-2012 loans — a much smaller, earlier population missing the bureau enrichment fields entirely — while the 2016 fold only reaches to 2013. The embargo penalty scales with how fast the population is drifting, which `evaluation/drift.py` measures independently. Confirmed not to be a data-volume artefact: on the 2016 fold the gap is flat from 25k to 150k training rows.

**Reporting the gaps as paired differences.** All three walk-forward regimes share an identical test set within each fold, so their differences are paired and cohort variance cancels. The temporal-vs-random_matched gap is −0.0037 with a **paired sd of 0.0021**, against unpaired fold-to-fold sds of 0.019–0.020 — a tenfold noise reduction, with all four folds agreeing in sign. Quoting differences of means against a spread five times the effect, as the first version did, invited the fair objection that the gap was indistinguishable from zero. Checks live in `models/validation_robustness.py`.

**Companion drift analysis** (`evaluation/drift.py`): 18 of 97 features show significant drift (PSI > 0.25) between the 2013–14 baseline and later cohorts. Fourteen are the vintage-driven columns identified in Phase 1, at PSI ≈ 18 with undefined KS — they moved entirely in availability, not in reported values. Nulls are carried as an explicit PSI bin; without that, field introduction is invisible to the calculation.

---

## 7. Weight-of-Evidence encoding for the interpretable reference model

**Decision.** Encode every feature as Weight of Evidence — the log-odds of non-default within its bin — before fitting the logistic regression, rather than the median-impute-and-one-hot pipeline most people reach for.

**Why.** WoE is a log-odds quantity, so a logistic regression works in units it is already additive in, and a monotone-but-curved relationship becomes linear without hand-crafted splines. More importantly for this dataset, it gives missingness its own bin with its own learned weight. Phase 1 found six `mths_since_*` columns that are missing precisely because the event never happened; median-imputing them rewrites "never delinquent" as "delinquent a middling time ago" and inverts the signal. WoE needs no special-casing to get that right, and the resulting bin/count/rate/weight table is readable by a credit officer, which is the point of keeping an interpretable model at all.

**Alternative considered, and measured.** The naive pipeline was fitted as `logistic_raw` specifically so the choice would be a measurement rather than an appeal to industry convention. Because both models share an identical test set within each fold, the comparison is paired: **+0.0056, −0.0006, −0.0043, +0.0059 — mean +0.0016, paired sd 0.0050, sign flipping.** The predicted damage to the `mths_since_*` columns did not show up as a measurable loss.

**A bug in this encoder, found in the Phase 3 audit and fixed.** `_numeric_bins` cut features into quantiles and, if the resulting edges deduplicated to fewer than three, returned `None` — which routed every non-null value into a single `__rare__` bucket, making the feature a constant. Silently. That branch fires whenever one value holds more than 1/`n_bins` of the mass, which on this data meant **11 of the 58 numeric features observed in the 2016 fold**: `acc_now_delinq` (99.8% at zero), `tax_liens` (99.3%), `pub_rec_bankruptcies` (92.2%), `pub_rec` (91.5%), `num_tl_90g_dpd_24m` (67.1%), `tot_coll_amt` (64.2%) and five more — the delinquency and derogatory-record fields a credit model most wants.

`evaluation/drift.py` had already hit the identical failure computing PSI and carries a discrete-value fallback; this encoder never got it. The fix gives the modal value its own bin and quantile-bins the remainder, which recovers all 11.

**Measured effect of the fix: −0.0005 PR-AUC, sign flipping.** Their information values are 0.0002–0.0011, so they carry almost no signal and returning them to a linear model is marginally harmful. The fix stands anyway: a scorecard that silently discards public records, bankruptcies and tax liens is indefensible to an auditor whatever it scores. Recorded here because the honest version of "I fixed a bug" includes the case where the bug was not costing anything. The three models that do not use WoE reproduced to six decimal places afterwards, confirming nothing else moved.

**So the justification is honesty about what it bought.** WoE is kept for auditability and for correct handling of informative missingness, not for accuracy — on this data those two pipelines are indistinguishable, and the write-up says so rather than claiming a win it cannot demonstrate. The cost is a fitted transform that leaks if fitted outside the fold, so `WOEEncoder.fit` is called on training data only, inside each fold.

---

## 8. Keeping the logistic regression as the reference model, not the gradient booster

**Decision.** Report logistic-WoE as this project's reference model and treat LightGBM as a challenger that has not yet earned the swap.

**Measured result.** Untuned, LightGBM beats logistic-WoE by **+0.0016 PR-AUC with the sign flipping across folds** (−0.0019, +0.0002, +0.0063, +0.0018) — indistinguishable from zero on n=4.

**The obvious objection, tested.** An untuned booster is a weak challenger, so a nested hyperparameter search was run: eight configurations ranked on an inner temporal split of each fold's own training window, the winner refit and scored once on the outer test. Tuning on the reported folds would have biased the estimate by roughly the size of the effect being measured. Tuning gained **+0.0059 (paired sd 0.0038)** over the untuned booster, lifting the gap over logistic-WoE to **+0.0069 (paired sd 0.0058) — still sign-flipping**, negative on the 2014 fold.

**The audit caught the comparison being unfair, and it was fixed.** The first version of this decision tuned LightGBM and left the logistic regression at scikit-learn's default `C=1.0` with default WoE bins — tuning one side of a comparison and not the other. `models/logistic_tuning.py` gives the scorecard the same eight-configuration nested search over `C`, `n_bins` and `min_bin_fraction` (the last matters: at the 2% default, `addr_state` collapses from 50 levels to 18 bins and `purpose` from 14 to 7, an information loss LightGBM does not suffer).

The scorecard gained **+0.0006, paired sd 0.0008, sign flipping** — nothing. It was already at its ceiling, which is itself consistent with the ablation above: the booster needed tuning because it was overfitting, and the scorecard was not. **Tuned against tuned, on equal search budgets, the gap is +0.0068 (paired sd 0.0059) and still flips sign** (−0.0015 on the 2014 fold). The conclusion survives the fair comparison, which is the only version of it worth reporting.

**Stated limitation.** The inner split respects temporal ordering but does not re-apply the 24-month embargo, because embargoing it would leave 1,813 loans for the 2014 fold's selection step. Configurations are therefore ranked under a slightly easier regime than they are scored in; the direction of that bias is unknown. This applies identically to both searches, so it does not bias the comparison between them.

**Where the tuning gain actually came from.** "Tuning helped" explains nothing, so it was decomposed (`models/gbm_ablation.py`). Every selected configuration used `num_leaves=15` against a default of 31 and `min_child_samples` 100–200 against 20; the grid also carried subsampling that no configuration varied, so it was inherited rather than chosen. Separating them: bagging alone is **+0.0015 (sign flips)**, lower capacity plus L2 alone is **+0.0037 (paired sd 0.0020, all four folds)**, together **+0.0056**, interaction +0.0003. The gain is overfitting correction, not the booster discovering structure the linear model missed — which is the expected shape when the 2014 embargoed window holds 39,786 loans with 66 of 87 numeric features entirely null.

**Consequence.** A 1960s scorecard technique matches a modern gradient booster on this problem, on equal search budgets. The booster costs interpretability that a credit regulator would ask for, and buys nothing this evidence can distinguish from noise, so it does not become the reference model. It stays in the repo as the measured alternative.

---

## 9. Doing nothing about class imbalance

**Decision.** Fit on the natural 9.6% default rate. No class weights, no resampling, no synthetic minority generation.

**Why the reflex is wrong here.** PR-AUC, ROC-AUC and KS depend only on the *order* of scores, and rebalancing applies a roughly monotone shift to the score distribution, so it cannot move a ranking metric much by construction. What it does move is calibration — and Phase 4 (calibration) and Phase 5 (cost-optimal thresholds) both need probabilities that mean what they say.

**Measured, four treatments against none, same folds, same model:**

| Treatment | Δ PR-AUC | Paired sd | Same sign | Mean predicted p (true 0.097) | Brier |
|---|---:|---:|---|---:|---:|
| Class weights (`balanced`) | −0.0033 | 0.0041 | No | 0.376 | 0.1787 |
| Random undersampling | −0.0057 | 0.0024 | Yes (4/4) | 0.463 | 0.2409 |
| SMOTE-NC | −0.0015 | 0.0048 | No | 0.091 | 0.0862 |
| SMOTE-NC + integer rounding | −0.0262 | 0.0082 | Yes (4/4) | 0.150 | 0.0979 |
| None | — | — | — | 0.080 | 0.0859 |

Nothing improved ranking. Undersampling was the worst of the four on discrimination and predicted a default probability **4.8× the truth** while nearly tripling Brier; it also discarded about 84% of the training rows.

**SMOTE's null result turned out to be an illusion, which is the more interesting finding.** A classifier can separate SMOTE-NC's synthetic defaults from real ones at **ROC-AUC 1.0000**. Of the 58 numeric features observed in that training window, 51 take only whole-number values in real data — and interpolation makes every one of them fractional (`revol_bal` in 99.7% of synthetic rows, FICO band endpoints in 92%). The booster learns `is_synthetic`, which predicts the positive label perfectly in training and does not exist at scoring time, so it partitions the synthetic half away and trains on the real half. That is why SMOTE fitted on a 50/50 book predicts a test mean of 0.091 while undersampling on an equally balanced book predicts 0.463.

**The obvious repair was tested and made it worse.** Rounding the integer features removes the tell and forces the model to actually learn from the synthetic rows — performance then collapses by **−0.0262 on all four folds**, seventeen times the unrounded effect. SMOTE was not neutral; it was being ignored, and it looked neutral only because of a flaw that also made it detectable.

**What this does not claim.** That SMOTE is useless in general — on a genuinely continuous feature space its interpolation assumption is reasonable. The claim is narrow and measured: on a space built largely of counts, categorical codes, and Phase 1's *missing because the event never happened* columns, the straight line between two minority points is not itself a plausible minority point. Full mechanism in `reports/smote_diagnostic.md`.

**One more cost worth recording.** SMOTE computes distances, so it cannot run on missing values at all: between 29 and 66 of the 87 numeric features had to be imputed first, and those with no observed value anywhere in the training window had to be filled with a flat constant. LightGBM routes missing values down their own branch and WoE gives them their own bin; SMOTE alone forces the data to be invented before it can be resampled.

---

## Open items

- **Survivorship bias is partly mitigated** by the fixed-window target (decision 5), which recovers 265,871 previously-discarded loans. No reweighting or inverse-probability correction has been attempted on top of that, and cohorts after early 2017 remain excluded for lack of maturity.
- **Free-text and high-cardinality columns are on the allow-list but unused.** `emp_title`, `desc`, `title`, raw `zip_code`.
- **Loans delinquent but not yet charged off at snapshot count as survivals** under decision 5. About 1.5% of the file; some will eventually charge off, so the measured default rate is slightly conservative.
- **No calibration or unit tests yet.** Calibration is Phase 4, and the imbalance results in decision 9 are the reason that ordering matters: three of four treatments wreck the probability scale while leaving ranking untouched.
- **Every paired comparison rests on n=4.** Enough to establish a consistent-sign effect of ~0.006, not enough to resolve differences below ~0.002. Effects that flip sign are reported as null rather than as small.
- **The 18-month window is a parameter, not a finding.** `WINDOW_MONTHS` was chosen from the time-to-default distribution, but no sensitivity analysis across 12 / 18 / 24 has been run.
