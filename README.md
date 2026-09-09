# Credit Default Risk: A Decision System, Not Just a Score

Predicting default on 2.26M Lending Club loans (2007–2018) — built as a decision system with a governance layer (cost-optimal thresholds, calibrated probabilities, a fairness audit, temporal validation, a model card) rather than a model that stops at reporting an AUC.

> **Status: in progress.** Phases 1–3 of 8 complete (data integrity, leakage audit, temporal validation, baselines and imbalance handling). Calibration, cost-sensitive decisioning and the fairness audit are not yet built. Every number below is reproducible from the committed code. See [Project status](#project-status).

---

## The headline: this dataset will hand you a 0.997 PR-AUC if you let it

The same model, same data, same split — trained twice. Once on every column the file offers, once on only the columns a lender would actually possess at the moment of the credit decision:

| Feature set | Features | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|
| Naive — every available column | 139 | **0.9972** | 0.9996 | 0.0034 |
| Disciplined — pre-decision columns only | 97 | **0.1944** | 0.6971 | 0.0830 |

A near-perfect credit risk model is not an achievement, it's a symptom. The naive version scores 0.997 because 38 of its columns — `recoveries`, `total_pymnt`, `last_fico_range_high`, the hardship-programme fields, the debt-settlement fields — only get populated *after* the loan is funded, and several only exist for loans that already went bad. The model isn't predicting the outcome; it's reading it.

**0.19 is what this problem actually looks like** — and even that is generous. (PR-AUC against a 9.63% base rate; ROC-AUC 0.70.) That figure comes from a random split, which is the methodology the rest of this README argues against. Under the honest regime Phase 2 establishes — walk-forward folds trained only on cohorts whose outcomes had matured — the best configuration reaches **0.167**. The number above isolates the leakage effect by holding everything else constant; [finding 7](#key-findings) carries the one you should judge the models by.

## Key findings

**1. The leakage gap is large enough to be self-diagnosing.** +0.80 PR-AUC between the naive and disciplined feature sets. Any credit model reporting near-perfect discrimination should be assumed broken until proven otherwise. This is the single most common silent failure in public work on this dataset.

**2. Random splitting barely inflates the score. What it hides is per-cohort variability — and the mechanism is the evaluation scheme, not the training scheme.** Four regimes, identical sample sizes:

| Split regime | PR-AUC | sd across folds | Tests on |
|---|---:|---:|---|
| Random (plain 80/20) | 0.1869 | **0.0020** | mixed pool, all cohorts |
| Random, test set matched | 0.1862 | **0.0200** | one cohort year |
| Temporal (train on earlier cohorts only) | 0.1825 | **0.0191** | one cohort year |
| Embargoed (only matured outcomes) | 0.1626 | **0.0279** | one cohort year |

Letting training see the future costs almost nothing: **−0.0037** with the test set held fixed. Because the regimes share an identical test set within each fold, that difference is *paired* — cohort variance cancels, and the noise drops roughly tenfold: **paired sd 0.0021** against unpaired fold-to-fold sds of 0.019–0.020, with all four folds agreeing in sign (−0.0040, −0.0043, −0.0058, −0.0007). Quoting a difference of means against a spread five times the effect would have invited the fair question of whether it differs from zero at all; paired, it clearly does.

An earlier version of this analysis reported the sd difference (0.0020 vs 0.0191) as evidence that "random splitting erases the variance". That attributed to the training scheme what is actually caused by the evaluation scheme: hold the test set fixed and random training shows sd 0.0200, essentially identical to temporal's 0.0191. The practical warning survives in corrected form — a practitioner running one random 80/20 reports a single confident number and never learns that per-cohort performance ranges **0.16 to 0.20** — but the cause is evaluating on a mixed pool, not shuffling the training data.

**3. The honest constraint costs far more than the split type — and it costs most where the data drifts fastest.** The *embargoed* regime trains only on cohorts whose 18-month outcomes were already known when the model would have been fitted — to predict year Y, train on loans issued before January of Y−2. That's the constraint a real underwriting deployment faces, and almost nobody implements it.

Isolating it properly required matching training sizes, because the raw gaps (−0.0344, −0.0282, −0.0097, −0.0071) track how much embargoed data each fold had. Capping both regimes at the embargo pool's size, all four folds:

| Fold | n (both) | Raw gap | Size-matched gap |
|---|---:|---:|---:|
| 2014 | 39,786 | −0.0344 | **−0.0172** |
| 2015 | 93,153 | −0.0282 | **−0.0184** |
| 2016 | 150,000 | −0.0097 | −0.0097 |
| 2017 | 150,000 | −0.0071 | −0.0071 |

**Embargo penalty: −0.0131 (sd 0.0055), all four folds, same sign** — about 3.5× the cost of temporal splitting alone. Sample size explained roughly half the 2014 gap and a third of 2015's, but not the rest: even at equal n, the early folds cost **2.1× more** than the late ones (−0.0178 vs −0.0084). A 24-month embargo forces the 2014 fold to train on pre-2012 loans — a very different population, missing the bureau fields entirely — while 2016 only reaches back to 2013. **The embargo hurts most exactly where the drift analysis says the population moved most.**

Confirmed not to be a data-volume artifact: on the 2016 fold the gap is flat across training sizes (−0.0094 at 25k, −0.0097 at 150k). Reproduce with `python -m models.validation_robustness`.

**4. Cohorts weren't comparable, and fixing it grew the dataset.** Defaults resolve fast; repayments resolve slowly. So a partly-matured cohort is enriched with whichever outcome finishes sooner — the 2016 and 2017 cohorts showed ~20% default against 12–15% for fully-resolved years, then 2018 fell *back* to 13%. The bias reverses direction with cohort age. Replacing "did it ever default" with "did it default within 18 months" flattens that to 6.9–9.8%, a shape consistent with a real credit cycle. It also *added* 100,000 loans, because a 2016 loan still performing in 2019 demonstrably did not default within 18 months — a known outcome the old target was discarding as unknown.

**5. Revised: Lending Club's own risk grade is not free to exclude after all.** `grade`, `sub_grade`, `int_rate` and `installment` are outputs of Lending Club's underwriting model rather than raw applicant characteristics, so training on them risks re-deriving their decision instead of learning the underlying risk. Measured across five seeds, adding them back is worth **+0.0110 PR-AUC (sd 0.0026), positive on every seed** — against a seed-to-seed spread in the baseline of only 0.0020. That is five times the measurement noise, roughly a 6% relative gain. An earlier version of this project measured +0.0037 on the retired target and called the exclusion costless; on the current target that claim is wrong. The columns are still excluded, on the circularity argument — but the decision now carries a real, stated price rather than a free one.

**6. Missingness encodes loan vintage, and the mechanism differs by column.** Fourteen bureau-enrichment columns are **100% missing before 2016 and ~0% after**, because Lending Club introduced the fields partway through. Six `mths_since_*` columns are missing because *the event never happened to that borrower* — median-imputing those would replace "never delinquent" with "delinquent a while ago" and invert the signal. Same headline percentage, opposite handling.

| Null rate (%) by issue year | 2013 | 2014 | 2015 | 2016 | 2017 |
|---|---:|---:|---:|---:|---:|
| `open_acc_6m` (field introduced) | 100 | 100 | 95 | **0** | **0** |
| `mths_since_last_delinq` (event never occurred) | 57 | 49 | 48 | 47 | 50 |

That Phase 1 finding predicted exactly which features would break temporal validation. The drift analysis confirms it independently: those same fourteen columns register **PSI ≈ 18** against a "significant" threshold of 0.25, with *undefined* KS — meaning they moved entirely in availability, not in the values they report. Full taxonomy in [`reports/data_quality.md`](reports/data_quality.md), full drift table in [`reports/drift.md`](reports/drift.md).

**7. Gradient boosting does not beat a logistic regression here, and tuning does not rescue it.** Four models on the embargoed folds:

| Model | PR-AUC | sd | ROC-AUC | KS | Brier |
|---|---:|---:|---:|---:|---:|
| Trivial (majority class) | 0.0974 | 0.0086 | 0.5000 | 0.0000 | 0.0882 |
| Logistic, median-impute + one-hot | 0.1634 | 0.0298 | 0.6577 | 0.2306 | 0.0864 |
| Logistic, Weight-of-Evidence | 0.1650 | 0.0259 | 0.6583 | 0.2294 | 0.0857 |
| LightGBM, untuned | 0.1666 | 0.0287 | 0.6598 | 0.2324 | 0.0859 |

Untuned, LightGBM over logistic-WoE is **+0.0016 with the sign flipping across folds** — indistinguishable from zero.

**Both models were then given the same eight-configuration nested search**, ranked on an inner temporal split of each fold's own training window so the reported folds never influence selection. Tuning one side and not the other would have made the comparison worthless:

| | Gain from tuning | Paired sd | Same sign |
|---|---:|---:|---|
| LightGBM | **+0.0059** | 0.0038 | Yes (4/4) |
| Logistic-WoE | +0.0006 | 0.0008 | No |

The booster needs tuning; the scorecard was already at its ceiling. **Tuned against tuned, the gap is +0.0068 (paired sd 0.0059), and it still flips sign** — negative on the 2014 fold. Four folds cannot establish an effect that small.

The honest summary: **on this problem, a 1960s-vintage scorecard technique matches a modern gradient booster**, and the booster costs interpretability a credit regulator would ask for. The trivial model makes the companion point — it scores **90.3% accuracy** while catching zero defaults, which is why accuracy appears nowhere else in this README.

**8. The tuning gain is overfitting correction, not capacity.** "Tuning helped" explains nothing, so the gain was decomposed. Every configuration the search selected used `num_leaves=15` against a default of 31 and `min_child_samples` of 100–200 against 20; the grid also carried row and column subsampling that no configuration varied. Separating them:

| Variant | vs untuned baseline | Paired sd | Same sign |
|---|---:|---:|---|
| Bagging only (`subsample`/`colsample` 0.8) | +0.0015 | 0.0020 | No |
| Lower capacity + L2 only | **+0.0037** | 0.0020 | **Yes (4/4)** |
| Both (what the search actually fitted) | +0.0056 | 0.0038 | Yes (4/4) |

Two-thirds of the gain is capacity reduction and explicit L2; bagging alone does not clear its own noise. The effects are additive (interaction +0.0003). That is the expected shape when the early training windows hold ~40,000 loans and, per Phase 1, **66 of 87 numeric features are entirely null inside the 2014 embargoed window** — the default 31-leaf tree simply has more capacity than the data supports. Reproduce with `python -m models.gbm_ablation`.

**9. Every imbalance treatment made things worse, and SMOTE's null result was an illusion.** Four treatments against doing nothing, same folds, same model:

| Treatment | Δ PR-AUC | Paired sd | Same sign | Mean predicted p | Brier |
|---|---:|---:|---|---:|---:|
| Class weights (`balanced`) | −0.0033 | 0.0041 | No | 0.376 (3.9× true) | 0.1787 |
| Random undersampling | **−0.0057** | 0.0024 | **Yes (4/4)** | 0.463 (4.8× true) | 0.2409 |
| SMOTE-NC | −0.0015 | 0.0048 | No | 0.091 | 0.0862 |
| SMOTE-NC + integer rounding | **−0.0262** | 0.0082 | **Yes (4/4)** | 0.150 | 0.0979 |
| *(no treatment)* | — | — | — | 0.080 (true rate 0.097) | 0.0859 |

Nothing improved ranking, which is expected: PR-AUC, ROC-AUC and KS depend only on score *order*, and rebalancing applies a roughly monotone shift. What rebalancing does destroy is calibration — undersampling predicts a default probability **4.8× the truth** and nearly triples Brier — and Phases 4 and 5 need probabilities that mean what they say. Undersampling also discards ~84% of the training rows.

The interesting case is SMOTE. Its apparent harmlessness is an artefact: **a classifier can separate SMOTE's synthetic defaults from real ones at ROC-AUC 1.0000.** **51 of the 58 numeric features observed in that training window take only whole-number values in real data, and interpolation makes every one of them fractional** (`revol_bal` in 99.7% of synthetic rows, FICO band endpoints in 92%) — a borrower with 7.43 open accounts does not exist. The booster learns `is_synthetic`, which predicts the positive label perfectly in training and does not exist at scoring time, so it partitions the synthetic half away and trains on the real half. That is why a model fitted on a 50/50 book still predicts a test mean of 0.091 while undersampling on an equally balanced book predicts 0.463.

Removing the tell confirms it. Rounding the integer features so synthetic rows are no longer trivially detectable forces the model to actually learn from them — and performance collapses by **−0.0262 on all four folds**, seventeen times the unrounded effect. SMOTE was not neutral; it was being ignored, and it only looked neutral because of a flaw that also made it detectable. Mechanism and both tests in [`reports/smote_diagnostic.md`](reports/smote_diagnostic.md).

**10. Found in audit: the WoE encoder was silently discarding 11 of the scorecard's 58 usable features.** Quantile binning fails without error on a column where one value holds more than 1/`n_bins` of the mass: every quantile lands on that value, the bin edges deduplicate to fewer than three, and the encoder falls through to a branch that maps every non-null value into a single bucket. The feature becomes a constant.

On this data that is not an edge case. In the 2016 fold it destroyed exactly the fields a credit model most wants:

| Feature | Share at the modal value | Fate before the fix |
|---|---:|---|
| `acc_now_delinq` | 99.8% | one bin |
| `tax_liens` | 99.3% | one bin |
| `pub_rec_bankruptcies` | 92.2% | one bin |
| `pub_rec` | 91.5% | one bin |
| `num_tl_90g_dpd_24m` | 67.1% | one bin |
| `tot_coll_amt` | 64.2% | one bin |

`evaluation/drift.py` had already hit the identical failure computing PSI and carried a fix; `features/woe.py` never got it. Two modules in the same repo, one of which had learned the lesson.

**The fix, and the honest result:** giving the modal value its own bin and quantile-binning the remainder recovers all 11 features — and changes PR-AUC by **−0.0005, with the sign flipping**. Their information values are 0.0002–0.0011, so they carried essentially no signal and adding them back to a linear model is marginally *harmful*. The bug was real and worth fixing on the merits — a scorecard that silently drops public records and tax liens is indefensible to an auditor regardless of what it costs in PR-AUC — but the headline did not depend on it. The three models that do not use WoE reproduced to six decimal places, confirming nothing else moved.


## Target definition

`default_window = 1` if the loan charged off having stopped paying within **18 months of origination**; `0` if it survived that window. A loan is eligible only once observed for 18 + 6 months, the extra six covering the ~120–150 day delinquency-to-charge-off lag.

Window length chosen from the data, not convention: median time from origination to final payment on charged-off 36-month loans is 17 months, and 18 months captures 56% of eventual charge-offs versus 34% at twelve.

**Population: 1,445,560 loans, 9.63% default rate.** The target means "defaulted early", not "ever defaulted" — a narrower question, but one measured identically across every cohort. Superseded the original final-status target; see [`DECISIONS.md`](DECISIONS.md) decision 5.

## Validation design

Four walk-forward folds (test years 2014–2017), four regimes at identical sample sizes. The `random_matched` regime exists specifically so the headline gap is read off a comparison where only the training selection varies; plain `random` is reported for reference as the practice being critiqued, but its test population differs and it should not be used for the gap. Metrics reported as a distribution across folds, never a single number. Results in [`reports/temporal_validation.csv`](reports/temporal_validation.csv).

Covariate drift between the 2013–14 baseline and later cohorts: **18 of 97 features show significant drift (PSI > 0.25), 12 moderate, 67 stable.** Nulls are carried as an explicit PSI bin — without that, field introduction is invisible, and it is the largest distributional shift in the dataset. The one drifter that isn't a missingness artifact is `initial_list_status` (PSI 0.50): whole-loan listings went from 7% of originations in 2012 to 77% in 2016, a genuine platform change.

## The leakage audit

All 151 raw columns classified individually with a stated reason each — an allow-list, not a drop-list. Full detail in [`data/feature_audit.py`](data/feature_audit.py); reasoning for this and every other modelling decision in [`DECISIONS.md`](DECISIONS.md).

| Decision | Count | What it covers |
|---|---:|---|
| `keep` | 101 | Application data and the credit-bureau report pulled for it |
| `drop_leakage` | 38 | Payment history, recoveries, post-origination FICO pulls, hardship (15) and debt-settlement (7) programmes |
| `drop_other` | 8 | Identifiers, constants, redundant columns, and the label source |
| `review` | 4 | `grade` / `sub_grade` / `int_rate` / `installment` — see finding 5 |

## Limitations

Stated plainly, because they bound what the current numbers mean:

- **Discrimination is modest in absolute terms.** The best honest configuration reaches PR-AUC ~0.17 against a 0.097 base rate, ROC-AUC ~0.66. That is what this feature set supports once leakage and the embargo are respected; published numbers far above it on this dataset are almost always one of the two failures Phase 1 and Phase 2 measure.
- **No calibration yet.** Every model here is scored on ranking and raw Brier; reliability curves and a proper calibration comparison are Phase 4, and the imbalance results above show why that ordering matters.
- **`logistic_raw` silently drops all-null training features.** scikit-learn's `SimpleImputer` skips columns with no observed value, so in the 2014 fold it discards 66 of 87 numeric features rather than erroring. Those columns carry no information in that window and neither WoE nor LightGBM can use them either, so the comparison stays fair — but the pipeline reports it as a warning, not a failure, which is worth knowing before trusting any imputer on vintage-partitioned data.
- **Four folds is few.** Every paired comparison rests on n=4, which is enough to establish a consistent-sign effect of ~0.006 but not to resolve differences below ~0.002. Effects that flip sign are reported as null rather than as small.
- **Survivorship bias is partly mitigated, not eliminated.** The fixed window recovers 265,871 previously-discarded loans, but cohorts after early 2017 are still excluded for lack of maturity.
- **Loans delinquent-but-not-yet-charged-off at snapshot count as survivals.** ~1.5% of the file; some will eventually charge off, so the measured rate is slightly conservative.
- **Free-text and high-cardinality columns are unused.** `emp_title`, `desc`, `title` and raw `zip_code` are on the allow-list but not yet engineered into features.
- **Proxy fairness only.** The dataset contains no direct protected attributes; the planned audit uses geography and income as imperfect proxies, which bounds the conclusions it can support.
- **No unit tests yet.**

## Repo layout

```
data/        loading, target definition, per-column leakage audit, data quality
features/    Weight-of-Evidence encoding
models/      training and experiments
evaluation/  drift; calibration and cost curves to follow (Phases 4-5)
fairness/    group metrics and mitigation     (Phase 6)
reports/     generated analysis output
```

## Reproducing

```bash
# Python 3.10+
pip install -r requirements.txt

# Download accepted_2007_to_2018Q4.csv from the Kaggle dataset
# "wordsforthewise/lending-club" and place it in data/raw/
python -m data.convert_raw          # CSV -> 46 chunked Parquet files (~2 min)
python -m data.quality_report       # missingness mechanisms, cardinality, coverage
python -m data.vintage_target       # target definition + cohort comparability
python -m models.leakage_experiment # the three-way leakage comparison
python -m models.seed_stability     # is the grade/int_rate effect above noise?
python -m models.temporal_validation# walk-forward folds, three split regimes
python -m evaluation.drift          # PSI / KS drift table
python -m models.validation_robustness # paired folds + embargo robustness checks
python -m models.baselines          # trivial / logistic (WoE, naive) / LightGBM
python -m models.tuning             # nested hyperparameter search (LightGBM)
python -m models.logistic_tuning    # the same search budget for the scorecard
python -m models.gbm_ablation       # which knob produced the tuning gain
python -m models.imbalance          # none / weights / undersample / SMOTE
python -m models.smote_diagnostic   # why SMOTE has no effect here
```

Runs are seeded (`random_state=42`) and dependencies pinned. The raw CSV is ~1.6GB and the Parquet output ~400MB; neither is committed. `LC_RAW_CSV` and `LC_PARQUET_DIR` override the default data locations.

## Project status

| Phase | Status |
|---|---|
| 1. Data integrity & leakage audit | Complete |
| 2. Temporal validation design | Complete |
| 3. Baselines & imbalance handling | Complete |
| 4. Calibration | Next |
| 5. Cost-sensitive decisioning | Not started |
| 6. Fairness & bias audit | Not started |
| 7. Explainability | Not started |
| 8. Model card & packaging | Not started |

## Data

[Lending Club accepted loans, 2007–2018](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (Kaggle) — 2,260,701 loans, 151 columns, origination dates spanning June 2007 to December 2018 with no missing months.
