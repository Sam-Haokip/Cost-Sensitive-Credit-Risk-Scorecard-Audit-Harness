# Credit Default Risk: A Decision System, Not Just a Score

Predicting default on 2.26M Lending Club loans (2007–2018) — built as a decision system with a governance layer (cost-optimal thresholds, calibrated probabilities, a fairness audit, temporal validation, a model card) rather than a model that stops at reporting an AUC.

> **Status: in progress.** Phase 1 of 8 complete (data integrity + leakage audit). Temporal validation, calibration, cost-sensitive decisioning, and the fairness audit are not yet built. Every number below is reproducible from the committed code. See [Project status](#project-status).

---

## The headline: this dataset will hand you a 0.997 PR-AUC if you let it

The same model, same data, same split — trained twice. Once on every column the file offers, once on only the columns a lender would actually possess at the moment of the credit decision:

| Feature set | Features | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|
| Naive — every available column | 139 | **0.9973** | 0.9996 | 0.0011 |
| Disciplined — pre-decision columns only | 97 | **0.4064** | 0.7300 | 0.1415 |

A near-perfect credit risk model is not an achievement, it's a symptom. The naive version scores 0.997 because 38 of its columns — `recoveries`, `total_pymnt`, `last_fico_range_high`, the hardship-programme fields, the debt-settlement fields — only get populated *after* the loan is funded, and several only exist for loans that already went bad. The model isn't predicting the outcome; it's reading it.

0.41 is what this problem actually looks like.

## Key findings

**1. The leakage gap is large enough to be self-diagnosing.** +0.59 PR-AUC between the naive and disciplined feature sets. Any credit model reporting near-perfect discrimination should be assumed broken until proven otherwise. This is the single most common silent failure in public work on this dataset.

**2. Negative result: Lending Club's own risk grade adds essentially nothing.** `grade`, `sub_grade`, `int_rate` and `installment` are available before funding, so they aren't leakage in the usual sense — but they're outputs of Lending Club's underwriting model, not raw applicant characteristics. Training on them risks re-deriving their decision rather than learning the underlying risk. Testing rather than assuming, and across five seeds rather than one: the gain is **+0.0037 mean (sd 0.0017, range +0.0014 to +0.0056)** — consistently positive, so the effect is real, but *smaller than the seed-to-seed spread in the headline PR-AUC itself* (sd 0.0042). The independently-derived credit-bureau features already capture nearly everything the grade encodes. Excluding them is therefore close to free, and the caution costs nothing to act on. Reproduce with `python -m models.seed_stability`.

**3. 40.5% of the dataset cannot be used, and the loss is not random.** Only loans with a resolved outcome can carry a label. Excluding the 913k still-active loans skews the training population toward older cohorts — by 2018, 88% of 36-month loans and 90% of 60-month loans had not yet resolved:

| Issue year | Term | Loans | Labelled | Still active |
|---|---|---:|---:|---:|
| 2014 | 60 mo | 73,059 | 82.9% | 17.1% |
| 2016 | 36 mo | 323,495 | 71.8% | 28.2% |
| 2016 | 60 mo | 110,912 | 54.8% | 45.2% |
| 2017 | 36 mo | 320,419 | 40.1% | 59.9% |
| 2018 | 36 mo | 344,671 | 12.0% | **88.0%** |
| 2018 | 60 mo | 150,571 | 10.0% | **90.0%** |

This is measured, not corrected — the model is trained on a population systematically older than the one it would score in production, and that limitation is stated rather than hidden.

**4. Missingness encodes loan vintage, not just absence — and the mechanism differs by column.** Fourteen bureau-enrichment columns (`open_acc_6m`, `il_util`, `all_util`, …) are **100% missing before 2016 and ~0% after**, because Lending Club introduced the fields partway through. A model can read origination era straight off that pattern, which is a live hazard for temporal validation. Meanwhile six `mths_since_*` columns are missing because *the event never happened to that borrower* — median-imputing them would replace "never delinquent" with "delinquent a while ago" and invert the signal. Same headline percentage, opposite handling. Full taxonomy, with per-year evidence, in [`reports/data_quality.md`](reports/data_quality.md).

| Null rate (%) by issue year | 2013 | 2014 | 2015 | 2016 | 2017 |
|---|---:|---:|---:|---:|---:|
| `open_acc_6m` (field introduced) | 100 | 100 | 95 | **0** | **0** |
| `mths_since_last_delinq` (event never occurred) | 57 | 49 | 48 | 47 | 50 |

## Target definition

`default = 1` for Charged Off / Default, `0` for Fully Paid. Everything else is excluded from the modelling population entirely:

- **Non-terminal statuses** (Current, In Grace Period, Late 16-30, Late 31-120 — 912,569 loans). No outcome exists yet. Labelling them "safe" would inject an error correlated with *how recently the loan was issued*, since recent loans are overwhelmingly still Current simply because time hasn't passed — which would corrupt the temporal validation the project depends on.
- **"Does not meet the credit policy"** statuses (2,749 loans). Outcome known, but underwritten under rules since retired.
- **Null status** (33 rows). Corrupted footer rows, not loans.

Resulting population: **1,345,350 loans, 19.97% default rate.**

## The leakage audit

All 151 raw columns classified individually with a stated reason each — an allow-list, not a drop-list. Any column that cannot be justified as available pre-decision is cut. Full detail in [`data/feature_audit.py`](data/feature_audit.py); the reasoning behind this and every other modelling decision is in [`DECISIONS.md`](DECISIONS.md).

| Decision | Count | What it covers |
|---|---:|---|
| `keep` | 101 | Application data and the credit-bureau report pulled for it |
| `drop_leakage` | 38 | Payment history, recoveries, post-origination FICO pulls, hardship (15) and debt-settlement (7) programmes |
| `drop_other` | 8 | Identifiers, constants, redundant columns, and the label source |
| `review` | 4 | `grade` / `sub_grade` / `int_rate` / `installment` — see finding 2 |

## Limitations

Stated plainly, because they bound what the current numbers mean:

- **Validation is a random split.** On temporally ordered financial data this inflates performance — a random shuffle lets the model learn from 2018 loans to predict 2015 ones. Time-based splits and a walk-forward backtest are the next phase; the 0.406 figure should be expected to fall.
- **Survivorship bias is measured, not mitigated.** See finding 3.
- **The model is deliberately quick.** One gradient-boosting configuration, no hyperparameter search, no calibration, 400k-row subsample. It exists to size the leakage effect, not to be a good model.
- **Headline PR-AUC is a single-split point estimate.** Across five seeds the clean model ranges 0.3973–0.4064 (sd 0.0042), so treat 0.406 as approximate. Phase 2's walk-forward folds will replace it with a proper distribution.
- **Free-text and high-cardinality columns are unused.** `emp_title`, `desc`, `title` and raw `zip_code` are on the allow-list but not yet engineered into features.
- **Proxy fairness only.** The dataset contains no direct protected attributes; the planned audit uses geography and income as imperfect proxies, which is a real limitation of the conclusions it can support.

## Repo layout

```
data/        loading, target definition, per-column leakage audit
features/    feature engineering            (Phase 3)
models/      training and experiments
evaluation/  metrics, calibration, cost curves (Phases 3–5)
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
python -m data.target               # target definition + survivorship table
python -m models.leakage_experiment # the three-way comparison above
python -m models.seed_stability     # is the grade/int_rate effect above noise?
```

Runs are seeded (`random_state=42`) and dependencies pinned in `requirements.txt`. The raw CSV is ~1.6GB and the Parquet output ~400MB; neither is committed.

## Project status

| Phase | Status |
|---|---|
| 1. Data integrity & leakage audit | Complete |
| 2. Temporal validation design | Next |
| 3. Baselines & imbalance handling | Not started |
| 4. Calibration | Not started |
| 5. Cost-sensitive decisioning | Not started |
| 6. Fairness & bias audit | Not started |
| 7. Explainability | Not started |
| 8. Model card & packaging | Not started |

## Data

[Lending Club accepted loans, 2007–2018](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (Kaggle) — 2,260,701 loans, 151 columns, origination dates spanning June 2007 to December 2018 with no missing months.
