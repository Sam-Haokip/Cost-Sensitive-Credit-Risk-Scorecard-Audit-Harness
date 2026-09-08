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

**Measured cost of getting this wrong.** Same model, same split, trained on all available columns versus the allow-list: PR-AUC 0.9973 vs 0.4064. A +0.59 gap. A near-perfect credit risk model is a symptom, not an achievement — the naive version is reading the outcome, not predicting it.

---

## 4. Excluding Lending Club's own risk grade

**Decision.** Exclude `grade`, `sub_grade`, `int_rate` and `installment` from the model.

**The tension.** These are available before the loan is funded, so they are not leakage in the "happened afterwards" sense that decision 3 covers. But they are *outputs of Lending Club's own underwriting model*, not raw applicant characteristics: `grade` is computed from the applicant's risk profile, and `int_rate` (hence `installment`) follows from the grade. Training on them risks a model that re-derives Lending Club's decision rather than learning the underlying default relationship — strong reported performance that wouldn't transfer to a setting where their grade doesn't exist.

**Alternative considered.** Keep them and add policy vintage as a control. This is what most public work does, and it is not unreasonable — they genuinely are known pre-funding.

**Why measured rather than argued.** Both positions are defensible from reasoning alone, so reasoning alone shouldn't decide it.

**Measured cost — a negative result.** Adding all four moves PR-AUC by **+0.0037 on average across five seeds** (sd 0.0017, range +0.0014 to +0.0056; `models/seed_stability.py`). The direction is consistent on every seed, so the effect is genuine rather than noise — but it is smaller than the seed-to-seed variation in the clean model's own PR-AUC (sd 0.0042), which is the honest basis for calling it negligible rather than simply quoting one split. The independently-derived credit-bureau features already capture nearly everything the grade encodes, which makes sense: the same underlying data feeds both. Excluding them costs essentially nothing, so the more conservative choice is also the cheap one. The caution turned out to be almost free — worth knowing, and worth reporting even though it makes the original concern look overstated.

---

## Open items

- **Temporal validation is not yet built.** Every number above comes from a random 80/20 split, which on temporally ordered data is inflated. The 0.4064 figure should be expected to fall once time-based splits land, and that gap is itself a planned finding.
- **Survivorship bias is measured, not mitigated.** No reweighting or inverse-probability correction has been attempted.
- **Free-text and high-cardinality columns are on the allow-list but unused.** `emp_title`, `desc`, `title`, raw `zip_code`.
