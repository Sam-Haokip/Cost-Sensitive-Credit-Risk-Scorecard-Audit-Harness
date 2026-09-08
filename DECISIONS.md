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

## 5. Replacing the target with a fixed 18-month performance window

**Decision.** Supersede decision 2. A loan is labelled `1` if it charged off having stopped paying within **18 months of origination**, `0` if it survived that window (repaid, still performing, or charged off only afterwards). A loan is eligible only if observed for 18 + 6 months, the extra six covering Lending Club's ~120-150 day delinquency-to-charge-off lag. Implemented in `data/vintage_target.py`.

**What forced the revision.** Decision 2's "final status" target makes cohorts non-comparable, and Phase 2 is built on comparing cohorts. Defaults resolve fast and full repayments resolve slowly, so a partly-resolved cohort is enriched with whichever outcome finishes sooner. Measured that way, the 2016 and 2017 36-month cohorts show ~20% default against 12-15% for every fully-resolved year before them — and the 2018 cohort drops back to 13%, because at under twelve months of age the pool is dominated by early full repayments instead. The bias is not merely present, it reverses direction with cohort age. A naive time-based split would have reported all of that as credit drift.

**Alternatives considered.** (a) Restrict to fully-resolved cohorts only — clean, but discards 2016 onward and leaves too few cohorts for a walk-forward backtest. (b) Proceed naively and document the bias — what most public work does, and now demonstrably wrong.

**Why 18 months, measured rather than assumed.** Time from origination to final payment on charged-off 36-month loans (2012-2014 cohorts, fully resolved so the timing itself isn't censored) has a median of 17 months. Cumulative share of eventual charge-offs caught: 34% by month 12, **56% by month 18**, 76% by 24, 92% by 30. Longer windows capture more defaults but cost usable cohorts, since every loan needs window + lag months of observation. 18 captures a majority while preserving 2012 to early-2017 for folds. Set by `WINDOW_MONTHS`; changing it is one line.

**Measured effect.** Cohort default rate for 36-month loans goes from a 12.3%-20.0% swing under decision 2 to 6.9%-9.8% under the window — a gentle rise peaking in 2016 and easing in 2017, consistent with a real credit cycle rather than an artefact. 60-month loans sit consistently above 36-month at every vintage (13.3% vs 9.8% in 2016), a sanity check that real risk is still being captured.

**Cost, and an unexpected gain.** The target now means "defaulted early" rather than "ever defaulted", so it answers a narrower question and the base rate falls from 19.97% to 9.63%. But the eligible population *grows*, from 1,345,350 to 1,445,560, because 265,871 still-open loans become usable: a loan still performing well past the window demonstrably did not default inside it. Decision 2 was discarding known outcomes as unknown, so this partly mitigates the Phase 1 survivorship bias rather than only measuring it.

**Known caveat.** Loans sitting delinquent-but-not-yet-charged-off at snapshot are counted as survivals. They are ~1.5% of the file and some will eventually charge off, so the measured rate is very slightly conservative.

---

## Open items

- **Temporal validation is not yet built.** Every number above comes from a random 80/20 split, which on temporally ordered data is inflated. The 0.4064 figure should be expected to fall once time-based splits land, and that gap is itself a planned finding.
- **Survivorship bias is partly mitigated** by the fixed-window target (decision 5), which recovers 265,871 previously-discarded loans. No reweighting or inverse-probability correction has been attempted on top of that.
- **Free-text and high-cardinality columns are on the allow-list but unused.** `emp_title`, `desc`, `title`, raw `zip_code`.
