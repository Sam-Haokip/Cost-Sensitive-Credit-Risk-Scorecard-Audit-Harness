# Model card: Cost-Sensitive Credit Risk Scorecard

Format follows Mitchell et al., 2019 ("Model Cards for Model Reporting"). Every number here is reproducible from the code and reports committed in this repository — see [`README.md`](README.md) for the full findings and [`DECISIONS.md`](DECISIONS.md) for the reasoning behind each modelling choice. This card describes the **logistic-WoE scorecard**, the model this project recommends for deployment (see "Model selection" below) — not the LightGBM model, which was built and evaluated throughout as a comparison point.

## Model details

- **Developed by**: Samuel Haokip, as a portfolio project (not a production system; see "Out-of-scope uses").
- **Model date**: 2026-09-14 (Phase 7, most recent retraining).
- **Model type**: Logistic regression on Weight-of-Evidence (WoE)-encoded features (`features/woe.py`, `evaluation/calibration.py::fit_logistic_woe`). WoE encoding bins each raw feature and replaces it with the log-odds of default in that bin, fit only on the training cohort.
- **Calibration**: Platt scaling, fit on a calibration cohort adjacent to (but excluded from) the training window, refit per fold (decision 10).
- **License / repository**: see the repository root. Trained on public data; see "Training data."

## Intended use

- **Primary intended use**: a portfolio/demonstration piece showing an end-to-end credit-risk decision system — leakage-aware feature selection, temporally-honest validation, cost-sensitive thresholding, a fairness audit, and explainability — built for review by ML/data-science hiring evaluators, not for production lending decisions.
- **Primary intended users**: technical reviewers (recruiters, interviewers, engineers) assessing the author's ML engineering practice.
- **Out-of-scope uses**:
  - **Real lending decisions on real applicants.** The model was trained on 2007–2018 Lending Club data; its discrimination is modest (PR-AUC ~0.165 against a 9.6% base rate, ROC-AUC ~0.66 — see "Metrics"), it has not been validated on any population outside Lending Club's unsecured-consumer-loan book, and the fairness audit (below) found unresolved gaps and an unexplained interaction between the model's geography feature and fairness-relevant behavior. None of this clears the bar for a real underwriting decision.
  - **Any use where the demographic proxies used in the fairness audit** (`income_quintile`, `emp_length`, `geo_race_proxy`) **are treated as actual protected-class labels.** They are stated, imperfect stand-ins — the dataset contains no race, sex, or age field. See "Ethical considerations."
  - **Deployment without ongoing monitoring.** Decision 10 measures that calibration decays to a fixed error floor (ECE ≈0.019) within about two years regardless of which calibrator is used, and finding 12 documents the same drift risk for the decision threshold itself. A frozen, unmonitored deployment of this model (or one like it) would silently miscalibrate over time.

## Model selection: why the scorecard, not the gradient booster

Both a logistic-WoE scorecard and a tuned LightGBM model were built and evaluated identically throughout this project (findings 7–8, 11, 14). The scorecard is the recommended model for three independently-measured reasons, not a default preference for simplicity:

1. **Discrimination is indistinguishable between them.** Tuned-against-tuned, LightGBM beats the scorecard by +0.0068 PR-AUC (paired sd 0.0059) and the sign flips on one of four folds — not a real effect at this sample size (finding 7).
2. **The scorecard's probabilities need no repair; LightGBM's do.** Uncalibrated LightGBM has ECE 0.0308 and a calibration slope of 0.59 (badly under-confident); the uncalibrated scorecard is already close to well-calibrated (ECE 0.0203, slope 0.75). Calibration is worth −0.0118 ECE for LightGBM and an indistinguishable-from-zero −0.0021 for the scorecard (finding 11).
3. **The scorecard is interpretable by construction.** Every feature's contribution is `coefficient × WoE-encoded value`, auditable by a credit officer or regulator without a post-hoc explainability layer. LightGBM requires SHAP to achieve the same transparency, and Phase 7 shows that even SHAP's account of LightGBM leaves an unresolved question (`addr_state`'s SHAP-vs-permutation disagreement, finding 14).

A credit regulator would ask for exactly the properties the scorecard already has and LightGBM does not. This is a case where the "boring" model wins on the metrics that matter for the deployment context, not just on convention.

## Training data

- **Source**: [Lending Club accepted loans, 2007–2018](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (Kaggle mirror of Lending Club's public data release), 2,260,701 loans, 151 raw columns, origination dates June 2007–December 2018 with no missing months.
- **Feature set**: 96 features surviving a per-column leakage audit that classified all 151 raw columns individually (`data/feature_audit.py`) — 101 `keep`, 38 `drop_leakage` (payment history, recoveries, post-origination FICO pulls, hardship/debt-settlement programme fields), 8 `drop_other` (identifiers, constants, the label source), 4 `review` (`grade`/`sub_grade`/`int_rate`/`installment` — Lending Club's own underwriting outputs, included with a stated circularity caveat; see DECISIONS.md decision 5). Free-text and high-cardinality fields (`emp_title`, `desc`, `title`, raw `zip_code`) are on the allow-list but not engineered into features.
- **Target**: `default_window = 1` if the loan charged off within 18 months of origination, `0` if it survived that window; a loan is eligible only once observed for 18+6 months (the extra six covers the typical delinquency-to-charge-off lag). Chosen over Lending Club's raw final-status field specifically to make cohorts of different ages comparable (finding 4).
- **Population after target construction**: **1,445,560 loans, 9.63% default rate.**
- **Preprocessing**: Weight-of-Evidence binning per feature (10 target bins, 2% minimum bin fraction, additive smoothing 0.5, explicit `__missing__`/`__rare__` categories rather than imputation) — chosen so that missingness mechanisms that differ by column (event-never-happened vs. field-not-yet-collected; finding 6) are preserved rather than erased by a single imputation strategy.

## Evaluation data

Four walk-forward, embargoed folds (test years 2014–2017; `models/tuning.py::outer_folds`). Each fold trains only on loans issued early enough that their 18-month outcome was already knowable at the point a real underwriting deployment would have retrained — a stricter, more realistic constraint than a random or even a plain chronological split, and one finding 3 shows costs −0.0131 PR-AUC (sd 0.0055, all four folds) relative to a plain temporal split that peeks at not-yet-matured cohorts. Calibration is fit on a cohort adjacent to, but excluded from, the training window and evaluated on the true test cohort two or more years later, so the reported numbers already reflect calibration drift rather than best-case in-sample calibration.

## Metrics

Reported as mean (sd) across the four folds; every comparison is paired on identical folds because cohort-to-cohort variance is 5–10× most of the effects being measured (README, "Effects that change sign across folds are reported as null").

| Metric | Logistic-WoE (Platt-calibrated) |
|---|---:|
| PR-AUC | 0.165 (sd 0.026), against a 9.7% test-fold base rate |
| ROC-AUC | 0.658 |
| Brier score | 0.0864 |
| Expected Calibration Error | 0.0183 |
| Calibration slope | 0.86 |

For reference, a trivial majority-class classifier scores 90.3% accuracy while catching zero defaults — accuracy is not reported as a headline metric anywhere in this project because it is actively misleading at this base rate.

**Decision-level performance** (cost-optimal threshold, exact search over every distinct calibrated score, `evaluation/decisioning.py`): mean profit $1,222/applicant across the four folds, approving 98.1–99.4% of applicants. This barely beats a naive 0.5 cutoff (+$5.42/applicant, sd $8.96, sign flips on one fold — reported as a null result, not suppressed) because calibrated scores never exceed 0.45 at this base rate, so "reject above 0.5" and "approve nearly everyone" are nearly the same policy. The decision-relevant threshold is closer to 0.20 (the loss-given-default/margin break-even point), and it costs a mean $12.47/applicant (1.1% of oracle profit) when the threshold is set from a cohort two-plus years stale — real, but an order of magnitude smaller in relative terms than the calibration drift above.

## Performance across groups (fairness audit)

Lending Club's public data contains no race, sex, or age field. Three proxies were used, each explicitly a stand-in rather than a measurement: `income_quintile` and `emp_length` (class proxies), and `geo_race_proxy` (a loan's ZIP3 mapped to the Census-measured plurality race/ethnicity of that ZIP3's population — weaker than the surname+geography BISG method regulators use, since Lending Club's data has no borrower name, and it inherits the ecological fallacy: an individual borrower's actual race is never observed, only their ZIP3's plurality). All gaps below are measured at the shared cost-optimal threshold from the "Decision-level performance" row above, not a threshold picked to flatter this audit.

| Proxy | Demographic parity gap | Disparate impact ratio | Equal-opportunity (TPR) gap |
|---|---:|---:|---:|
| income_quintile | 0.024 (sd 0.015) | 0.98 | 0.022 (sd 0.013) |
| emp_length | **0.055 (sd 0.030)** | 0.94 | 0.052 (sd 0.029) |
| geo_race_proxy | 0.011 (sd 0.005) | 0.99 | 0.010 (sd 0.005) |

Every proxy shows both demographic-parity and equal-opportunity gaps simultaneously nonzero at a single shared threshold — the numeric demonstration of the classical impossibility result (Chouldechova 2017; Kleinberg, Mullainathan & Raghavan 2016), not an incidental miss. This project's own near-universal-approval threshold (96–99.7% approved) numerically mutes this tension; a stricter, more typical underwriting approval rate would very likely show it more sharply, and this project has not yet built the code to measure that precisely — stated as an open item, not as a number this card doesn't have.

Per-group thresholds can close either gap: the smallest, best-evidenced mitigation cost measured is `geo_race_proxy`/LightGBM demographic-parity mitigation at +$0.28/applicant (sd $0.65), closing a 0.010 gap to near-zero. Most other measured mitigation costs are not distinguishable from zero given n=4 folds and a cost sd 1–4× the mean.

## Ethical considerations

- **Proxy fairness, not protected-attribute fairness.** Every fairness number in this card and the underlying reports is a statement about an imperfect, stated proxy, never about a borrower's actual race, sex, or age. `geo_race_proxy` specifically labels a ZIP3's Census-measured plurality demographic, not any individual borrower's.
- **An unresolved interaction between the model's geography feature and its own fairness-relevant behavior.** Phase 7 found `addr_state` ranks 4th–7th of 96 features by SHAP importance in both models (comparable in magnitude to `annual_inc`) but its permutation importance — the effect of the feature on held-out predictive accuracy — is indistinguishable from zero. Two candidate explanations (small-sample overfitting, correlation with another feature) were checked and ruled out; no confirmed explanation exists. Because `addr_state` (as a ZIP3-level proxy) is also where the measured geographic fairness gap comes from, this is flagged here as an open question that would need resolving before this model, or one like it, was used in any context where the geographic fairness finding carries real weight.
- **A single shared decision threshold cannot satisfy both demographic parity and equalized odds simultaneously** when group base rates differ, which they measurably do here (up to a 4.5-point default-rate spread by income quintile). This is a property of the problem, not a fixable bug, and any deployment has to choose which criterion (if either) it will optimize for, and accept a cost for doing so.
- **The training population is a public secondary-market consumer-lending dataset from 2007–2018**, not a representative sample of any specific current lending population, and lending patterns, underwriting standards, and the applicant pool itself have all changed since 2018.

## Caveats and recommendations

- **Discrimination is modest in absolute terms** (PR-AUC ~0.165 against a 9.6% base rate). This is measured to be what this feature set genuinely supports once leakage and the temporal embargo are respected — published numbers far above this on this same dataset are, per finding 1, almost always leakage.
- **Four folds is a small evaluation sample.** Every paired comparison in this project rests on n=4, enough to establish a consistent-sign effect of about ±0.006 but not to resolve differences below about ±0.002. Effects that flip sign across folds are reported as null throughout, including in this card.
- **Calibration and the decision threshold both drift and require periodic refitting** — this project measured the drift, it did not solve it. Any deployment needs a monitoring plan, not a one-time calibration.
- **The fairness tension was measured at one, near-universal-approval operating point.** A stricter approval rate would very likely show a sharper trade-off; this has not been measured and should not be assumed to generalize.
- **Loans delinquent-but-not-yet-charged-off at the data snapshot are counted as survivals** (~1.5% of the file), making the measured default rate slightly conservative.
- **Recommendation if this model (or a descendant of it) were ever considered for a real decision**: do not deploy on the strength of this card alone. At minimum: revalidate on current data and a current applicant population; resolve the `addr_state` open item; measure the fairness tension at the approval rate actually intended for use, not this project's own near-100% baseline; and put a recalibration and threshold-monitoring cadence in place before go-live.
