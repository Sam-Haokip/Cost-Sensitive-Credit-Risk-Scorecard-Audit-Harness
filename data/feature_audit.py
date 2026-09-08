"""
Phase 1 leakage audit: a documented allow-list, not a drop-list, per column.

Every one of the 151 raw columns gets an explicit decision and a one-line
reason. Four possible decisions:

  keep         -- available at the moment of the credit decision (loan
                  application + the credit-bureau report pulled for it).
                  Safe to use as a model input.
  drop_leakage -- only exists, or only gets populated, because of something
                  that happened AFTER the loan was funded (payments,
                  delinquency, hardship programs, debt settlement, updated
                  credit pulls). Using these would let the model see the
                  outcome, directly or indirectly.
  drop_other   -- not leakage, just not a usable model input: it's an
                  identifier, a constant, 100% null, redundant with another
                  kept column, or literally the label itself.
  review       -- available before the loan is funded, technically, but it's
                  an OUTPUT of Lending Club's own underwriting/pricing model
                  rather than a raw applicant characteristic. Using it isn't
                  leakage in the classic sense, but it risks the model just
                  re-deriving LC's own decision instead of learning the
                  underlying risk relationship. Flagged for an explicit call,
                  not silently included or excluded.
"""

# (decision, reason)
FEATURE_AUDIT = {
    # --- identifiers / administrative, not leakage, just not features ---
    "id": ("drop_other", "database key, not a risk characteristic"),
    "member_id": ("drop_other", "100% null in this public release"),
    "url": ("drop_other", "just embeds the loan id, no information"),
    "issue_d": ("drop_other", "origination timestamp -- used to build cohorts/folds, not fed to the model as a feature"),
    "loan_status": ("drop_other", "this IS the source of the target; including it as a feature would be perfect leakage"),
    "policy_code": ("drop_other", "constant (1.0) across the entire dataset -- zero variance, no information"),

    # --- loan terms requested / assigned -- REVIEW: pricing outputs vs raw inputs ---
    "loan_amnt": ("keep", "amount the borrower requested at application"),
    "funded_amnt": ("drop_other", "redundant with loan_amnt -- effectively always equal in the modern dataset"),
    "funded_amnt_inv": ("drop_other", "redundant with loan_amnt for the same reason"),
    "term": ("keep", "loan term chosen at application (36 vs 60 months)"),
    "int_rate": ("review", "set by LC's own grading model AS PART of the decision -- not a raw applicant characteristic"),
    "installment": ("review", "derived directly from loan_amnt/term/int_rate -- inherits int_rate's circularity"),
    "grade": ("review", "LC's own risk grade -- an output of their underwriting model, not an input to ours"),
    "sub_grade": ("review", "same concern as grade, finer-grained"),

    # --- borrower self-reported / application-time info ---
    "emp_title": ("keep", "free-text job title given at application (high cardinality -- will need grouping in Phase 3)"),
    "emp_length": ("keep", "employment length stated at application"),
    "home_ownership": ("keep", "stated at application"),
    "annual_inc": ("keep", "self-reported income at application"),
    "verification_status": ("keep", "whether LC verified income as part of underwriting -- known at decision time"),
    "pymnt_plan": ("drop_leakage", "flips to 'y' only when a struggling borrower is placed on an alternative payment plan mid-life"),
    "desc": ("keep", "free-text description written by the borrower at application (94% missing -- largely because LC stopped collecting it after a point, not borrower-driven; NLP feature work, not core tabular set)"),
    "purpose": ("keep", "stated loan purpose at application"),
    "title": ("keep", "borrower's free-text loan title -- mostly redundant with purpose, likely low value but not leakage"),
    "zip_code": ("keep", "3-digit ZIP prefix at application -- also our geography proxy for the fairness audit"),
    "addr_state": ("keep", "state at application -- also a fairness-audit proxy"),
    "application_type": ("keep", "Individual vs Joint, known at application"),
    "annual_inc_joint": ("keep", "co-borrower income, joint applications only (~95% null because most loans aren't joint -- MAR, not random)"),
    "dti_joint": ("keep", "co-borrower DTI, joint applications only"),
    "verification_status_joint": ("keep", "co-borrower verification, joint applications only"),
    "disbursement_method": ("keep", "set at funding (Cash vs DirectPay) -- contemporaneous with the decision, not post-hoc"),
    "initial_list_status": ("keep", "platform/administrative listing status at funding -- pre-funding, weak signal expected but not leakage"),

    # --- credit bureau snapshot pulled at application (the bulk of the allow-list) ---
    "dti": ("keep", "debt-to-income from the credit pull at application"),
    "delinq_2yrs": ("keep", "credit-report history through the application date"),
    "earliest_cr_line": ("keep", "credit-report field -- length of credit history"),
    "fico_range_low": ("keep", "FICO at application (contrast with last_fico_range_*, which is post-origination and dropped)"),
    "fico_range_high": ("keep", "FICO at application"),
    "inq_last_6mths": ("keep", "credit-report field as of application"),
    "mths_since_last_delinq": ("keep", "credit-report field as of application"),
    "mths_since_last_record": ("keep", "credit-report field as of application"),
    "open_acc": ("keep", "credit-report field as of application"),
    "pub_rec": ("keep", "credit-report field as of application"),
    "revol_bal": ("keep", "credit-report field as of application"),
    "revol_util": ("keep", "credit-report field as of application"),
    "total_acc": ("keep", "credit-report field as of application"),
    "collections_12_mths_ex_med": ("keep", "credit-report field as of application per LC data dictionary"),
    "mths_since_last_major_derog": ("keep", "credit-report field as of application"),
    "acc_now_delinq": ("keep", "credit-report field as of application"),
    "tot_coll_amt": ("keep", "credit-report field as of application"),
    "tot_cur_bal": ("keep", "credit-report field as of application"),
    "total_rev_hi_lim": ("keep", "credit-report field as of application"),
    "open_acc_6m": ("keep", "bureau-enrichment field LC added partway through 2007-2018 (~38% null on older loans -- MNAR tied to origination date, not the borrower)"),
    "open_act_il": ("keep", "same bureau-enrichment cohort as open_acc_6m"),
    "open_il_12m": ("keep", "same bureau-enrichment cohort"),
    "open_il_24m": ("keep", "same bureau-enrichment cohort"),
    "mths_since_rcnt_il": ("keep", "same bureau-enrichment cohort"),
    "total_bal_il": ("keep", "same bureau-enrichment cohort"),
    "il_util": ("keep", "same bureau-enrichment cohort"),
    "open_rv_12m": ("keep", "same bureau-enrichment cohort"),
    "open_rv_24m": ("keep", "same bureau-enrichment cohort"),
    "max_bal_bc": ("keep", "same bureau-enrichment cohort"),
    "all_util": ("keep", "same bureau-enrichment cohort"),
    "inq_fi": ("keep", "same bureau-enrichment cohort"),
    "total_cu_tl": ("keep", "same bureau-enrichment cohort"),
    "inq_last_12m": ("keep", "same bureau-enrichment cohort"),
    "acc_open_past_24mths": ("keep", "credit-report field as of application"),
    "avg_cur_bal": ("keep", "credit-report field as of application"),
    "bc_open_to_buy": ("keep", "credit-report field as of application"),
    "bc_util": ("keep", "credit-report field as of application"),
    "chargeoff_within_12_mths": ("keep", "prior chargeoffs BEFORE this loan, from the credit report"),
    "delinq_amnt": ("keep", "credit-report field as of application"),
    "mo_sin_old_il_acct": ("keep", "credit-report field as of application"),
    "mo_sin_old_rev_tl_op": ("keep", "credit-report field as of application"),
    "mo_sin_rcnt_rev_tl_op": ("keep", "credit-report field as of application"),
    "mo_sin_rcnt_tl": ("keep", "credit-report field as of application"),
    "mort_acc": ("keep", "credit-report field as of application"),
    "mths_since_recent_bc": ("keep", "credit-report field as of application"),
    "mths_since_recent_bc_dlq": ("keep", "credit-report field as of application"),
    "mths_since_recent_inq": ("keep", "credit-report field as of application"),
    "mths_since_recent_revol_delinq": ("keep", "credit-report field as of application"),
    "num_accts_ever_120_pd": ("keep", "credit-report field as of application"),
    "num_actv_bc_tl": ("keep", "credit-report field as of application"),
    "num_actv_rev_tl": ("keep", "credit-report field as of application"),
    "num_bc_sats": ("keep", "credit-report field as of application"),
    "num_bc_tl": ("keep", "credit-report field as of application"),
    "num_il_tl": ("keep", "credit-report field as of application"),
    "num_op_rev_tl": ("keep", "credit-report field as of application"),
    "num_rev_accts": ("keep", "credit-report field as of application"),
    "num_rev_tl_bal_gt_0": ("keep", "credit-report field as of application"),
    "num_sats": ("keep", "credit-report field as of application"),
    "num_tl_120dpd_2m": ("keep", "credit-report field as of application"),
    "num_tl_30dpd": ("keep", "credit-report field as of application"),
    "num_tl_90g_dpd_24m": ("keep", "credit-report field as of application"),
    "num_tl_op_past_12m": ("keep", "credit-report field as of application"),
    "pct_tl_nvr_dlq": ("keep", "credit-report field as of application"),
    "percent_bc_gt_75": ("keep", "credit-report field as of application"),
    "pub_rec_bankruptcies": ("keep", "credit-report field as of application"),
    "tax_liens": ("keep", "credit-report field as of application"),
    "tot_hi_cred_lim": ("keep", "credit-report field as of application"),
    "total_bal_ex_mort": ("keep", "credit-report field as of application"),
    "total_bc_limit": ("keep", "credit-report field as of application"),
    "total_il_high_credit_limit": ("keep", "credit-report field as of application"),

    # --- co-applicant (joint loans only, ~95% null -- MAR by application_type) ---
    "revol_bal_joint": ("keep", "co-applicant credit field, joint applications only"),
    "sec_app_fico_range_low": ("keep", "co-applicant credit field, joint applications only"),
    "sec_app_fico_range_high": ("keep", "co-applicant credit field, joint applications only"),
    "sec_app_earliest_cr_line": ("keep", "co-applicant credit field, joint applications only"),
    "sec_app_inq_last_6mths": ("keep", "co-applicant credit field, joint applications only"),
    "sec_app_mort_acc": ("keep", "co-applicant credit field, joint applications only"),
    "sec_app_open_acc": ("keep", "co-applicant credit field, joint applications only"),
    "sec_app_revol_util": ("keep", "co-applicant credit field, joint applications only"),
    "sec_app_open_act_il": ("keep", "co-applicant credit field, joint applications only"),
    "sec_app_num_rev_accts": ("keep", "co-applicant credit field, joint applications only"),
    "sec_app_chargeoff_within_12_mths": ("keep", "co-applicant credit field, joint applications only"),
    "sec_app_collections_12_mths_ex_med": ("keep", "co-applicant credit field, joint applications only"),
    "sec_app_mths_since_last_major_derog": ("keep", "co-applicant credit field, joint applications only"),

    # --- post-origination servicing / payment history: textbook leakage ---
    "out_prncp": ("drop_leakage", "outstanding principal -- only meaningful once the loan is being serviced"),
    "out_prncp_inv": ("drop_leakage", "same as out_prncp"),
    "total_pymnt": ("drop_leakage", "cumulative payments received -- post-origination"),
    "total_pymnt_inv": ("drop_leakage", "same as total_pymnt"),
    "total_rec_prncp": ("drop_leakage", "cumulative principal received -- post-origination"),
    "total_rec_int": ("drop_leakage", "cumulative interest received -- post-origination"),
    "total_rec_late_fee": ("drop_leakage", "late fees only accrue after a missed payment"),
    "recoveries": ("drop_leakage", "money recovered after charge-off -- literally post-outcome"),
    "collection_recovery_fee": ("drop_leakage", "fee on recoveries -- post-outcome"),
    "last_pymnt_d": ("drop_leakage", "date of most recent payment -- post-origination"),
    "last_pymnt_amnt": ("drop_leakage", "amount of most recent payment -- post-origination"),
    "next_pymnt_d": ("drop_leakage", "scheduled next payment -- a defaulted loan simply won't have one, directly encoding the outcome"),
    "last_credit_pull_d": ("drop_leakage", "date LC last pulled credit for servicing -- post-origination"),
    "last_fico_range_high": ("drop_leakage", "updated FICO from a post-origination pull -- will visibly crater for defaulters"),
    "last_fico_range_low": ("drop_leakage", "same as last_fico_range_high"),

    # --- hardship program: only exists for loans already in trouble ---
    "hardship_flag": ("drop_leakage", "whether the loan EVER entered a hardship program -- a mid-life event"),
    "hardship_type": ("drop_leakage", "hardship program details -- post-origination"),
    "hardship_reason": ("drop_leakage", "hardship program details -- post-origination"),
    "hardship_status": ("drop_leakage", "hardship program details -- post-origination"),
    "deferral_term": ("drop_leakage", "hardship program details -- post-origination"),
    "hardship_amount": ("drop_leakage", "hardship program details -- post-origination"),
    "hardship_start_date": ("drop_leakage", "hardship program details -- post-origination"),
    "hardship_end_date": ("drop_leakage", "hardship program details -- post-origination"),
    "payment_plan_start_date": ("drop_leakage", "hardship program details -- post-origination"),
    "hardship_length": ("drop_leakage", "hardship program details -- post-origination"),
    "hardship_dpd": ("drop_leakage", "days past due during hardship -- post-origination"),
    "hardship_loan_status": ("drop_leakage", "loan status snapshot taken during the hardship program -- post-origination"),
    "orig_projected_additional_accrued_interest": ("drop_leakage", "hardship program details -- post-origination"),
    "hardship_payoff_balance_amount": ("drop_leakage", "hardship program details -- post-origination"),
    "hardship_last_payment_amount": ("drop_leakage", "hardship program details -- post-origination"),

    # --- debt settlement: only exists for loans already seriously delinquent ---
    "debt_settlement_flag": ("drop_leakage", "whether a debt-settlement company got involved -- a mid-life event"),
    "debt_settlement_flag_date": ("drop_leakage", "debt-settlement details -- post-origination"),
    "settlement_status": ("drop_leakage", "debt-settlement details -- post-origination"),
    "settlement_date": ("drop_leakage", "debt-settlement details -- post-origination"),
    "settlement_amount": ("drop_leakage", "debt-settlement details -- post-origination"),
    "settlement_percentage": ("drop_leakage", "debt-settlement details -- post-origination"),
    "settlement_term": ("drop_leakage", "debt-settlement details -- post-origination"),
}


if __name__ == "__main__":
    from collections import Counter

    counts = Counter(v[0] for v in FEATURE_AUDIT.values())
    print(f"total columns audited: {len(FEATURE_AUDIT)}")
    for decision, n in counts.most_common():
        print(f"  {decision}: {n}")
