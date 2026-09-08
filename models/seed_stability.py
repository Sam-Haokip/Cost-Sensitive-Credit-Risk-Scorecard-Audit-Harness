"""
Is the grade/int_rate effect real, or is it inside run-to-run noise?

The leakage experiment reports a +0.003 PR-AUC gain from adding grade,
sub_grade, int_rate and installment back to the allow-list -- and concludes
that excluding them is therefore nearly free. That conclusion rests on a
single random split, which is exactly the kind of point estimate this project
is supposed to distrust: an effect that small could easily be an artefact of
one lucky shuffle.

This re-runs the clean-vs-grade_kept comparison across five seeds and reports
the spread, so the claim is a distribution rather than a number.

Result (see reports/data_quality.md context and the README): the gap is
consistently positive across every seed, so the direction is real -- but it is
smaller than the seed-to-seed variation in the headline PR-AUC itself, which
is the honest way to describe "negligible".
"""
import statistics as st

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from models.leakage_experiment import KEEP_COLS, REVIEW_COLS, load_master

SEEDS = (0, 1, 2, 3, 4)
SAMPLE_SIZE = 400_000


if __name__ == "__main__":
    master = load_master()
    master, _ = train_test_split(
        master, train_size=SAMPLE_SIZE, random_state=42, stratify=master["default"]
    )
    y = master["default"]

    clean_cols = KEEP_COLS + ["credit_history_months"]
    grade_cols = clean_cols + REVIEW_COLS

    print(f"{'seed':>5} {'clean':>9} {'grade_kept':>11} {'gap':>9}")
    clean_scores, gaps = [], []
    for seed in SEEDS:
        scores = {}
        for name, cols in (("clean", clean_cols), ("grade_kept", grade_cols)):
            X_train, X_test, y_train, y_test = train_test_split(
                master[cols], y, test_size=0.2, random_state=seed, stratify=y
            )
            model = HistGradientBoostingClassifier(
                categorical_features="from_dtype", max_iter=150, random_state=seed
            )
            model.fit(X_train, y_train)
            scores[name] = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
        gap = scores["grade_kept"] - scores["clean"]
        clean_scores.append(scores["clean"])
        gaps.append(gap)
        print(f"{seed:>5} {scores['clean']:>9.4f} {scores['grade_kept']:>11.4f} {gap:>+9.4f}")

    print()
    print(f"clean PR-AUC across seeds: mean {st.mean(clean_scores):.4f}, "
          f"sd {st.stdev(clean_scores):.4f}, "
          f"range {min(clean_scores):.4f}-{max(clean_scores):.4f}")
    print(f"grade_kept gap across seeds: mean {st.mean(gaps):+.4f}, "
          f"sd {st.stdev(gaps):.4f}, "
          f"range {min(gaps):+.4f} to {max(gaps):+.4f}")
    print()
    if all(g > 0 for g in gaps):
        print("Direction is consistent across every seed -- the effect is real, not noise.")
    else:
        print("Sign flips across seeds -- the effect is not distinguishable from noise.")
    if st.stdev(clean_scores) > abs(st.mean(gaps)):
        print("But the seed-to-seed spread in clean PR-AUC exceeds the effect itself,")
        print("which is the honest justification for calling it negligible.")
