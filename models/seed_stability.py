"""
Is the grade/int_rate effect real, or inside run-to-run noise?

The leakage experiment reports a single-split gain from adding grade,
sub_grade, int_rate and installment back to the allow-list. That is exactly the
kind of point estimate this project is meant to distrust: a small effect can
easily be one lucky shuffle.

This re-runs the clean-vs-grade_kept comparison across five seeds and reports
the spread, so the claim is a distribution rather than a number. The comparison
to make is between the size of the effect and the seed-to-seed variation in the
headline PR-AUC itself -- an effect smaller than its own measurement noise is
not something to build an argument on.
"""
import statistics as st

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from data.dataset import FEATURE_COLS, load_modelling_frame
from models.leakage_experiment import CATEGORICAL_REVIEW, REVIEW_COLS

SEEDS = (0, 1, 2, 3, 4)
SAMPLE_SIZE = 400_000


if __name__ == "__main__":
    df = load_modelling_frame(extra_cols=REVIEW_COLS, extra_categorical=CATEGORICAL_REVIEW)
    df, _ = train_test_split(df, train_size=SAMPLE_SIZE, random_state=42,
                             stratify=df["default_window"])
    y = df["default_window"]

    clean_cols = FEATURE_COLS
    grade_cols = FEATURE_COLS + REVIEW_COLS

    print(f"{'seed':>5} {'clean':>9} {'grade_kept':>11} {'gap':>9}")
    clean_scores, gaps = [], []
    for seed in SEEDS:
        scores = {}
        for name, cols in (("clean", clean_cols), ("grade_kept", grade_cols)):
            X_train, X_test, y_train, y_test = train_test_split(
                df[cols], y, test_size=0.2, random_state=seed, stratify=y
            )
            model = HistGradientBoostingClassifier(
                categorical_features="from_dtype", max_iter=150, random_state=seed
            )
            model.fit(X_train, y_train)
            scores[name] = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
        gap = scores["grade_kept"] - scores["clean"]
        clean_scores.append(scores["clean"])
        gaps.append(gap)
        print(f"{seed:>5} {scores['clean']:>9.4f} {scores['grade_kept']:>11.4f} {gap:>+9.4f}",
              flush=True)

    print()
    print(f"clean PR-AUC across seeds : mean {st.mean(clean_scores):.4f}, "
          f"sd {st.stdev(clean_scores):.4f}, "
          f"range {min(clean_scores):.4f}-{max(clean_scores):.4f}")
    print(f"grade_kept gap across seeds: mean {st.mean(gaps):+.4f}, "
          f"sd {st.stdev(gaps):.4f}, range {min(gaps):+.4f} to {max(gaps):+.4f}")
    print()
    if all(g > 0 for g in gaps):
        print("Direction consistent across every seed -- the effect is real, not noise.")
    else:
        print("Sign flips across seeds -- not distinguishable from noise.")
    if st.stdev(clean_scores) > abs(st.mean(gaps)):
        print("The seed-to-seed spread in clean PR-AUC exceeds the effect itself,")
        print("which is the honest justification for calling it negligible.")
    else:
        print("The effect exceeds the seed-to-seed spread in clean PR-AUC,")
        print("so it is NOT dismissible as measurement noise.")
