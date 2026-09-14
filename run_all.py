"""
Phase 8: a single command that regenerates every table and figure in this
repository, closing the brief's reproducibility requirement literally
rather than leaving it as ~20 commands a reader has to copy out of the
README one at a time.

WHY A THIN ORCHESTRATOR RATHER THAN A REWRITE
------------------------------------------------
Every step below is an existing, already-tested `python -m module` command
-- this file does not reimplement any of them, it runs them in the one
order that respects the project's own dependencies (e.g. `data.convert_raw`
must produce the Parquet files before anything else can read them;
`models.baselines` must run before `models.tuning` can compare against it
implicitly documented, not enforced by code, so getting the order right
here IS the thing this file is for). The step list is kept in exact sync
with the README's "Reproducing" section by hand -- if you add a phase,
update both.

WHAT THIS SCRIPT DOES NOT DO, STATED RATHER THAN HIDDEN
-----------------------------------------------------------
It does not parallelise independent steps (several could run concurrently
-- e.g. `models.imbalance` and `evaluation.drift` do not depend on each
other -- but sequencing them makes failures easier to attribute to a single
step, and this project's own steps are CPU-bound enough that parallelising
on a laptop would mostly contend for the same cores anyway). It does not
retry a failed step. It does not fetch the raw CSV or run the optional
Census refetch (`fairness.geo_proxy`) by default, both for reasons stated
inline below.

MEASURED COST, NOT ASSUMED
-----------------------------
This has NOT been timed end-to-end on the full ~1.6GB raw file (that run
was never executed as a single sequential command before this script
existed -- each phase's script was run and verified individually, on its
own schedule, over the life of this project). The one component with a
directly measured cost is `explainability.explain`: ~28 minutes on the full
4 folds (see that module's docstring). Extrapolating from how the other
phases' own scripts behaved when they were built (walk-forward fits on
folds up to 150,000 rows, several nested hyperparameter searches), a full
run is realistically on the order of an hour or more on a laptop, dominated
by `models.tuning`, `models.logistic_tuning`, `models.imbalance`, and
`explainability.explain`. This script prints that estimate and asks for
confirmation before running the slow steps, rather than silently starting
a run whose length nobody agreed to.
"""
import argparse
import subprocess
import sys
import time

# (name, command, est_minutes, note) -- order matches README.md's
# "Reproducing" section exactly; est_minutes is a rough, stated guess
# except where noted as measured.
STEPS = [
    ("convert_raw", [sys.executable, "-m", "data.convert_raw"], 2,
     "CSV -> chunked Parquet"),
    ("quality_report", [sys.executable, "-m", "data.quality_report"], 1, None),
    ("vintage_target", [sys.executable, "-m", "data.vintage_target"], 1, None),
    ("leakage_experiment", [sys.executable, "-m", "models.leakage_experiment"], 3, None),
    ("seed_stability", [sys.executable, "-m", "models.seed_stability"], 3, None),
    ("temporal_validation", [sys.executable, "-m", "models.temporal_validation"], 5, None),
    ("drift", [sys.executable, "-m", "evaluation.drift"], 1, None),
    ("validation_robustness", [sys.executable, "-m", "models.validation_robustness"], 5, None),
    ("baselines", [sys.executable, "-m", "models.baselines"], 3, None),
    ("tuning", [sys.executable, "-m", "models.tuning"], 15, "nested search, LightGBM"),
    ("logistic_tuning", [sys.executable, "-m", "models.logistic_tuning"], 10, None),
    ("gbm_ablation", [sys.executable, "-m", "models.gbm_ablation"], 5, None),
    ("imbalance", [sys.executable, "-m", "models.imbalance"], 10, "includes SMOTE-NC"),
    ("smote_diagnostic", [sys.executable, "-m", "models.smote_diagnostic"], 3, None),
    ("calibration", [sys.executable, "-m", "evaluation.calibration"], 5, None),
    ("decisioning", [sys.executable, "-m", "evaluation.decisioning"], 3, None),
    ("fairness_audit", [sys.executable, "-m", "fairness.audit"], 3,
     "uses the committed Census CSV; does not refetch"),
    ("explainability", [sys.executable, "-m", "explainability.explain"], 28,
     "measured, not estimated -- see explainability/explain.py's docstring"),
    ("figures", [sys.executable, "reports/make_figures.py"], 1,
     "regenerates PNGs from the CSVs above, not from raw data"),
    ("tests", [sys.executable, "-m", "pytest", "-q"], 1,
     "82 tests; runs without the dataset, included here as a final check"),
]

OPTIONAL_STEPS = [
    ("census_refetch", [sys.executable, "-m", "fairness.geo_proxy"], 5,
     "needs LC_CENSUS_API_KEY and network access to api.census.gov "
     "(blocked on some networks -- see DECISIONS.md decision 12); "
     "the committed data/processed/census_zip3_race_proxy.csv already "
     "makes this unnecessary to reproduce fairness_audit's numbers"),
]


def run_step(name, cmd, note):
    print(f"\n{'=' * 70}\n{name}" + (f"  ({note})" if note else "") + f"\n{'=' * 70}")
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"-- {name}: {status} in {elapsed:.1f}s")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run every reproduction step in this repository, in "
                     "the order README.md's Reproducing section documents.")
    parser.add_argument("--from-step", default=None,
                         help="Skip steps before this one (by name; see STEPS above). "
                              "Useful for resuming after a failure without re-running "
                              "everything that already succeeded.")
    parser.add_argument("--only", default=None,
                         help="Run exactly one named step and exit.")
    parser.add_argument("--include-census-refetch", action="store_true",
                         help="Also run the optional live Census refetch "
                              "(requires LC_CENSUS_API_KEY). Off by default "
                              "because the committed CSV already covers it.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the steps that would run, with their "
                              "estimated time, and exit without running anything.")
    parser.add_argument("--yes", action="store_true",
                         help="Skip the confirmation prompt before the slow steps.")
    args = parser.parse_args()

    steps = STEPS + (OPTIONAL_STEPS if args.include_census_refetch else [])

    if args.only:
        steps = [s for s in steps if s[0] == args.only]
        if not steps:
            names = ", ".join(s[0] for s in STEPS + OPTIONAL_STEPS)
            print(f"Unknown step '{args.only}'. Known steps: {names}", file=sys.stderr)
            return 2
    elif args.from_step:
        names = [s[0] for s in steps]
        if args.from_step not in names:
            print(f"Unknown step '{args.from_step}'. Known steps: {', '.join(names)}",
                  file=sys.stderr)
            return 2
        steps = steps[names.index(args.from_step):]

    total_minutes = sum(s[2] for s in steps)
    print(f"{len(steps)} step(s) queued, roughly {total_minutes} minutes total "
          f"(mostly a stated estimate -- see this file's docstring for what's "
          f"actually measured vs. guessed).")
    for name, _, minutes, note in steps:
        print(f"  {name:24s} ~{minutes:>3d} min" + (f"  -- {note}" if note else ""))

    if args.dry_run:
        return 0

    if not args.yes and total_minutes > 5:
        reply = input(f"\nProceed with an estimated {total_minutes} minutes of work? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    overall_start = time.time()
    for name, cmd, _, note in steps:
        if not run_step(name, cmd, note):
            print(f"\nStopping after {name} failed. Re-run with "
                  f"--from-step {name} once it's fixed to resume without "
                  f"repeating everything before it.")
            return 1
    print(f"\nAll {len(steps)} step(s) completed in {(time.time() - overall_start) / 60:.1f} minutes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
