"""
baseline.py — measure run-to-run variance, then pin the first real baseline.

Phase 2's whole point is a detector that shouts when the model gets worse. That
needs two things the mock provider cannot give:

  1. a REAL baseline, captured from a live model on the human-verified golden set
  2. a threshold calibrated to this model's actual noise, not a guess

The mock is deterministic, so its variance is zero and any threshold looks
sensible. A live model samples, so the same prompt on the same emails scores
slightly differently every run. Threshold 0 then reports a REGRESSION on a
single flipped row - which is noise wearing a uniform.

So: run the same eval N times, measure the spread, report the threshold that
spread justifies.

    threshold ~= 2 x standard deviation of run-to-run accuracy

2 SD means a real drop has to clear ordinary sampling noise before it is called
a regression. It is a starting point, not a law: if the threshold this produces
would hide a real regression, the fix is MORE RUNS or a BIGGER DATASET - never
a smaller threshold bought by hoping.

Deliberate choices
------------------
* The baseline is run 1, a REAL run. Not the mean of the runs. An averaged run
  is a record that never happened - the same false-provenance failure the
  golden dataset was built to avoid.
* Setting the baseline requires --set-baseline. Measuring is the default, so a
  curious first run cannot silently move the bar everything else is judged by.
* An existing baseline under the same key is never overwritten without --force.
* load_golden_dataset() refuses an unverified dataset on its own, so this
  cannot baseline on labels a human has not finished deciding.

Usage
-----
    py baseline.py --runs 3                       # measure + store runs, no baseline
    py baseline.py --runs 3 --set-baseline        # ...and pin run 1 as the baseline
    py baseline.py --runs 5 --prompt v2 --dry-run # measure only, store nothing
    OLLAMA_MODEL=qwen2.5:1.5b py baseline.py --runs 3 --set-baseline
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluator import (  # noqa: E402
    default_baseline_key,
    evaluate,
    load_golden_dataset,
)
from providers import get_client  # noqa: E402
from storage import get_store  # noqa: E402


def _as_row(run) -> dict:
    """Flatten a RunResult into the dict storage.save_run expects."""
    return {
        "run_id": run.run_id,
        "created_at": run.created_at,
        "provider": run.provider,
        "prompt_version": run.prompt_version,
        "dataset_version": run.dataset_version,
        "n_samples": run.n_samples,
        "accuracy": run.accuracy,
        "metrics": run.metrics.model_dump(),
    }


def _flip_report(runs) -> list[tuple[str, int]]:
    """Which records disagree with themselves across runs.

    Aggregate accuracy can look stable while the same records keep changing
    their answer - the model is guessing, not agreeing. This is the view that
    shows it, and it is what tells you whether variance is spread thinly or
    concentrated in a handful of genuinely ambiguous emails.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        for p in run.predictions:
            seen[p.sample_id].add(p.category)
    flips = [(sid, cats) for sid, cats in seen.items() if len(cats) > 1]
    return sorted(flips, key=lambda t: -len(t[1]))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure live run-to-run variance and optionally pin a baseline."
    )
    ap.add_argument("--runs", type=int, default=3, help="how many times to repeat the eval")
    ap.add_argument("--prompt", default="v1", help="prompt FILE version to score (default v1)")
    ap.add_argument("--provider", default="ollama", help="live provider (default ollama)")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N records (smoke test - never baseline on this)")
    ap.add_argument("--set-baseline", action="store_true",
                    help="pin run 1 as the baseline for this key")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing baseline under the same key")
    ap.add_argument("--dry-run", action="store_true",
                    help="measure only: store no runs, set no baseline")
    args = ap.parse_args()

    if args.runs < 2:
        print("--runs must be at least 2: with one run there is no variance to measure.")
        return 2

    # Refuses an unverified dataset on its own - this is the gate that stops a
    # baseline being taken on labels still under review.
    rows, dataset_version = load_golden_dataset()
    if args.limit:
        rows = rows[:args.limit]
        print(f"NOTE: --limit {args.limit} - scoring a subset. Useful for smoke tests,")
        print("      but a threshold or baseline taken from a subset measures the")
        print("      subset, not the model.")
        print()
    client = get_client(args.provider)
    store = None if args.dry_run else get_store()

    key = None
    if store is not None:
        probe = evaluate(client=client, samples=rows[:1], prompt_version=args.prompt,
                         dataset_version=dataset_version)
        key = default_baseline_key(probe)
        existing = store.get_baseline(key)
        if existing and args.set_baseline and not args.force:
            print(f"ABORT: a baseline already exists under {key} "
                  f"(accuracy {existing.get('accuracy')}).")
            print("       Re-run with --force to replace it. Baselines are not")
            print("       overwritten casually - everything since has judged itself")
            print("       against that number.")
            return 3

    print(f"dataset : {dataset_version}  ({len(rows)} records)")
    print(f"model   : {getattr(client, 'name', 'unknown')}"
          f"  prompt: {args.prompt}")
    if key:
        print(f"key     : {key}")
    print(f"runs    : {args.runs}{'  (dry run - nothing stored)' if args.dry_run else ''}")
    print()

    runs = []
    for i in range(args.runs):
        t = time.time()
        # One failed call kills the whole run by design: the contract raises at
        # a single boundary rather than letting a malformed answer skew every
        # metric downstream. That is correct, and it is also why a run is
        # retried here rather than tolerated - a half-scored run is not a run.
        # Earlier successful runs are already stored, so a crash costs at most
        # the run in flight.
        for attempt in (1, 2):
            try:
                run = evaluate(client=client, samples=rows, prompt_version=args.prompt,
                               dataset_version=dataset_version)
                break
            except Exception as exc:
                if attempt == 2:
                    print(f"  run {i+1}/{args.runs} FAILED after 2 attempts: "
                          f"{type(exc).__name__}: {str(exc)[:100]}")
                    print("    earlier runs are already stored; re-run to add more.")
                    if store is not None:
                        store.close()
                    return 4
                print(f"  run {i+1}/{args.runs} attempt {attempt} failed "
                      f"({type(exc).__name__}), retrying in 5s...")
                time.sleep(5)
        runs.append(run)
        if store is not None:
            store.save_run(_as_row(run), [p.model_dump() for p in run.predictions])
        print(f"  run {i+1}/{args.runs}  accuracy {run.accuracy:.4f} "
              f"({run.metrics.n_correct}/{run.n_samples})  {time.time()-t:.0f}s")

    accs = [r.accuracy for r in runs]
    mean = statistics.fmean(accs)
    sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
    spread = max(accs) - min(accs)
    recommended = round(2 * sd, 4)

    print()
    print("variance across runs")
    print(f"  mean   : {mean:.4f}")
    print(f"  sd     : {sd:.4f}")
    print(f"  min/max: {min(accs):.4f} / {max(accs):.4f}  (spread {spread:.4f})")
    print(f"  n correct per run: {[r.metrics.n_correct for r in runs]}")

    flips = _flip_report(runs)
    print()
    print(f"unstable records: {len(flips)} of {runs[0].n_samples} "
          f"changed their predicted category at least once")
    for sid, cats in flips[:5]:
        print(f"    {sid}: {' -> '.join(sorted(cats))}")
    if len(flips) > 5:
        print(f"    ... and {len(flips) - 5} more")

    print()
    print("threshold")
    print(f"  recommended = 2 x sd = {recommended}")
    if recommended == 0.0:
        print("  NOTE: sd is 0.0000 - this model is reproducing itself exactly.")
        print("        Either it is near-deterministic or the runs are too few to")
        print("        show spread. Treat a 0.0 threshold as unproven, not proven.")
    print("  A drop larger than this is a real change, not sampling noise.")
    print("  A drop smaller than this is indistinguishable from the model having")
    print("  a slightly different morning.")

    if store is None or not args.set_baseline:
        print()
        print("No baseline set (measurement only). To pin run 1:")
        print(f"    py baseline.py --runs {args.runs} --prompt {args.prompt} --set-baseline")
        return 0

    store.set_baseline(key, _as_row(runs[0]))
    print()
    print(f"BASELINE SET: {key} = run 1, accuracy {runs[0].accuracy:.4f}")
    print(f"  suggested threshold for future runs: {recommended}")

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
