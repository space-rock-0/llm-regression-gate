"""scripts/ci_summary.py — render a CI-friendly Markdown summary of an MRDS run.

Usage:
    py scripts/ci_summary.py [--run-id ID] [--baseline-key KEY]
                            [--report-path PATH] [--summary-file PATH]
                            [--provider P] [--prompt V]

Loads the run (re-attaching predictions from SQLite so a stored run is not
missing per-case data), compares it to the baseline, writes the HTML diff
report, and prints a Markdown summary. Exits 1 only on CRITICAL_REGRESSION so
the workflow can block merges on that status.
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluator import (  # noqa: E402
    CaseMetrics,
    Prediction,
    RunResult,
    compare_to_baseline,
    default_baseline_key,
)
from reporter import generate_report  # noqa: E402
from storage import get_store  # noqa: E402


def _attach_predictions(store, run: RunResult) -> RunResult:
    """Safely re-attach per-case predictions from SQLite (avoids KeyError)."""
    try:
        rows = store.get_predictions(run.run_id)
    except Exception:
        rows = []
    preds = []
    for p in rows or []:
        try:
            cm = p.get("case_metrics") if isinstance(p, dict) else None
            case_metrics = CaseMetrics.model_validate(cm) if cm else None
        except Exception:
            case_metrics = None
        try:
            preds.append(
                Prediction(
                    sample_id=p["sample_id"],
                    expected=p.get("expected", ""),
                    category=p.get("category", ""),
                    correct=bool(p.get("correct", False)),
                    summary=p.get("summary", "") or "",
                    case_metrics=case_metrics,
                )
            )
        except Exception:
            # skip malformed rows rather than crash the whole summary
            continue
    if preds:
        run.predictions = preds
    return run


def _load_run(store, run_id=None, provider=None, prompt_version=None):
    if run_id:
        row = store.get_run(run_id)
        if row is None:
            raise SystemExit(f"unknown run_id: {run_id}")
    else:
        runs = list(store.list_runs() or [])
        if provider:
            runs = [r for r in runs if r.get("provider") == provider]
        if prompt_version:
            runs = [r for r in runs if r.get("prompt_version") == prompt_version]
        if not runs:
            raise SystemExit("no runs found in store")
        runs.sort(key=lambda r: r.get("created_at", ""))
        row = runs[-1]
    run = RunResult.model_validate(row)
    run = _attach_predictions(store, run)
    return run


def _badge(status):
    colors = {
        "PASS": "brightgreen",
        "WARN": "yellow",
        "CRITICAL_REGRESSION": "red",
        "BASELINE_CREATED": "blue",
    }
    color = colors.get(status, "lightgrey")
    label = status.replace("_", "%20")
    return f"![{status}](https://img.shields.io/badge/MRDS-{label}-{color})"


def _md(run, verdict, report_path):
    status = verdict.status
    cur = verdict.current_accuracy
    base = verdict.baseline_accuracy
    delta = verdict.accuracy_delta
    reg_count = len(verdict.regressions or [])
    lines = []
    lines.append("## MRDS Evaluation Result")
    lines.append("")
    lines.append(f"{_badge(status)} **{status}**")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Provider | {run.provider} |")
    lines.append(f"| Prompt version | {run.prompt_version} |")
    lines.append(f"| Dataset version | {run.dataset_version} |")
    lines.append(f"| Samples | {run.n_samples} |")
    lines.append(f"| Pass rate | {cur * 100:.1f}% |")
    lines.append(f"| Baseline | {(base * 100 if base is not None else 0):.1f}% |")
    lines.append(f"| Delta | {(delta * 100 if delta is not None else 0):+.2f}% |")
    lines.append(f"| Regressed cases | {reg_count} |")
    lines.append(f"| HTML report | `{report_path}` |")
    lines.append("")
    lines.append("### Per-category breakdown")
    lines.append("")
    lines.append("| Category | Current | Baseline | Delta |")
    lines.append("|---|---|---|---|")
    baseline_metrics = {}
    try:
        baseline = None
        # best-effort baseline metrics via store lookup happens in caller; fall back to verdict
    except Exception:
        baseline_metrics = {}
    # current metrics
    cur_totals = run.metrics.per_category_total or {}
    cur_correct = run.metrics.per_category_correct or {}
    deltas = verdict.per_category_delta or {}
    for cat in sorted(cur_totals):
        ct = int(cur_totals.get(cat, 0))
        cc = int(cur_correct.get(cat, 0))
        cur_acc = (cc / ct) if ct else 0.0
        d = deltas.get(cat)
        base_acc = cur_acc - (d or 0.0)
        lines.append(
            f"| {cat} | {cc}/{ct} ({cur_acc * 100:.1f}%) | "
            f"{base_acc * 100:.1f}% | {(d * 100 if d is not None else 0):+.2f}% |"
        )
    lines.append("")
    if reg_count:
        lines.append(f"### Regressed cases ({reg_count})")
        lines.append("")
        for rid in (verdict.regressions or [])[:15]:
            lines.append(f"- `{rid}`")
        if reg_count > 15:
            lines.append(f"- ... and {reg_count - 15} more")
        lines.append("")
    if status == "CRITICAL_REGRESSION":
        lines.append("> ⛔ Merge blocked: CRITICAL_REGRESSION detected.")
    elif status == "WARN":
        lines.append("> ⚠️ Warning: pass rate dropped beyond warn threshold.")
    elif status == "BASELINE_CREATED":
        lines.append("> 🆕 No prior baseline; this run was pinned as the baseline.")
    else:
        lines.append("> ✅ No regression detected.")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", "--run_id", dest="run_id", default=None)
    ap.add_argument("--baseline-key", "--baseline_key", dest="baseline_key", default=None)
    ap.add_argument("--report-path", "--report_path", dest="report_path", default=None)
    ap.add_argument("--summary-file", "--summary_file", dest="summary_file", default=None)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--store", default=None, help="path to sqlite/json store (optional)")
    args = ap.parse_args()

    if args.store:
        from reporter import _as_store as _store_factory

        store = _store_factory(args.store)
    else:
        store = get_store()
    run = _load_run(store, args.run_id, args.provider, args.prompt)
    baseline_key = args.baseline_key or default_baseline_key(run)

    warn = float(os.getenv("WARN_THRESHOLD", "0.03"))
    crit = float(os.getenv("CRITICAL_THRESHOLD", "0.08"))
    verdict = compare_to_baseline(
        run, store=store, baseline_key=baseline_key,
        warn_threshold=warn, critical_threshold=crit,
    )

    report_path = args.report_path or str(ROOT / "data" / "ci_report.html")
    try:
        generated = generate_report(
            run, verdict, store=store, baseline_key=baseline_key,
            output_path=report_path,
        )
        report_path = generated
    except Exception as exc:
        print(f"[ci_summary] report generation failed: {exc}", file=sys.stderr)

    md = _md(run, verdict, report_path)
    if args.summary_file:
        p = Path(args.summary_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md, encoding="utf-8")
    print(md)

    # GitHub Actions step summary
    ghs = os.getenv("GITHUB_STEP_SUMMARY")
    if ghs:
        try:
            with open(ghs, "a", encoding="utf-8") as f:
                f.write(md + "\n")
        except Exception:
            pass

    if verdict.status == "CRITICAL_REGRESSION":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
