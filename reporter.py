"""reporter.py — standalone HTML diff report for one eval run vs its baseline.

Polished, dependency-free (stdlib only) report: run metadata, headline
scorecard, per-category breakdown, side-by-side regressed cases, and a trend
table across the last N runs in SQLite. Styled from the ui-ux-pro-max
"Data-Dense Dashboard" palette (blue data + amber highlights).
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluator import RunResult, Verdict, default_baseline_key
from storage import SQLiteStore, get_store

# ---------- palette (ui-ux-pro-max: Data-Dense Dashboard) ----------
COLORS = {
    "primary": "#1E40AF",
    "secondary": "#3B82F6",
    "accent": "#D97706",
    "bg": "#F8FAFC",
    "fg": "#1E3A8A",
    "card": "#FFFFFF",
    "muted": "#E9EEF6",
    "muted_fg": "#475569",
    "border": "#DBEAFE",
    "destructive": "#DC2626",
    "success": "#16A34A",
    "warning": "#D97706",
}

DEFAULT_LAST_N = 10
DEFAULT_OUTPUT_DIR = Path("data") / "reports"


# ---------- normalisation helpers ----------
def _as_store(store: Any = None) -> Any:
    if store is None:
        return get_store()
    if isinstance(store, (str, Path)):
        return SQLiteStore(store)
    return store


def _as_run(run: Any, store: Any = None) -> RunResult:
    if isinstance(run, RunResult):
        rr = run
    elif isinstance(run, str):
        st = _as_store(store)
        row = st.get_run(run)
        if row is None:
            raise ValueError(f"unknown run_id: {run!r}")
        rr = RunResult.model_validate(row)
        # Re-attach predictions that SQLite keeps in a side table so stored
        # runs are not missing per-case data (avoids KeyError/empty diffs).
        try:
            preds = st.get_predictions(run)
            if preds and not rr.predictions:
                from evaluator import Prediction

                rr.predictions = [
                    Prediction(
                        sample_id=p.get("sample_id", ""),
                        expected=p.get("expected", ""),
                        category=p.get("category", ""),
                        correct=bool(p.get("correct", False)),
                        summary=p.get("summary", "") or "",
                    )
                    for p in preds
                    if p.get("sample_id") is not None
                ]
        except Exception:
            pass
    elif isinstance(run, dict):
        rr = RunResult.model_validate(run)
    elif hasattr(run, "model_dump"):
        rr = RunResult.model_validate(run.model_dump())
    else:
        raise TypeError(f"cannot coerce {type(run)!r} to RunResult")
    # attach predictions for dict/run objects that lack them but have a store
    if not getattr(rr, "predictions", None) and store is not None:
        try:
            st = _as_store(store)
            preds = st.get_predictions(rr.run_id)
            if preds:
                from evaluator import Prediction

                rr.predictions = [
                    Prediction(
                        sample_id=p.get("sample_id", ""),
                        expected=p.get("expected", ""),
                        category=p.get("category", ""),
                        correct=bool(p.get("correct", False)),
                        summary=p.get("summary", "") or "",
                    )
                    for p in preds
                    if p.get("sample_id") is not None
                ]
        except Exception:
            pass
    return rr


def _as_verdict(verdict: Any) -> Verdict | None:
    if verdict is None:
        return None
    if isinstance(verdict, Verdict):
        return verdict
    if isinstance(verdict, dict):
        return Verdict.model_validate(verdict)
    if hasattr(verdict, "model_dump"):
        return Verdict.model_validate(verdict.model_dump())
    return None


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%"


def _fmt_num(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _safe(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _load_sample_text(sample_id: str) -> tuple[str, list[str]]:
    """Best-effort lookup of an email body + edge-case tags for a sample id."""
    snippet = ""
    tags: list[str] = []
    # samples/emails.json
    try:
        p = Path("samples") / "emails.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for row in data:
                if str(row.get("id")) == str(sample_id):
                    snippet = row.get("body") or row.get("text") or row.get("email") or ""
                    tags = row.get("tags") or row.get("edge_case_tags") or []
                    return snippet, list(tags)
    except Exception:
        pass
    # golden dataset
    try:
        for gp in [
            Path("golden_dataset") / "golden_dataset_v0.4_reviewed.json",
            Path("data") / "golden_dataset_v0.4_reviewed.json",
        ]:
            if gp.exists():
                data = json.loads(gp.read_text(encoding="utf-8"))
                rows = data.get("records", []) if isinstance(data, dict) else data
                for row in rows:
                    if str(row.get("id")) == str(sample_id):
                        snippet = row.get("input") or row.get("body") or row.get("text") or ""
                        tags = row.get("review_flags") or row.get("edge_case_tags") or row.get("tags") or []
                        return snippet, list(tags)
    except Exception:
        pass
    return snippet, tags


def _snippet_text(text: str, limit: int = 220) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _status_class(status: str | None) -> str:
    s = (status or "").upper()
    if "CRITICAL" in s:
        return "critical"
    if "WARN" in s or "DRIFT" in s:
        return "warn"
    if "PASS" in s:
        return "pass"
    return "neutral"


def _status_emoji(status: str | None) -> str:
    s = (status or "").upper()
    if "CRITICAL" in s:
        return "🔴"
    if "WARN" in s or "DRIFT" in s:
        return "⚠️"
    if "PASS" in s:
        return "🟢"
    if "BASELINE" in s:
        return "🆕"
    return "ℹ️"


def _resolve_output_path(
    run: RunResult,
    output_path: Any = None,
    path: Any = None,
    report_path: Any = None,
    output_dir: Any = None,
    file: Any = None,
    filename: Any = None,
    **_kw: Any,
) -> Path:
    cand = output_path if output_path is not None else path
    if cand is None:
        cand = report_path if report_path is not None else file
    if cand is not None:
        p = Path(cand)
        if p.exists() and p.is_dir():
            name = filename or f"mrds_report_{run.run_id}.html"
            return p / name
        if str(cand).endswith(("/", "\\")):
            name = filename or f"mrds_report_{run.run_id}.html"
            return Path(cand) / name
        return p
    base = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    name = filename or f"mrds_report_{run.run_id}.html"
    return base / name


def _per_category_rows(run: RunResult, baseline_metrics: dict | None) -> list[dict]:
    cur_totals = run.metrics.per_category_total or {}
    cur_correct = run.metrics.per_category_correct or {}
    base_totals = (baseline_metrics or {}).get("per_category_total", {}) or {}
    base_correct = (baseline_metrics or {}).get("per_category_correct", {}) or {}
    cats = sorted(set(cur_totals) | set(base_totals))
    rows = []
    for cat in cats:
        ct = int(cur_totals.get(cat, 0))
        cc = int(cur_correct.get(cat, 0))
        bt = int(base_totals.get(cat, ct))
        bc = int(base_correct.get(cat, cc))
        cur_acc = (cc / ct) if ct else 0.0
        base_acc = (bc / bt) if bt else 0.0
        rows.append(
            {
                "category": cat,
                "current_correct": cc,
                "current_total": ct,
                "current_acc": cur_acc,
                "baseline_correct": bc,
                "baseline_total": bt,
                "baseline_acc": base_acc,
                "delta": cur_acc - base_acc,
            }
        )
    return rows


def _regressed_rows(run: RunResult, store: Any, baseline: dict | None) -> list[dict]:
    if not baseline:
        return []
    baseline_preds = store.get_predictions(baseline.get("run_id", ""))
    base_by_id = { (p.get("sample_id") if isinstance(p, dict) else getattr(p, "sample_id", None)): p for p in baseline_preds }
    cur_by_id = {}
    for p in (getattr(run, "predictions", None) or []):
        sid = p.get("sample_id") if isinstance(p, dict) else getattr(p, "sample_id", None)
        cur_by_id[sid] = p
    out = []
    for sid in (getattr(run, "verdict", None) and [] ) or []:
        pass
    # prefer verdict regressions if available later; caller can pass via kwargs
    def _g(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    reg_ids = []
    for sid, cur in cur_by_id.items():
        base = base_by_id.get(sid)
        if base is not None and bool(_g(base, "correct")) and not bool(_g(cur, "correct")):
            reg_ids.append(sid)
    rows = []
    for sid in reg_ids:
        cur = cur_by_id[sid]
        base = base_by_id[sid]
        snippet, tags = _load_sample_text(sid)
        rows.append(
            {
                "id": sid,
                "tags": tags,
                "snippet": _snippet_text(snippet),
                "ground_truth": _g(base, "expected", _g(cur, "expected", "")),
                "baseline_output": f"{_g(base, 'category')} — {_g(base, 'summary', '')}",
                "current_output": f"{_g(cur, 'category')} — {_g(cur, 'summary', '')}",
            }
        )
    return rows


def _trend_rows(store: Any, last_n: int) -> list[dict]:
    try:
        runs = store.list_runs()
    except Exception:
        return []
    runs = sorted(runs, key=lambda r: r.get("created_at", ""))
    runs = runs[-last_n:]
    out = []
    for r in runs:
        m = r.get("metrics") or {}
        lat = m.get("latency") or {}
        judge = m.get("judge") or {}
        toks = m.get("tokens") or {}
        out.append(
            {
                "run_id": r.get("run_id"),
                "created_at": r.get("created_at"),
                "provider": r.get("provider"),
                "prompt_version": r.get("prompt_version"),
                "accuracy": float(r.get("accuracy", 0.0)),
                "p50": lat.get("p50_ms"),
                "p90": lat.get("p90_ms"),
                "tokens": toks.get("total_tokens", 0),
                "judge": judge.get("mean_score"),
            }
        )
    return out


def _render_html(
    run: RunResult,
    verdict: Verdict | None,
    store: Any,
    baseline: dict | None,
    baseline_key: str,
    regressed_rows: list[dict],
    last_n: int,
) -> str:
    status = verdict.status if verdict else "n/a"
    status_cls = _status_class(status)
    status_emoji = _status_emoji(status)
    baseline_acc = verdict.baseline_accuracy if verdict else (baseline["accuracy"] if baseline else None)
    delta = verdict.accuracy_delta if verdict else None
    cur_acc = run.accuracy
    m = run.metrics
    lat = m.latency
    p50 = lat.p50_ms if lat else None
    p90 = lat.p90_ms if lat else None
    tokens = m.tokens.total_tokens if m.tokens else 0
    judge_mean = m.judge.mean_score if m.judge else None
    n_regressions = len(verdict.regressions) if verdict else len(regressed_rows)
    reg_ids = (verdict.regressions if verdict else [r["id"] for r in regressed_rows])

    meta_rows = [
        ("Run ID", run.run_id),
        ("Provider", run.provider),
        ("Prompt version", run.prompt_version),
        ("Dataset version", run.dataset_version),
        ("Baseline key", baseline_key),
        ("Executed at (UTC)", run.created_at),
        ("Run duration", f"{m.run_duration_ms:.0f} ms" if m.run_duration_ms else "n/a"),
        ("Samples", str(run.n_samples)),
    ]

    cat_rows = _per_category_rows(run, baseline["metrics"] if baseline else None)
    trend = _trend_rows(store, last_n)

    # ---------- HTML ----------
    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>MRDS Report — {_safe(run.run_id)}</title>")
    parts.append("<style>")
    parts.append(
        """
:root {
  --color-primary: #1E40AF; --color-secondary: #3B82F6; --color-accent: #D97706;
  --color-bg: #F8FAFC; --color-fg: #1E3A8A; --color-card: #FFFFFF;
  --color-muted: #E9EEF6; --color-muted-fg: #475569; --color-border: #DBEAFE;
  --color-destructive: #DC2626; --color-success: #16A34A; --color-warning: #D97706;
}
* { box-sizing: border-box; }
body { margin:0; font-family: "Fira Sans", system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  background: var(--color-bg); color: var(--color-fg); line-height:1.5; }
code, .mono { font-family: "Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
h1 { margin:0 0 4px; font-size: 1.6rem; color: var(--color-primary); }
.sub { color: var(--color-muted-fg); font-size: .9rem; margin-bottom: 16px; }
.banner { padding: 16px 20px; border-radius: 10px; margin: 16px 0; border: 1px solid var(--color-border);
  display:flex; align-items:center; gap:12px; }
.banner.pass { background:#ECFDF5; border-color:#BBF7D0; }
.banner.warn { background:#FFFBEB; border-color:#FDE68A; }
.banner.critical { background:#FEF2F2; border-color:#FECACA; }
.banner.neutral { background: var(--color-muted); }
.banner .emoji { font-size: 1.6rem; }
.banner .title { font-weight:700; font-size:1.1rem; }
.grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin: 16px 0; }
.card { background: var(--color-card); border:1px solid var(--color-border); border-radius:10px; padding:14px 16px; }
.card .label { color: var(--color-muted-fg); font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }
.card .value { font-size:1.35rem; font-weight:700; margin-top:4px; color: var(--color-primary); }
.card .value.crit { color: var(--color-destructive); }
table { width:100%; border-collapse: collapse; background: var(--color-card); border:1px solid var(--color-border);
  border-radius: 8px; overflow:hidden; margin: 12px 0; }
th, td { padding: 8px 10px; border-bottom: 1px solid var(--color-border); text-align:left; font-size:.88rem; vertical-align:top; }
th { background: var(--color-muted); color: var(--color-fg); font-weight:600; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #F1F5F9; }
.pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:.75rem; font-weight:600; }
.pill.pass { background:#DCFCE7; color:#166534; }
.pill.warn { background:#FEF3C7; color:#92400E; }
.pill.critical { background:#FEE2E2; color:#991B1B; }
.muted { color: var(--color-muted-fg); }
.section { margin-top: 24px; }
.section h2 { font-size:1.15rem; color: var(--color-primary); margin: 0 0 8px; border-bottom:2px solid var(--color-border); padding-bottom:6px; }
.kv { display:grid; grid-template-columns: 180px 1fr; gap:6px 12px; }
.kv div:nth-child(odd) { color: var(--color-muted-fg); font-weight:600; }
.snippet { max-width: 420px; }
.footer { margin-top:32px; color: var(--color-muted-fg); font-size:.75rem; text-align:center; }
@media (max-width: 720px) { .grid { grid-template-columns: 1fr 1fr; } table { font-size:.8rem; } }
"""
    )
    parts.append("</style></head><body><div class=\"container\">")

    parts.append(f"<h1>MRDS Regression Report</h1>")
    parts.append(f'<div class="sub">Model Regression Detection System — {_safe(run.provider)} / {_safe(run.prompt_version)}</div>')

    # status banner
    parts.append(f'<div class="banner {status_cls}">')
    parts.append(f'<span class="emoji">{status_emoji}</span>')
    parts.append(f'<div><div class="title">{_safe(status)}</div>')
    if verdict is not None:
        parts.append(
            f'<div class="muted">{_safe(verdict.details[0] if verdict.details else "")}</div>'
        )
    parts.append("</div></div>")

    # headline scorecard
    parts.append('<div class="section"><h2>Headline Scorecard</h2><div class="grid">')
    scorecard = [
        ("Status", status, status_cls),
        ("Current accuracy", _fmt_pct(cur_acc), "crit" if (delta or 0) < 0 else ""),
        ("Baseline accuracy", _fmt_pct(baseline_acc), ""),
        ("Pass-rate delta", f"{delta:+.4f}" if delta is not None else "n/a",
         "crit" if (delta or 0) < 0 else ""),
        ("Regressions", str(n_regressions), "crit" if n_regressions else ""),
        ("Mean judge score", _fmt_num(judge_mean, 2), ""),
        ("Latency p50", f"{_fmt_num(p50)} ms", ""),
        ("Latency p90", f"{_fmt_num(p90)} ms", ""),
        ("Token consumption", str(tokens), ""),
    ]
    for label, value, cls in scorecard:
        parts.append(
            f'<div class="card"><div class="label">{_safe(label)}</div>'
            f'<div class="value {cls}">{_safe(value)}</div></div>'
        )
    parts.append("</div></div>")

    # run metadata
    parts.append('<div class="section"><h2>Run Metadata</h2><div class="kv">')
    for k, v in meta_rows:
        parts.append(f"<div>{_safe(k)}</div><div class=\"mono\">{_safe(v)}</div>")
    parts.append("</div></div>")

    # per-category breakdown
    parts.append('<div class="section"><h2>Per-Category Breakdown</h2>')
    parts.append("<table><thead><tr><th>Category</th><th>Baseline</th><th>Current</th><th>Delta</th></tr></thead><tbody>")
    for r in cat_rows:
        parts.append(
            "<tr>"
            f"<td>{_safe(r['category'])}</td>"
            f"<td>{r['baseline_correct']}/{r['baseline_total']} ({_fmt_pct(r['baseline_acc'])})</td>"
            f"<td>{r['current_correct']}/{r['current_total']} ({_fmt_pct(r['current_acc'])})</td>"
            f"<td>{r['delta']:+.4f}</td>"
            "</tr>"
        )
    if not cat_rows:
        parts.append('<tr><td colspan="4" class="muted">No category data.</td></tr>')
    parts.append("</tbody></table></div>")

    # regressed cases
    parts.append('<div class="section"><h2>Side-by-Side Regressed Cases</h2>')
    if regressed_rows:
        parts.append(
            "<table><thead><tr>"
            "<th>Case ID</th><th>Tags</th><th>Email snippet</th><th>Ground truth</th>"
            "<th>Baseline output</th><th>New output</th></tr></thead><tbody>"
        )
        for r in regressed_rows:
            tags = ", ".join(r["tags"]) if r["tags"] else "—"
            parts.append(
                "<tr>"
                f"<td class=\"mono\">{_safe(r['id'])}</td>"
                f"<td>{_safe(tags)}</td>"
                f"<td class=\"snippet\">{_safe(r['snippet'])}</td>"
                f"<td>{_safe(r['ground_truth'])}</td>"
                f"<td>{_safe(r['baseline_output'])}</td>"
                f"<td>{_safe(r['current_output'])}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
    else:
        parts.append('<p class="muted">No PASS→FAIL flips in this run.</p>')
    if reg_ids and not regressed_rows:
        parts.append(f'<p class="muted">Regression ids: {_safe(", ".join(reg_ids))}</p>')
    parts.append("</div>")

    # trend
    parts.append('<div class="section"><h2>Trend (last {} runs)</h2>'.format(last_n))
    parts.append(
        "<table><thead><tr><th>Run</th><th>When</th><th>Provider</th><th>Prompt</th>"
        "<th>Accuracy</th><th>p50 ms</th><th>p90 ms</th><th>Tokens</th><th>Judge</th></tr></thead><tbody>"
    )
    for t in trend:
        parts.append(
            "<tr>"
            f"<td class=\"mono\">{_safe(t['run_id'])}</td>"
            f"<td>{_safe(t['created_at'])}</td>"
            f"<td>{_safe(t['provider'])}</td>"
            f"<td>{_safe(t['prompt_version'])}</td>"
            f"<td>{_fmt_pct(t['accuracy'])}</td>"
            f"<td>{_fmt_num(t['p50'])}</td>"
            f"<td>{_fmt_num(t['p90'])}</td>"
            f"<td>{_safe(t['tokens'])}</td>"
            f"<td>{_fmt_num(t['judge'], 2)}</td>"
            "</tr>"
        )
    if not trend:
        parts.append('<tr><td colspan="9" class="muted">No historical runs.</td></tr>')
    parts.append("</tbody></table></div>")

    parts.append('<div class="footer">Generated by MRDS reporter.py — palette: Data-Dense Dashboard (ui-ux-pro-max).</div>')
    parts.append("</div></body></html>")
    return "\n".join(parts)


def generate_report(run=None, verdict=None, store=None, output_path=None,
                    baseline_key=None, last_n=DEFAULT_LAST_N) -> str:
    """Render a standalone HTML report to disk and return its path (str)."""
    st = _as_store(store)
    rr = _as_run(run, st)
    vv = _as_verdict(verdict)
    bk = baseline_key if baseline_key is not None else default_baseline_key(rr)
    baseline = None
    try:
        baseline = st.get_baseline(bk)
    except Exception:
        baseline = None
    regressed_rows = _regressed_rows(rr, st, baseline)

    html_text = _render_html(rr, vv, st, baseline, bk, regressed_rows, last_n)

    out = _resolve_output_path(rr, output_path=output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    return str(out)


class Reporter:
    def __init__(self, store=None, output_dir=None) -> None:
        self.store = _as_store(store)
        self.output_dir = output_dir

    def generate(self, *a, **kw) -> str:
        if self.store is not None and "store" not in kw:
            kw["store"] = self.store
        if self.output_dir is not None and "output_dir" not in kw:
            kw["output_dir"] = self.output_dir
        return generate_report(*a, **kw)