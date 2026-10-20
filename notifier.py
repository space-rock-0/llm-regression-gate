"""notifier.py — Slack alerting engine (live webhook or MockNotifier fallback)."""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluator import RunResult, Verdict, default_baseline_key
from storage import get_store, SQLiteStore

def _as_store(store: Any = None) -> Any:
    if store is None:
        return get_store()
    if isinstance(store, (str, Path)):
        return SQLiteStore(store)
    return store

def _as_run(run: Any, store: Any = None) -> RunResult | None:
    if run is None:
        return None
    if isinstance(run, RunResult):
        return run
    if isinstance(run, str):
        try:
            st = _as_store(store)
            row = st.get_run(run)
            if row is not None:
                rr = RunResult.model_validate(row)
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
                return rr
        except Exception:
            return None
        return None
    if isinstance(run, dict):
        try:
            return RunResult.model_validate(run)
        except Exception:
            return None
    if hasattr(run, "model_dump"):
        try:
            return RunResult.model_validate(run.model_dump())
        except Exception:
            return None
    return None

def _as_verdict(verdict: Any) -> Verdict | None:
    if verdict is None:
        return None
    if isinstance(verdict, Verdict):
        return verdict
    if isinstance(verdict, dict):
        try:
            return Verdict.model_validate(verdict)
        except Exception:
            return None
    if hasattr(verdict, "model_dump"):
        try:
            return Verdict.model_validate(verdict.model_dump())
        except Exception:
            return None
    return None

def _metrics(run: RunResult | None) -> Any:
    return getattr(run, "metrics", None) if run is not None else None

def _latency(run: RunResult | None):
    m = _metrics(run)
    return getattr(m, "latency", None) if m is not None else None

def _status_header(verdict: Verdict | None, drift: Any = None) -> str:
    status = (getattr(verdict, "status", None) or "").upper() if verdict else ""
    delta = getattr(verdict, "accuracy_delta", None) if verdict else None
    drop_pct = abs((delta or 0.0) * 100) if delta is not None else 0.0
    if status == "PASS":
        return "🟢 PASS"
    if status == "WARN":
        return f"⚠️ WARNING: {drop_pct:.1f}% drop"
    if "CRITICAL" in status:
        return f"🔴 CRITICAL REGRESSION: {drop_pct:.1f}% drop"
    if "BASELINE" in status:
        return "🆕 BASELINE CREATED"
    if drift is not None and getattr(drift, "alert", False):
        return "⚠️ SLOW DRIFT WARNING"
    return "ℹ️ MRDS ALERT"

def _headline(run: RunResult | None, verdict: Verdict | None) -> str:
    if verdict is None:
        if run is not None:
            return f"Run {run.run_id}: accuracy {run.accuracy * 100:.0f}%."
        return "MRDS notification."
    n_reg = len(getattr(verdict, "regressions", []) or [])
    cur = getattr(verdict, "current_accuracy", None)
    base = getattr(verdict, "baseline_accuracy", None)
    if base is None:
        return f"Baseline created at {(cur or 0) * 100:.0f}% accuracy."
    if cur is None:
        return f"{n_reg} regressions detected."
    drop = (base - cur) * 100
    if drop >= 0:
        return f"{n_reg} regressions detected. Accuracy dropped from {base * 100:.0f}% to {cur * 100:.0f}%."
    return f"{n_reg} regressions detected. Accuracy changed from {base * 100:.0f}% to {cur * 100:.0f}%."

def build_slack_payload(
    run: Any = None,
    verdict: Any = None,
    report_path: Any = None,
    drift: Any = None,
    store: Any = None,
    result: Any = None,
    run_result: Any = None,
    run_obj: Any = None,
    verdict_result: Any = None,
    **kwargs: Any,
) -> dict:
    """Build a Slack Block Kit payload (dict) for a run/verdict."""
    if run is None:
        run = result if result is not None else (run_result if run_result is not None else run_obj)
    if verdict is None and verdict_result is not None:
        verdict = verdict_result
    rr = _as_run(run, store) if not isinstance(run, RunResult) else run
    # allow run to be omitted and recovered from verdict? no
    vv = _as_verdict(verdict)
    header = _status_header(vv, drift)
    headline = _headline(rr, vv)

    lat = _latency(rr)
    p50 = getattr(lat, "p50_ms", None) if lat is not None else None
    p90 = getattr(lat, "p90_ms", None) if lat is not None else None
    m = _metrics(rr)
    tokens = getattr(getattr(m, "tokens", None), "total_tokens", 0) if m is not None else 0
    judge_mean = getattr(getattr(m, "judge", None), "mean_score", None) if m is not None else None

    fields = []
    if vv is not None:
        fields.append({"type": "mrkdwn", "text": f"*Status:*\n{vv.status}"})
        if vv.baseline_accuracy is not None:
            fields.append({"type": "mrkdwn", "text": f"*Baseline acc:*\n{vv.baseline_accuracy * 100:.1f}%"})
        fields.append({"type": "mrkdwn", "text": f"*Current acc:*\n{vv.current_accuracy * 100:.1f}%"})
        if vv.accuracy_delta is not None:
            fields.append({"type": "mrkdwn", "text": f"*Delta:*\n{vv.accuracy_delta:+.4f}"})
        fields.append({"type": "mrkdwn", "text": f"*Regressions:*\n{len(vv.regressions or [])}"})
    fields.append({"type": "mrkdwn", "text": f"*Latency p50:*\n{(p50 if p50 is not None else 0):.1f} ms" if isinstance(p50, (int, float)) else f"*Latency p50:*\n{p50}"})
    fields.append({"type": "mrkdwn", "text": f"*Latency p90:*\n{(p90 if p90 is not None else 0):.1f} ms" if isinstance(p90, (int, float)) else f"*Latency p90:*\n{p90}"})
    fields.append({"type": "mrkdwn", "text": f"*Tokens:*\n{tokens}"})
    fields.append({"type": "mrkdwn", "text": f"*Judge score:*\n{judge_mean if judge_mean is not None else 'n/a'}"})

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": header, "emoji": True}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{headline}*"}},
        {"type": "section", "fields": fields},
    ]
    if vv is not None and (vv.regressions or []):
        ids = ", ".join(vv.regressions[:10])
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Regressed cases:* {ids}"}})
    if report_path is not None:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Report:* {report_path}"},
            }
        )
    if drift is not None and getattr(drift, "alert", False):
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Drift:* {getattr(drift, 'message', 'SLOW DRIFT WARNING')}",
                },
            }
        )

    payload = {
        "text": header if not headline else f"{header} — {headline}",
        "blocks": blocks,
    }
    # convenience aliases
    payload["header"] = header
    payload["headline"] = headline
    payload["status"] = vv.status if vv else None
    payload["report_path"] = report_path
    return payload

class MockNotifier:
    def __init__(self, log_path: Any = None, path: Any = None, file: Any = None, console: bool = True, **kw: Any) -> None:
        self.log_path = log_path if log_path is not None else path
        if self.log_path is None:
            self.log_path = file
        if self.log_path is None and "output_path" in kw:
            self.log_path = kw.get("output_path")
        self.console = console
        self.payloads: list[dict] = []
        self.emitted: list[dict] = []
        self.messages: list[str] = []
        self.logs: list[str] = []
        self.records: list[dict] = []
        self.history: list[dict] = []
        # default log file
        if self.log_path is None:
            self.log_path = Path("data") / "mock_slack.log"

    def _log(self, payload: dict) -> None:
        text = json.dumps(payload, ensure_ascii=False)
        self.logs.append(text)
        self.messages.append(payload.get("text", ""))
        try:
            p = Path(self.log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception:
            pass
        if self.console:
            print(f"[MockNotifier] {payload.get('text', '')}")

    def send(self, payload: dict) -> dict:
        self.payloads.append(payload)
        self.emitted.append(payload)
        self.records.append(payload)
        self.history.append(payload)
        self._log(payload)
        return {"ok": True, "mock": True, "payload": payload}

    def notify(self, run: Any = None, verdict: Any = None, report_path: Any = None, drift: Any = None, store: Any = None, **kw: Any) -> dict:
        payload = build_slack_payload(run=run, verdict=verdict, report_path=report_path, drift=drift, store=store, **kw)
        return self.send(payload)

    # aliases

class SlackNotifier:
    def __init__(self, webhook_url: str | None = None, url: str | None = None, timeout: int = 10, **kw: Any) -> None:
        self.webhook_url = webhook_url or url or os.environ.get("SLACK_WEBHOOK_URL")
        self.timeout = timeout
        self.payloads: list[dict] = []
        self.emitted: list[dict] = []
        self.messages: list[str] = []
        self.logs: list[str] = []
        self.records: list[dict] = []
        self.history: list[dict] = []

    def send(self, payload: dict) -> dict:
        self.payloads.append(payload)
        self.emitted.append(payload)
        self.records.append(payload)
        self.history.append(payload)
        self.messages.append(payload.get("text", ""))
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return {"ok": True, "status": getattr(resp, "status", 200), "body": body}

    def notify(self, run: Any = None, verdict: Any = None, report_path: Any = None, drift: Any = None, store: Any = None, **kw: Any) -> dict:
        payload = build_slack_payload(run=run, verdict=verdict, report_path=report_path, drift=drift, store=store, **kw)
        return self.send(payload)

def get_notifier(webhook_url: Any = None, **kw: Any) -> Any:
    url = webhook_url
    if url is None:
        url = kw.pop("url", None)
    if url is None:
        url = os.environ.get("SLACK_WEBHOOK_URL")
    if url:
        return SlackNotifier(webhook_url=url, **kw)
    return MockNotifier(**kw)

