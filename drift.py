"""drift.py — rolling drift detection over historical runs in SQLite."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from storage import get_store, SQLiteStore


def _as_store(store=None):
    if store is None:
        return get_store()
    if isinstance(store, (str, Path)):
        return SQLiteStore(store)
    return store


@dataclass
class DriftResult:
    moving_average: float | None = None
    alert: bool = False
    status: str = "OK"
    message: str = ""
    window: int = 7
    n_runs: int = 0
    threshold: float = -0.03
    baseline_moving_average: float | None = None
    delta: float | None = None
    moving_averages: list = field(default_factory=list)
    accuracies: list = field(default_factory=list)
    runs: list = field(default_factory=list)


def compute_moving_average(values, window=7):
    if not values:
        return None
    w = min(window, len(values))
    return sum(values[-w:]) / w


def rolling_averages(values, window=7):
    out = []
    for i in range(1, len(values) + 1):
        w = min(window, i)
        out.append(sum(values[i - w:i]) / w)
    return out


def detect_drift(store=None, window=7, threshold=-0.03, baseline_key=None,
                 provider=None, prompt_version=None, min_runs=1,
                 runs=None, accuracies=None):
    if accuracies is not None:
        accs = [float(x) for x in accuracies]
        run_rows = runs or []
    else:
        st = _as_store(store)
        run_rows = list(runs) if runs is not None else []
        if not run_rows:
            try:
                run_rows = st.list_runs()
            except Exception:
                run_rows = []
        if provider is not None:
            run_rows = [r for r in run_rows if r.get("provider") == provider]
        if prompt_version is not None:
            run_rows = [r for r in run_rows if r.get("prompt_version") == prompt_version]
        run_rows = sorted(run_rows, key=lambda r: r.get("created_at", ""))
        accs = [float(r.get("accuracy", 0.0)) for r in run_rows]

    mas = rolling_averages(accs, window)
    current_ma = mas[-1] if mas else None

    reference = None
    if baseline_key is not None:
        try:
            b = _as_store(store).get_baseline(baseline_key)
            if b is not None:
                reference = float(b.get("accuracy", b.get("metrics", {}).get("accuracy", 0.0)))
        except Exception:
            reference = None
    if reference is None and accs:
        reference = accs[0]
    if reference is None and mas:
        reference = mas[0]

    delta = (current_ma - reference) if (current_ma is not None and reference is not None) else None
    alert = bool(current_ma is not None and reference is not None and delta is not None
                 and len(accs) >= min_runs and delta <= float(threshold))
    status = "SLOW_DRIFT_WARNING" if alert else "OK"
    if alert:
        message = (f"SLOW DRIFT WARNING: {window}-run moving average {current_ma:.4f} "
                   f"dropped {abs(delta):.4f} vs reference {reference:.4f} "
                   f"(threshold {threshold:.4f}).")
    else:
        message = (f"No drift: {window}-run moving average "
                   f"{current_ma if current_ma is not None else 'n/a'} "
                   f"(reference {reference if reference is not None else 'n/a'}, "
                   f"threshold {threshold}).")

    return DriftResult(
        moving_average=current_ma, alert=alert, status=status, message=message,
        window=window, n_runs=len(accs), threshold=float(threshold),
        baseline_moving_average=reference, delta=delta,
        moving_averages=mas, accuracies=accs, runs=run_rows if not accuracies else (runs or []),
    )