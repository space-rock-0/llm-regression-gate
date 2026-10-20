"""
demo.py — verification walkthrough for the typed contract.

    py demo.py

1. valid round-trip: raw email -> typed input -> typed output -> dict -> SQLite
2. malformed model output REJECTED at the contract boundary
3. v1 and v2 both run under the contract, rows still separated in the DB
4. the swap seam: a completely different classifier implementation is evaluated
   without changing evaluator.py
"""

from __future__ import annotations

import json
from pathlib import Path

from classifier import classify_email
from contract import (
    ContractViolationError,
    EmailClassification,
    EmailInput,
)
from evaluator import (
    BASELINE_KEY_FORMAT,
    DEFAULT_DATASET_VERSION,
    compare_to_baseline,
    default_baseline_key,
    evaluate,
    run_evaluation,
)
from prompt_registry import PROMPTS_DIR, list_versions, load_prompt
from providers import get_client
from storage import SQLiteStore
from reporter import generate_report
from notifier import get_notifier
from drift import detect_drift

DATA_DIR = Path(__file__).parent / "data"
DEMO_DB = DATA_DIR / "demo.sqlite3"


class BrokenClient:
    """Fake client that always says 'general'. Stands in for a real regression."""

    name = "mock-broken"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps({"category": "general", "summary": "Customer needs help."})


class MalformedClient:
    """Returns something that is not the contract's output shape at all."""

    name = "mock-malformed"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "Sure! Hope this helps. Let me know what you think!"


class RefundSniffer:
    """
    A DIFFERENT classifier implementation - no LLM, no prompt, no provider.

    It exists to prove the seam: `evaluate(classify=...)` scores it with zero
    changes to evaluator.py. Same contract in, same dict out.
    """

    name = "keyword-sniffer"

    def __call__(self, email: EmailInput, prompt=None, client=None) -> dict:
        text = f"{email.subject} {email.body}".lower()
        if "refund" in text or "invoice" in text:
            category = "billing"
        elif "error" in text or "api" in text:
            category = "technical"
        elif "password" in text or "sign in" in text:
            category = "account"
        else:
            category = "general"
        typed = EmailClassification(
            category=category,
            summary=f"Keyword match on: {email.email_id}.",
            provider=self.name,
            prompt_version=prompt.version if prompt else "unknown",
            # v2 forbids `general`: the contract must know the version's labels.
            allowed_categories=prompt.allowed_categories if prompt else (),
        )
        return typed.model_dump()


def _banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def _dump_runs(store: SQLiteStore) -> None:
    """Print the actual run rows so version separation is visible, not asserted."""
    rows = store._conn.execute(
        "SELECT run_id, provider, prompt_ver, n_samples, accuracy "
        "FROM runs ORDER BY created_at, run_id"
    ).fetchall()
    print(f"{'run_id':<38} {'provider':<17} {'prompt_ver':<10} {'acc':>6}")
    for row in rows:
        print(
            f"{row['run_id']:<38} {row['provider']:<17} "
            f"{row['prompt_ver']:<10} {row['accuracy']:>6.3f}"
        )
    base = store._conn.execute("SELECT key, accuracy FROM baseline").fetchall()
    print("baselines: " + ", ".join(f"{b['key']}={b['accuracy']:.3f}" for b in base))


def main() -> None:
    client = get_client()  # mock by default
    print(f"provider in use : {client.name}")
    print(f"prompt directory: {PROMPTS_DIR}")
    print(f"prompt versions : {list_versions()}")

    # --- Step 1: valid contract round-trip -----------------------------
    _banner("STEP 1 - valid round-trip: email -> typed -> dict -> SQLite")
    store = SQLiteStore(DEMO_DB)
    _clear(store)

    typed_input = EmailInput(
        body="Hi, I was charged twice and I want a refund for the duplicate invoice.",
        subject="Charged twice",
        email_id="s1",
    )
    print(f"1. typed input   : {typed_input.email_id=} {typed_input.subject=}")

    out_dict = classify_email(typed_input, client=client, prompt_version="v1")
    print(f"2. dict out      : {json.dumps(out_dict)}")

    typed_back = EmailClassification(**out_dict)
    print(f"3. re-typed out  : {typed_back.category=} {typed_back.prompt_version=}")

    store.save_run(
        {
            "run_id": "run_manual_001", "created_at": "2026-10-18T00:00:00+00:00",
            "provider": out_dict["provider"], "prompt_version": out_dict["prompt_version"],
            "n_samples": 1, "accuracy": 1.0,
            "metrics": {"accuracy": 1.0, "n_samples": 1, "n_correct": 1,
                        "per_category_correct": {"billing": 1},
                        "per_category_total": {"billing": 1}, "confusion": {}},
        },
        [{"sample_id": "s1", "expected": "billing", "category": out_dict["category"],
          "correct": 1, "summary": out_dict["summary"]}],
    )
    stored = store.get_run("run_manual_001")
    print(f"4. in SQLite     : prompt_version={stored['prompt_version']} "
          f"accuracy={stored['accuracy']}")

    # --- Step 2: malformed output rejected at the boundary -------------
    _banner("STEP 2 - malformed model output is REJECTED at the boundary")
    print(f"model said      : {MalformedClient().complete('', '')!r}")
    try:
        classify_email("Please refund my invoice.", client=MalformedClient())
    except ContractViolationError as exc:
        print(f"REJECTED        : {type(exc).__name__}: {str(exc)[:90]}")

    class Rude:
        name = "mock-rude"

        def complete(self, s, u):
            return json.dumps({"category": "refund_request", "summary": "x"})

    try:
        classify_email("Please refund my invoice.", client=Rude())
    except ContractViolationError as exc:
        print(f"REJECTED        : {type(exc).__name__}: {str(exc)[:90]}")

    # --- Step 3: v1 + v2 under the contract ---------------------------
    _banner("STEP 3 - v1 and v2 under the contract, rows separated in the DB")
    run1, v1a = run_evaluation(
        client=client, store=store, prompt_version="v1", threshold=0.0
    )
    print(f"v1: {run1.accuracy:.4f} ({run1.metrics.n_correct}/{run1.n_samples}) "
          f"key={default_baseline_key(run1)} -> {v1a.status}")

    run2, v2a = run_evaluation(
        client=client, store=store, prompt_version="v2", threshold=0.0
    )
    print(f"v2: {run2.accuracy:.4f} ({run2.metrics.n_correct}/{run2.n_samples}) "
          f"key={default_baseline_key(run2)} -> {v2a.status}")
    print(f"v2 allowed labels: {list(load_prompt('v2').categories)} "
          f"(no 'general' - general mail lands in "
          f"{load_prompt('v2').categories[-1]})")
    _dump_runs(store)

    # --- Step 4: the swap seam ----------------------------------------
    _banner("STEP 4 - swap seam: different implementation, evaluator untouched")
    alt = RefundSniffer()
    run3, v3 = run_evaluation(
        store=store, classify=alt, samples=None, prompt_version="v1", threshold=0.0
    )
    print(f"impl           : {alt.name} (no LLM, no prompt, no provider)")
    print(f"accuracy       : {run3.accuracy:.4f} -> {v3.status}")
    print(f"baseline used  : {default_baseline_key(run3)}  "
          f"(the LLM client was still the mock, so this run is scored against "
          f"the mock baseline - that is the point: a weaker implementation shows "
          f"up as a CRITICAL_REGRESSION)")
    print("evaluator.py   : 0 lines changed to evaluate it")

    # --- Step 5: regression still detected ----------------------------
    _banner("STEP 5 - regression detection under the contract")
    bad = evaluate(client=BrokenClient(), prompt_version="v1")
    store.save_run(
        {
            "run_id": bad.run_id, "created_at": bad.created_at,
            "provider": bad.provider, "prompt_version": bad.prompt_version,
            "n_samples": bad.n_samples, "accuracy": bad.accuracy,
            "metrics": bad.metrics.model_dump(),
        },
        [p.model_dump() for p in bad.predictions],
    )
    # Baseline keys are "<provider>:<prompt_version>:<dataset_version>"; the
    # broken client reports provider "mock-broken", so pin the mock's v1 key
    # explicitly. Built from the format constant rather than typed as a literal,
    # so this line cannot rot the last time the key format changes.
    mock_v1_key = BASELINE_KEY_FORMAT.format(
        provider="mock",
        prompt_version="v1",
        dataset_version=DEFAULT_DATASET_VERSION,
    )
    verdict = compare_to_baseline(bad, store=store, baseline_key=mock_v1_key, threshold=0.0)
    print(f"accuracy : {bad.accuracy:.4f}  delta {verdict.accuracy_delta:+.4f}")
    print(f"status   : {verdict.status}")
    for line in verdict.details:
        print(f"  - {line}")

    # --- Step 6: Phase 4 alerting & reporting -------------------------
    _banner("STEP 6 - Phase 4: alerting, reporting and drift detection")
    report_path = generate_report(run2, v2a, store=store,
                                 output_path=DATA_DIR / "demo_report.html")
    print(f"html report : {report_path}")
    notifier = get_notifier()
    result = notifier.notify(run2, v2a, report_path=report_path)
    payload = result["payload"]
    print(f"slack alert : {payload['text']}")
    drift = detect_drift(store)
    print(f"drift       : {drift.status} (moving average "
          f"{drift.moving_average})")

    store.close()
    print(f"\ndatabase : {DEMO_DB}")


def _clear(store: SQLiteStore) -> None:
    """Start from an empty DB so the demo is repeatable."""
    with store._conn:
        for table in ("predictions", "runs", "baseline"):
            store._conn.execute(f"DELETE FROM {table}")


if __name__ == "__main__":
    main()