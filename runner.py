"""
runner.py — async batch execution engine (Phase 3).

Why async at all: the golden set is 167 records. At ~2.6 s per live call a
serial run spends ~7 minutes doing nothing but waiting, and a judge pass would
double it. The model server handles requests concurrently, so the runner sends
several at once and the wall-clock time collapses toward
`n / concurrency x per-call latency`.

Why a Semaphore, not free-for-all: a local Ollama server has finite workers.
Open all 167 sockets at once and you queue anyway - just with more chance of
timeouts, connection resets, and a wedged GPU process. The semaphore is the
knob: bounded parallelism, tunable per machine (--concurrency).

Two transports, one contract:
    AsyncHTTPChatClient  httpx POST to an OpenAI-compatible /chat/completions
                         endpoint (Ollama, OpenAI). This is the path that can
                         read `usage` - real prompt/completion token counts -
                         because it sees the raw JSON response.
    AsyncWrappedClient   any sync provider (mock, Anthropic, Gemini SDKs)
                         pushed through asyncio.to_thread, so a blocking call
                         never freezes the event loop. Reports 0 tokens: the
                         sync seam returns text only, and 0 is the honest
                         number for "usage not reported".

Both build prompts through PromptConfig and validate through
config.parse_classification - the SAME contract the production classifier
uses. The only thing that differs between paths is the transport. A parity
test in test_mrds.py pins that down.

Usage:
    py runner.py                              # full golden set, ollama, judge on
    py runner.py --concurrency 8 --limit 20   # smoke test
    py runner.py --set-baseline --force       # (re)pin this run as the bar
    py runner.py --no-judge --dry-run         # measure only, store nothing
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).parent))

from contract import EmailInput, PromptConfig  # noqa: E402
from evaluator import (  # noqa: E402
    BASELINE_KEY_FORMAT,
    CaseResult,
    assemble_run,
    compare_to_baseline,
    judge_summary_async,
    load_golden_dataset,
)
from prompt_registry import load_prompt_config  # noqa: E402
from providers import get_client  # noqa: E402
from storage import get_store  # noqa: E402


# --------------------------------------------------------------------------
# Completion payload: text + the usage numbers the HTTP API reports
# --------------------------------------------------------------------------


class Completion:
    """What one async call returns: the model's text and - when the transport
    can see it - its token usage."""

    __slots__ = ("text", "prompt_tokens", "completion_tokens")

    def __init__(
        self,
        text: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class AsyncChatClient(Protocol):
    """Anything that can turn (system, user) text into text, asynchronously."""

    name: str

    async def acomplete(self, system_prompt: str, user_prompt: str) -> Completion:
        ...


# --------------------------------------------------------------------------
# Transport 1: httpx against an OpenAI-compatible HTTP API
# --------------------------------------------------------------------------


class AsyncHTTPChatClient:
    """Async LLM over HTTP. No SDK - httpx speaks the same wire format, and
    seeing the raw response is exactly what token accounting requires."""

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str = "local",
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        import httpx  # lazy: only HTTP providers need it

        self.name = name
        self._model = model
        self._temperature = temperature
        # Full URL built explicitly: RFC 3986 joining against a base WITHOUT a
        # trailing slash would silently replace the "/v1" segment.
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def acomplete(self, system_prompt: str, user_prompt: str) -> Completion:
        response = await self._client.post(
            self._url,
            json={
                "model": self._model,
                "temperature": self._temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        return Completion(
            text=data["choices"][0]["message"].get("content") or "",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------
# Transport 2: sync providers, off the event loop
# --------------------------------------------------------------------------


class AsyncWrappedClient:
    """Run a blocking ChatClient on a worker thread so one slow call cannot
    stall every other in-flight case. Token counts stay 0 - the sync seam does
    not report them, and inventing an estimate would corrupt the metric."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.name = getattr(inner, "name", "unknown")

    async def acomplete(self, system_prompt: str, user_prompt: str) -> Completion:
        text = await asyncio.to_thread(self._inner.complete, system_prompt, user_prompt)
        return Completion(text=text)


# --------------------------------------------------------------------------
# Factory - mirrors providers.get_client's env-var resolution
# --------------------------------------------------------------------------


def get_async_client(
    provider: str | None = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> AsyncChatClient:
    """
    Build the async client. Same resolution order as providers.get_client
    (explicit arg > PROVIDER > USE_MOCK > mock), so a run's provider is decided
    the same way whether it goes through the sync or the async path.
    """
    name = (provider or os.environ.get("PROVIDER") or "").strip().lower()
    if not name:
        use_mock = (os.environ.get("USE_MOCK", "true") or "true").strip().lower()
        name = "mock" if use_mock not in ("false", "0", "no") else "openai"

    if name not in {"mock", "ollama", "openai", "anthropic", "gemini"}:
        raise ValueError(
            f"Unknown provider {name!r}. Choose one of: "
            "['anthropic', 'gemini', 'mock', 'ollama', 'openai']"
        )

    if name == "ollama":
        return AsyncHTTPChatClient(
            name="ollama",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            api_key="ollama",
            temperature=temperature,
            timeout=timeout,
        )
    if name == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Provider 'openai' needs OPENAI_API_KEY. Set it, or run with "
                "--provider mock / --provider ollama."
            )
        return AsyncHTTPChatClient(
            name="openai",
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
        )
    # mock (offline), anthropic, gemini: sync SDKs -> worker threads.
    return AsyncWrappedClient(get_client(name))


# --------------------------------------------------------------------------
# The batch engine
# --------------------------------------------------------------------------


async def run_batch(
    rows: list[dict],
    config: PromptConfig,
    client: AsyncChatClient,
    concurrency: int = 4,
    judge: bool = True,
    judge_client: AsyncChatClient | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[CaseResult], float]:
    """
    Classify (and judge) every row concurrently, bounded by a semaphore.

    Returns (case_results, wall_clock_ms) ready for evaluator.assemble_run -
    this function MEASURES, it does not score; scoring lives in the evaluator
    so the sync and async paths cannot drift apart.

    Concurrency: one asyncio.Semaphore shared by all tasks. A task acquires it
    BEFORE the model call and releases after, so at most `concurrency` requests
    are ever in flight - the guard against rate limits and against starving a
    local Ollama server of its workers.

    Failure policy: exceptions propagate (asyncio.gather default). One
    contract violation aborts the run loudly - a half-scored run is not a run,
    and a silent partial result would compare against the baseline as if it
    were complete. Earlier completed runs are already stored, so a crash costs
    at most the run in flight.

    Args:
        judge:         score summary relevance 1-5 per case (default on - this
                       is the Phase 3 path the spec asks for).
        judge_client:  grade with a different model than the one under test.
                       Defaults to the same client (note: self-grading bias).
        on_progress:   optional callback(done, total) as cases finish.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))
    grading_client = judge_client or client
    total = len(rows)
    finished = 0

    async def _one(row: dict) -> CaseResult:
        nonlocal finished
        async with semaphore:
            email = EmailInput(
                body=row["body"],
                subject=row.get("subject", ""),
                email_id=row["id"],
            )
            started = time.perf_counter()
            completion = await client.acomplete(
                config.build_system_prompt(),
                config.build_user_prompt(email),
            )
            latency_ms = (time.perf_counter() - started) * 1000

            # Single validation boundary - same call production uses.
            parsed = config.parse_classification(
                completion.text, provider=getattr(client, "name", "unknown")
            )

            outcome = await judge_summary_async(
                parsed.summary,
                row.get("expected_summary"),
                row["body"],
                client=grading_client,
                enabled=judge,
            )

            if on_progress is not None:
                finished += 1
                on_progress(finished, total)

            return CaseResult(
                sample_id=row["id"],
                expected=row["expected"],
                category=parsed.category,
                summary=parsed.summary,
                latency_ms=latency_ms,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                judge=outcome,
            )

    started = time.perf_counter()
    cases = await asyncio.gather(*(_one(row) for row in rows))
    duration_ms = (time.perf_counter() - started) * 1000
    return list(cases), duration_ms


async def _aclose(client) -> None:
    """Close the HTTP pool when the transport has one. Safe to call on wraps."""
    closer = getattr(client, "aclose", None)
    if closer is not None:
        await closer()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the golden set asynchronously, score it, compare to baseline."
    )
    ap.add_argument("--prompt", default="v1", help="prompt FILE version (default v1)")
    ap.add_argument("--provider", default="ollama", help="provider (default ollama)")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="max in-flight model calls (default 4; lower it if the "
                         "local server struggles, raise it if calls are fast)")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N records (smoke test - never "
                         "baseline on this)")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the summary-relevance judge (faster, no judge tokens)")
    ap.add_argument("--set-baseline", action="store_true",
                    help="write this run as the baseline for its key (overwrites "
                         "only with --force)")
    ap.add_argument("--force", action="store_true",
                    help="allow --set-baseline to replace an existing baseline")
    ap.add_argument("--dry-run", action="store_true",
                    help="run + score + report, store nothing, no comparison")
    args = ap.parse_args(argv)

    # Refuses an unverified dataset on its own - same gate as baseline.py.
    rows, dataset_version = load_golden_dataset()
    if args.limit:
        rows = rows[: args.limit]
        print(f"NOTE: --limit {args.limit} - scoring a subset. Fine for smoke tests,")
        print("      never a basis for a baseline or threshold.")
        print()

    config = load_prompt_config(args.prompt)
    client = get_async_client(args.provider, temperature=config.spec.temperature)
    # The baseline key is fully known BEFORE spending a single model call:
    # provider name + prompt file version + dataset version.
    key = BASELINE_KEY_FORMAT.format(
        provider=getattr(client, "name", "unknown"),
        prompt_version=config.version,
        dataset_version=dataset_version,
    )
    store = None if args.dry_run else get_store()

    if store is not None and args.set_baseline and not args.force:
        existing = store.get_baseline(key)
        if existing:
            print(f"ABORT: a baseline already exists under {key} "
                  f"(pass rate {existing.get('accuracy')}).")
            print("       Re-run with --force to replace it. Baselines are not")
            print("       overwritten casually - everything since has judged")
            print("       itself against that number.")
            return 3

    print(f"dataset : {dataset_version}  ({len(rows)} records)")
    print(f"model   : {getattr(client, 'name', 'unknown')}  prompt: {config.version}")
    print(f"key     : {key}")
    print(f"concurrency: {args.concurrency}  judge: {'off' if args.no_judge else 'on'}"
          f"{'  (dry run - nothing stored)' if args.dry_run else ''}")
    print()

    def progress(finished: int, total: int) -> None:
        if finished % 20 == 0 or finished == total:
            print(f"  {finished}/{total} cases done")

    async def drive() -> tuple:
        try:
            cases, duration_ms = await run_batch(
                rows, config, client,
                concurrency=args.concurrency,
                judge=not args.no_judge,
                on_progress=progress,
            )
        finally:
            await _aclose(client)
        run = assemble_run(
            cases,
            provider=getattr(client, "name", "unknown"),
            prompt_version=config.version,
            dataset_version=dataset_version,
            duration_ms=duration_ms,
        )
        return run

    started = time.time()
    try:
        run = asyncio.run(drive())
    except Exception as exc:  # noqa: BLE001 - report the model's real failure
        print(f"RUN FAILED: {type(exc).__name__}: {str(exc)[:200]}")
        if store is not None:
            store.close()
        return 4

    if store is not None:
        # Persist BEFORE comparing: a crash during compare still leaves the raw
        # predictions on disk (same rule as run_evaluation).
        from evaluator import _as_row  # same-package helper, used by baseline.py too

        store.save_run(_as_row(run), [p.model_dump() for p in run.predictions])

    print()
    print(f"run     : {run.run_id}  {time.time() - started:.0f}s wall clock")
    print(f"pass    : {run.accuracy:.4f}  ({run.metrics.n_correct}/{run.n_samples})")
    lat = run.metrics.latency
    if lat:
        print(f"latency : p50 {lat.p50_ms:.0f} ms   p90 {lat.p90_ms:.0f} ms   "
              f"max {lat.max_ms:.0f} ms")
    tok = run.metrics.tokens
    print(f"tokens  : prompt {tok.prompt_tokens}  completion {tok.completion_tokens}  "
          f"total {tok.total_tokens}"
          + (f"  (+judge {tok.judge_total_tokens})" if tok.judge_total_tokens else ""))
    judge = run.metrics.judge
    if judge and judge.n_scored:
        print(f"judge   : mean {judge.mean_score}/5 over {judge.n_scored} cases "
              f"({judge.n_llm} llm, {judge.n_heuristic} heuristic fallback)")
    elif judge:
        print("judge   : disabled (--no-judge)")

    if store is None:
        print()
        print("dry run: nothing stored, no comparison made.")
        return 0

    verdict = compare_to_baseline(run, store=store, baseline_key=key)

    if args.set_baseline:
        from evaluator import _as_row

        store.set_baseline(key, _as_row(run))
        print(f"BASELINE SET: {key} = this run, pass rate {run.accuracy:.4f}")

    print()
    print(f"status  : {verdict.status}")
    if verdict.pass_rate_delta is not None:
        print(f"delta   : {verdict.pass_rate_delta:+.4f} "
              f"(warn < -{verdict.warn_threshold:.2f}, critical < -{verdict.threshold:.2f})")
    if verdict.per_category_delta:
        bits = "  ".join(
            f"{cat} {d:+.3f}" for cat, d in sorted(verdict.per_category_delta.items())
        )
        print(f"by cat  : {bits}")
    if verdict.regressions:
        print(f"regressed ({len(verdict.regressions)}): "
              + ", ".join(verdict.regressions[:10])
              + (" ..." if len(verdict.regressions) > 10 else ""))
    if verdict.improvements:
        print(f"improved ({len(verdict.improvements)}): "
              + ", ".join(verdict.improvements[:10])
              + (" ..." if len(verdict.improvements) > 10 else ""))

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
