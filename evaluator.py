"""
evaluator.py — score the classifier, compare it to the baseline, emit a verdict.

This is where "regression detection" actually happens:

    run classifier on labelled samples  -> metrics
    compare metrics to stored baseline -> PASS | WARN | CRITICAL_REGRESSION

Determinism requirement: with the mock provider, the same samples must always
produce identical numbers. That is what makes a metric drop meaningful signal
rather than sampling noise.

What this module depends on: the contract (contract.py) and the call signature
`classify_email(EmailInput, ...) -> dict`. It does NOT know how classification
happens, which provider runs, or how prompts are stored - so swapping the
implementation later touches this file zero times. See decision.md D26.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from classifier import classify_email
from contract import CATEGORIES as VALID_CATEGORIES
from contract import EmailClassification, EmailInput, PromptConfig, PromptSpec
from providers import ChatClient, get_client
from storage import Store, get_store

SAMPLES_PATH = Path(__file__).parent / "samples" / "emails.json"

# --- regression thresholds (Phase 3) ---------------------------------------
# Configurable in ONE place: the module constants are the defaults every
# caller inherits, and both are function arguments too, so a caller can override
# without editing this file.
#
#   delta = current pass rate - baseline pass rate   (negative = got worse)
#   drop  = -delta                                   (positive = got worse)
#
#   drop <= WARN     -> PASS               ordinary noise
#   drop <= CRITICAL -> WARN               suspicious, look before shipping
#   drop >  CRITICAL -> CRITICAL_REGRESSION  do not ship
import os as _os

WARN_THRESHOLD = float(_os.getenv("WARN_THRESHOLD", "0.03"))   # 3 percentage points
CRITICAL_THRESHOLD = float(_os.getenv("CRITICAL_THRESHOLD", "0.08"))  # 8 percentage points

# Baseline keys are scoped per prompt version AND per dataset version. Two
# dimensions, both of which make a comparison meaningless if they drift:
#
#   provider         different model, different numbers
#   prompt_version   v2 is a different contract than v1, not a regression
#   dataset_version  the bar itself moved - same provider and prompt scored
#                    against a different set of emails is NOT a comparison
#
# Without dataset_version, editing the golden set silently repoints every old
# baseline at a new test set, and the resulting delta is reported as a model
# regression. Format: "<provider>:<prompt_version>:<dataset_version>".
BASELINE_KEY_FORMAT = "{provider}:{prompt_version}:{dataset_version}"

# Used when a run cannot say which dataset it scored. It is deliberately a
# value that matches no real dataset version, so a run with an unknown dataset
# can never collide with, or silently reuse, a real dataset's baseline.
DEFAULT_DATASET_VERSION = "unknown"


def default_baseline_key(run: RunResult) -> str:
    """The baseline key a run is measured against: provider, prompt FILE
    version, and dataset version. Any one of the three changing means the
    numbers are no longer comparable, so all three are in the key."""
    return BASELINE_KEY_FORMAT.format(
        provider=run.provider,
        prompt_version=run.prompt_version,
        dataset_version=run.dataset_version,
    )


def read_dataset_version(path: str | Path) -> str:
    """
    Read `dataset_version` out of a dataset JSON file.

    Returns DEFAULT_DATASET_VERSION when the file has none, so the built-in
    samples file (which is not versioned) keeps working unchanged.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_DATASET_VERSION
    if isinstance(data, dict) and isinstance(data.get("dataset_version"), str):
        return data["dataset_version"].strip() or DEFAULT_DATASET_VERSION
    return DEFAULT_DATASET_VERSION


# --------------------------------------------------------------------------
# Output payloads (Pydantic)
# --------------------------------------------------------------------------


class LatencyStats(BaseModel):
    """Per-request latency distribution, in milliseconds."""

    p50_ms: float = Field(default=0.0, description="Median: half of calls were faster.")
    p90_ms: float = Field(default=0.0, description="90% of calls were faster than this.")
    mean_ms: float = 0.0
    max_ms: float = 0.0
    n: int = 0


class TokenUsage(BaseModel):
    """Token accounting for one run. Zeroes are honest: they mean the transport
    did not report usage (the mock and the sync SDK seam do not), not that the
    model ran for free."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    judge_total_tokens: int = 0


class JudgeStats(BaseModel):
    """Summary-relevance scores (LLM-as-Judge) across a run."""

    mean_score: float | None = Field(
        default=None, description="Average 1-5; None when no case was judged."
    )
    n_scored: int = 0
    n_llm: int = Field(default=0, description="Scores produced by a live model.")
    n_heuristic: int = Field(
        default=0,
        description="Scores produced by the offline fallback (offline, mocked, "
        "or the judge returned unusable output).",
    )


class CaseMetrics(BaseModel):
    """Everything measured for ONE test case. Stored per case in SQLite
    (case_metrics table) / inside each prediction in the JSON backend."""

    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    judge_score: int | None = Field(default=None, ge=1, le=5)
    judge_source: str = Field(
        default="none", description="llm | heuristic | none (judge disabled)."
    )
    judge_latency_ms: float = 0.0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0


class Metrics(BaseModel):
    """Scores for one run. Flat and JSON-friendly so it can be stored as-is."""

    accuracy: float = Field(description="Fraction of samples classified correctly.")
    n_samples: int
    n_correct: int
    per_category_correct: dict[str, int] = Field(default_factory=dict)
    per_category_total: dict[str, int] = Field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="expected label -> predicted label -> count.",
    )
    # Phase 3 additions. All defaulted, so metrics rows written before Phase 3
    # still load without a migration.
    latency: LatencyStats | None = None
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    judge: JudgeStats | None = None
    run_duration_ms: float = Field(
        default=0.0, description="Wall-clock time of the whole run."
    )


class Prediction(BaseModel):
    sample_id: str
    expected: str
    category: str
    correct: bool
    summary: str
    case_metrics: CaseMetrics | None = None


class RunResult(BaseModel):
    """One evaluation run, fully described."""

    run_id: str
    created_at: str
    provider: str
    prompt_version: str
    dataset_version: str = Field(
        default=DEFAULT_DATASET_VERSION,
        description="Version of the labelled set this run scored. Part of the "
        "baseline key: a baseline taken on one dataset must never be compared "
        "against a run on another.",
    )
    n_samples: int
    accuracy: float
    metrics: Metrics
    predictions: list[Prediction] = Field(default_factory=list)


class Verdict(BaseModel):
    """The answer MRDS exists to produce."""

    status: str = Field(
        description="PASS | WARN | CRITICAL_REGRESSION | BASELINE_CREATED "
        "(threshold=... override collapses WARN/CRITICAL into one line)"
    )
    run_id: str
    provider: str
    prompt_version: str
    dataset_version: str = DEFAULT_DATASET_VERSION
    current_accuracy: float
    baseline_accuracy: float | None
    accuracy_delta: float | None
    threshold: float = Field(
        description="The CRITICAL line this verdict was judged by: the "
        "critical_threshold, or the legacy single `threshold` when given."
    )
    warn_threshold: float = WARN_THRESHOLD
    details: list[str] = Field(default_factory=list)
    regression_detected: bool = Field(
        default=False, description="True for WARN and CRITICAL_REGRESSION."
    )
    # -- Phase 3 diff ------------------------------------------------------
    # accuracy IS the pass rate (correct / total), so pass_rate_delta repeats
    # accuracy_delta under the name the spec asks for. One number, two names.
    pass_rate_delta: float | None = None
    per_category_delta: dict[str, float] = Field(
        default_factory=dict,
        description="category -> current accuracy - baseline accuracy.",
    )
    regressions: list[str] = Field(
        default_factory=list,
        description="sample ids that PASSED at baseline and FAIL now.",
    )
    improvements: list[str] = Field(
        default_factory=list,
        description="sample ids that FAILED at baseline and PASS now.",
    )


# --------------------------------------------------------------------------
# Load labelled samples
# --------------------------------------------------------------------------


def load_samples(path: str | Path | None = None) -> list[dict]:
    """
    Read the labelled sample emails from samples/emails.json.

    Kept in a file (not a Python list) so adding test cases needs no code change,
    and so the exact evaluation set is versionable and reviewable.
    """
    sample_path = Path(path) if path else SAMPLES_PATH
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    emails = data["emails"] if isinstance(data, dict) else data

    for item in emails:
        if item.get("expected") not in VALID_CATEGORIES:
            raise ValueError(
                f"Sample {item.get('id')!r} has an invalid label "
                f"{item.get('expected')!r}. Must be one of {VALID_CATEGORIES}."
            )
    return emails


# --------------------------------------------------------------------------
# Step 1: run the feature under test and score it
# --------------------------------------------------------------------------


GOLDEN_PATH = (
    Path(__file__).parent
    / "golden_dataset"
    / "golden_dataset_v0.4_reviewed.json"
)


def load_golden_dataset(
    path: str | Path | None = None,
    require_verified: bool = True,
) -> tuple[list[dict], str]:
    """
    Load the human-reviewed golden dataset into the sample shape eval expects.

    Returns `(rows, dataset_version)`. The rows are exactly the shape
    `samples/emails.json` uses, so the result drops straight into
    `evaluate(samples=rows, dataset_version=version)` with no adapter.

    Why a loader instead of reading the JSON by hand: the golden set carries
    provenance rules that a naive reader would skip. This enforces them:

      * `provenance.human_verified` must be true. A set that is still under
        review is not a bar, and baselining on it makes every later delta
        meaningless. Set `require_verified=False` to inspect an in-progress
        file - but never to create a baseline.
      * every record must carry a category from MRDS's closed label set and a
        non-empty ideal summary.
      * ids must be unique, and every row must construct a typed `EmailInput`,
        so a blank or malformed body fails here rather than mid-run.

    Subject handling: a record may keep its subject inline as a leading
    `Subj:` line (that is how it appears in the source corpus). When it does,
    that line is moved into `EmailInput.subject`. The split only ever REMOVES
    that one leading line, so no reviewed text is dropped.

    Raises:
        FileNotFoundError - no dataset at `path`
        ValueError - unverified provenance, bad category, duplicate id,
            missing summary, or a row that will not type as EmailInput
    """
    dataset_path = Path(path) if path else GOLDEN_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"golden dataset not found: {dataset_path}")

    doc = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "records" not in doc:
        raise ValueError(f"{dataset_path} is not a golden dataset (no 'records')")

    version = doc.get("dataset_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(
            f"{dataset_path} has no dataset_version. A dataset that cannot name "
            "itself cannot key a baseline."
        )

    if require_verified:
        provenance = doc.get("provenance") or {}
        if provenance.get("human_verified") is not True:
            raise ValueError(
                f"{dataset_path} is NOT human-verified "
                f"(status={doc.get('status')!r}, "
                f"human_verified={provenance.get('human_verified')!r}). "
                "Refusing to build a baseline on labels a human has not finished "
                "deciding - the metric would measure reviewer progress, not the "
                "classifier."
            )

    rows: list[dict] = []
    seen_ids: set[str] = set()

    for record in doc["records"]:
        rid = record.get("id")
        if not rid:
            raise ValueError("a golden record has no stable id")
        if rid in seen_ids:
            raise ValueError(f"duplicate golden record id {rid!r}")
        seen_ids.add(rid)

        category = record.get("human_verified_category")
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"golden record {rid!r} has invalid category {category!r}; "
                f"must be one of {list(VALID_CATEGORIES)}"
            )

        summary = record.get("human_verified_ideal_summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"golden record {rid!r} has no ideal summary")

        body = (record.get("input") or "").strip()
        if not body:
            raise ValueError(f"golden record {rid!r} has empty input")

        subject = (record.get("subject") or "").strip()
        if not subject:
            first, sep, rest = body.partition("\n")
            if first.strip().lower().startswith("subj:"):
                # Move the inline subject line into the typed subject field.
                subject = first.strip()[len("subj:"):].strip()
                body = rest.strip()

        # Typed now, so a row that cannot be constructed fails before a single
        # model call is made instead of halfway through an eval run.
        EmailInput(body=body, subject=subject, email_id=rid)

        rows.append({
            "id": rid,
            "expected": category,
            "subject": subject,
            "body": body,
            # The human's ideal summary rides along: the LLM-as-Judge score
            # compares the model's summary against THIS text. Plain sample
            # files without one simply omit the key (judge falls back to
            # comparing against the email body).
            "expected_summary": summary,
        })

    if not rows:
        raise ValueError(f"{dataset_path} contains no records")

    return rows, version


# --------------------------------------------------------------------------
# Phase 3: multi-dimensional scoring helpers
# --------------------------------------------------------------------------
# One test case is scored on FOUR axes; these helpers produce the three the
# classifier itself cannot: latency percentiles, token totals, and the
# summary-relevance judge. Exact match stays where it was - in assemble_run,
# one line: predicted == expected.


# Providers a real LLM judge can run on. Everything else (mock, test doubles)
# goes straight to the offline heuristic: asking a keyword mock to grade
# summaries would return confident garbage, and parsing it would be theatre.
LLM_PROVIDER_NAMES = frozenset({"ollama", "openai", "anthropic", "gemini"})


def percentile(values: list[float], p: float) -> float:
    """
    Nearest-rank percentile: the value that `p` percent of samples fall under.

    Nearest-rank on purpose - no interpolation, so with a small sample every
    reported number is a latency that a real request actually had. Interpolated
    percentiles can invent a number between two requests that never happened.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(p / 100.0 * len(ordered))
    rank = min(max(rank, 1), len(ordered))
    return float(ordered[rank - 1])


_STOPWORDS = frozenset(
    "the a an and or of for to in on with is are was were be been i my me you "
    "your that this its as at by from not no have has had but so if then than "
    "there here what which who when why how can could would should will shall "
    "may might do does did done".split()
)


def _content_words(text: str | None) -> set[str]:
    """Lowercased meaningful words (len > 2, stopwords dropped) - the unit the
    heuristic judge measures overlap in."""
    return {
        w
        for w in re.findall(r"[a-z0-9']+", (text or "").lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def heuristic_judge_score(
    summary: str, ideal: str | None = None, body: str = ""
) -> int:
    """
    Offline 1-5 summary score: how much of the summary is grounded in a
    reference text (the human's ideal summary when we have one, else the
    email body itself).

    This is the FALLBACK, not the goal: word overlap cannot tell a vivid
    paraphrase from a bland restatement. It can tell grounded from invented -
    which is the 1-vs-rest boundary the rubric cares about most.
    """
    words = _content_words(summary)
    if not words:
        return 1  # nothing said -> nothing right
    ref = _content_words(ideal) or _content_words(body)
    if not ref:
        return 3  # no reference at all -> cannot judge, middle of the scale
    grounded = len(words & ref) / len(words)
    if grounded >= 0.75:
        return 5
    if grounded >= 0.55:
        return 4
    if grounded >= 0.35:
        return 3
    if grounded >= 0.15:
        return 2
    return 1  # invented / irrelevant: almost nothing ties to the reference


JUDGE_SYSTEM_PROMPT = """You grade one-sentence summaries of customer-support emails.
Score 1-5 against these criteria:
5 = perfect: one concise sentence, fully grounded in the email, nothing invented
4 = good: accurate and concise, only a minor wording issue
3 = acceptable: accurate but too long, too short, or vague
2 = weak: partly accurate, misses the main point or adds unsupported detail
1 = irrelevant or hallucinated: does not describe this email, or invents facts
Reply with ONLY this JSON and nothing else: {"score": <integer 1-5>}"""


def build_judge_prompts(
    summary: str, ideal: str | None, body: str
) -> tuple[str, str]:
    """(system, user) turns for the judge call. Kept here so the sync judge,
    the async judge, and any future transport grade IDENTICALLY."""
    lines = [
        "Grade the summary below.",
        "",
        "EMAIL:",
        body or "(none)",
        "",
        "SUMMARY TO GRADE:",
        summary or "(empty)",
    ]
    if ideal:
        lines += ["", "Reference summary written by a human:", ideal]
    return JUDGE_SYSTEM_PROMPT, "\n".join(lines)


def parse_judge_score(raw: str) -> int | None:
    """Model text -> score in 1..5, or None when the answer is unusable.
    None never raises: the caller falls back to the heuristic instead of
    killing a whole run over one bad judge reply."""
    try:
        text = (raw or "").strip()
        if "```" in text:
            for part in text.split("```"):
                candidate = part.removeprefix("json").strip()
                if candidate.startswith("{"):
                    text = candidate
                    break
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        data = json.loads(text[start : end + 1])
        score = int(data.get("score"))
        return score if 1 <= score <= 5 else None
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


class JudgeOutcome(BaseModel):
    """Result of judging ONE summary."""

    score: int | None = Field(default=None, ge=1, le=5)
    source: str = Field(
        default="none", description="llm (live model) | heuristic (offline fallback)."
    )
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def judge_summary(
    summary: str,
    ideal: str | None,
    body: str,
    client: ChatClient | None = None,
) -> JudgeOutcome:
    """
    Rate summary quality 1-5. LLM-as-Judge when a real provider is available,
    heuristic fallback otherwise - offline, mocked, or the judge answered
    something unusable. The judge NEVER fails a run: a grading problem must
    not look like a classifier regression.
    """
    name = getattr(client, "name", "") or ""
    if name in LLM_PROVIDER_NAMES:
        started = time.perf_counter()
        try:
            system, user = build_judge_prompts(summary, ideal, body)
            raw = client.complete(system, user)
            score = parse_judge_score(raw)
        except Exception:  # noqa: BLE001 - any judge failure degrades, never raises
            score = None
        latency = (time.perf_counter() - started) * 1000
        if score is not None:
            return JudgeOutcome(score=score, source="llm", latency_ms=latency)
        return JudgeOutcome(
            score=heuristic_judge_score(summary, ideal, body),
            source="heuristic",
            latency_ms=latency,
        )
    return JudgeOutcome(
        score=heuristic_judge_score(summary, ideal, body), source="heuristic"
    )


async def judge_summary_async(
    summary: str,
    ideal: str | None,
    body: str,
    client=None,
    enabled: bool = True,
) -> JudgeOutcome:
    """Same judge, async transport. The HTTP path reports real token usage for
    the judge call too; the mock/offline path never reaches a network."""
    if not enabled:
        return JudgeOutcome(score=None, source="none")
    name = getattr(client, "name", "") or ""
    if name in LLM_PROVIDER_NAMES:
        started = time.perf_counter()
        prompt_tokens = completion_tokens = 0
        score = None
        try:
            system, user = build_judge_prompts(summary, ideal, body)
            completion = await client.acomplete(system, user)
            score = parse_judge_score(completion.text)
            prompt_tokens = completion.prompt_tokens
            completion_tokens = completion.completion_tokens
        except Exception:  # noqa: BLE001 - graceful fallback, by design
            score = None
        latency = (time.perf_counter() - started) * 1000
        if score is not None:
            return JudgeOutcome(
                score=score, source="llm", latency_ms=latency,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            )
        return JudgeOutcome(
            score=heuristic_judge_score(summary, ideal, body),
            source="heuristic", latency_ms=latency,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )
    return JudgeOutcome(
        score=heuristic_judge_score(summary, ideal, body), source="heuristic"
    )


class CaseResult(BaseModel):
    """ONE executed test case, measured. Both the sync evaluate() loop and the
    async runner emit this shape, so assemble_run - and therefore every metric
    - is written exactly once for both paths."""

    sample_id: str
    expected: str
    category: str
    summary: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    judge: JudgeOutcome | None = None


def assemble_run(
    cases: list[CaseResult],
    provider: str,
    prompt_version: str,
    dataset_version: str,
    duration_ms: float = 0.0,
) -> RunResult:
    """
    Score executed cases into a RunResult. THE single place exact match,
    latency percentiles, token totals and judge stats are computed - the sync
    loop and the async runner both end here, so the two paths cannot drift
    apart metric by metric.
    """
    predictions: list[Prediction] = []
    per_cat_correct: Counter[str] = Counter()
    per_cat_total: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = {}
    latencies: list[float] = []
    judge_scores: list[int] = []
    n_llm = n_heuristic = 0
    prompt_tokens = completion_tokens = 0
    judge_prompt = judge_completion = 0

    for case in cases:
        correct = case.category == case.expected
        judge = case.judge
        if judge is not None and judge.score is not None:
            judge_scores.append(judge.score)
            if judge.source == "llm":
                n_llm += 1
            elif judge.source == "heuristic":
                n_heuristic += 1
            judge_prompt += judge.prompt_tokens
            judge_completion += judge.completion_tokens
        prompt_tokens += case.prompt_tokens
        completion_tokens += case.completion_tokens
        latencies.append(case.latency_ms)

        predictions.append(
            Prediction(
                sample_id=case.sample_id,
                expected=case.expected,
                category=case.category,
                correct=correct,
                summary=case.summary,
                case_metrics=CaseMetrics(
                    latency_ms=round(case.latency_ms, 2),
                    prompt_tokens=case.prompt_tokens,
                    completion_tokens=case.completion_tokens,
                    total_tokens=case.prompt_tokens + case.completion_tokens,
                    judge_score=judge.score if judge else None,
                    judge_source=judge.source if judge else "none",
                    judge_latency_ms=round(judge.latency_ms, 2) if judge else 0.0,
                    judge_prompt_tokens=judge.prompt_tokens if judge else 0,
                    judge_completion_tokens=judge.completion_tokens if judge else 0,
                ),
            )
        )
        per_cat_total[case.expected] += 1
        if correct:
            per_cat_correct[case.expected] += 1
        confusion.setdefault(case.expected, Counter())[case.category] += 1

    n_samples = len(predictions)
    n_correct = sum(1 for p in predictions if p.correct)

    return RunResult(
        # Short random id: readable in logs, unique per run. A timestamp alone
        # collides when two runs happen in the same second.
        run_id=f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        provider=provider,
        prompt_version=prompt_version,
        dataset_version=dataset_version,
        n_samples=n_samples,
        accuracy=(n_correct / n_samples) if n_samples else 0.0,
        metrics=Metrics(
            accuracy=(n_correct / n_samples) if n_samples else 0.0,
            n_samples=n_samples,
            n_correct=n_correct,
            per_category_correct=dict(per_cat_correct),
            per_category_total=dict(per_cat_total),
            confusion={k: dict(v) for k, v in confusion.items()},
            latency=LatencyStats(
                p50_ms=round(percentile(latencies, 50), 2),
                p90_ms=round(percentile(latencies, 90), 2),
                mean_ms=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                max_ms=round(max(latencies), 2) if latencies else 0.0,
                n=len(latencies),
            ),
            tokens=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                judge_prompt_tokens=judge_prompt,
                judge_completion_tokens=judge_completion,
                judge_total_tokens=judge_prompt + judge_completion,
            ),
            judge=JudgeStats(
                mean_score=(
                    round(sum(judge_scores) / len(judge_scores), 3)
                    if judge_scores
                    else None
                ),
                n_scored=len(judge_scores),
                n_llm=n_llm,
                n_heuristic=n_heuristic,
            ),
            run_duration_ms=round(duration_ms, 2),
        ),
        predictions=predictions,
    )


def evaluate(
    prompt: PromptConfig | PromptSpec | None = None,
    client: ChatClient | None = None,
    samples: list[dict] | None = None,
    samples_path: str | Path | None = None,
    prompt_version: str | None = None,
    classify: Callable[..., dict] | None = None,
    dataset_version: str | None = None,
    judge: bool = False,
) -> RunResult:
    """
    Classify every sample, compare to its human label, compute metrics.

    Every classification goes through the SAME public function
    `classify_email` that production would call. Evaluating a different code
    path than the one you ship would make the whole system worthless.

    `classify` is the SWAP SEAM: any callable with the signature
    `(EmailInput, prompt=..., client=...) -> dict` can be evaluated without
    changing a line of this module. That is what the contract buys.

    Typed on both sides: a dataset row becomes an `EmailInput`, and the answer
    is re-validated as an `EmailClassification` before it is scored. The
    evaluator trusts the contract, so no label or shape checks are repeated here.

    Args:
        prompt:         pass a PromptConfig / PromptSpec to override the YAML files.
        prompt_version: which prompt FILE to load, e.g. "v2". Used when
                        `prompt` is omitted. The resulting run's prompt_version
                        is read from that file.
        dataset_version: version of the labelled set being scored. Read from
                        the dataset JSON when `samples_path` points at one;
                        passed explicitly when `samples` is a plain list.
                        Lands in the baseline key, so two dataset versions can
                        never be compared to each other.
        classify:       alternative implementation. Defaults to classify_email.
        judge:          also score summary relevance (1-5) per case. Defaults
                        OFF here: the sync seam reports no token usage and the
                        Phase 2 variance tool (baseline.py) would silently make
                        a second full set of live calls. The Phase 3 async
                        runner is the live path, where the judge defaults ON.

    Latency and (from the HTTP transport) tokens are recorded per case either
    way - this loop times each `classify` call itself.
    """
    if prompt is None:
        from prompt_registry import load_prompt_config

        prompt = load_prompt_config(prompt_version)
    client = client or get_client()
    classify = classify or classify_email
    emails = samples if samples is not None else load_samples(samples_path)

    # Which bar are we measuring against? Read it from the dataset file when
    # there is one, so `evaluate(samples_path=golden)` needs no extra argument
    # and cannot accidentally report a run under the wrong dataset's baseline.
    if dataset_version is None:
        dataset_version = read_dataset_version(samples_path or SAMPLES_PATH)

    started = time.perf_counter()
    cases: list[CaseResult] = []

    for email in emails:
        # Dataset row -> typed input. Subject and id ride along, so a prediction
        # can always be traced back to its row without a parallel dict.
        typed_input = EmailInput(
            body=email["body"],
            subject=email.get("subject", ""),
            email_id=email["id"],
        )
        t0 = time.perf_counter()
        raw = classify(typed_input, prompt=prompt, client=client)
        latency_ms = (time.perf_counter() - t0) * 1000
        # The dict the storage layer consumes, re-validated as the typed output.
        # If a future classifier returned a wrong shape, this raises HERE, at one
        # boundary, instead of silently skewing every metric downstream.
        result = EmailClassification(**raw)

        outcome = None
        if judge:
            outcome = judge_summary(
                result.summary,
                email.get("expected_summary"),
                email["body"],
                client=client,
            )

        cases.append(
            CaseResult(
                sample_id=email["id"],
                expected=email["expected"],
                category=result.category,
                summary=result.summary,
                latency_ms=latency_ms,
                # Sync seam: providers.complete() returns text only, so token
                # usage is honestly 0 here. The async HTTP runner reads the
                # usage block the API returns and fills these in.
                judge=outcome,
            )
        )

    duration_ms = (time.perf_counter() - started) * 1000
    return assemble_run(
        cases,
        provider=getattr(client, "name", "unknown"),
        prompt_version=prompt.version,
        dataset_version=dataset_version,
        duration_ms=duration_ms,
    )


# --------------------------------------------------------------------------
# Step 2: compare against the stored baseline
# --------------------------------------------------------------------------


def classify_run_status(
    delta: float,
    warn_threshold: float = WARN_THRESHOLD,
    critical_threshold: float = CRITICAL_THRESHOLD,
) -> str:
    """
    Pass-rate delta -> PASS | WARN | CRITICAL_REGRESSION. Pure function: no
    store, no run, no side effects - so the threshold rule can be tested at
    its exact boundaries instead of only through a full evaluation.

        delta = current - baseline      (negative = worse)
        drop  = -delta

        drop <= warn_threshold     -> PASS               noise, ship it
        drop <= critical_threshold -> WARN               look before shipping
        drop >  critical_threshold -> CRITICAL_REGRESSION do not ship
    """
    drop = -delta
    if drop <= warn_threshold:
        return "PASS"
    if drop <= critical_threshold:
        return "WARN"
    return "CRITICAL_REGRESSION"


def _flip_lists(
    run: RunResult, baseline_predictions: list[dict]
) -> tuple[list[str], list[str]]:
    """Split per-case outcomes into (regressions, improvements).

    A pass-rate delta says HOW MUCH changed; these lists say WHICH cases -
    the difference between "down 5 points" and "these four emails broke".
    Sample ids absent from the baseline (dataset rows added since) are skipped:
    they had no baseline state to regress from.
    """
    base = {p["sample_id"]: bool(p["correct"]) for p in baseline_predictions}
    regressions: list[str] = []
    improvements: list[str] = []
    for p in run.predictions:
        was = base.get(p.sample_id)
        if was is None:
            continue
        if was and not p.correct:
            regressions.append(p.sample_id)
        elif not was and p.correct:
            improvements.append(p.sample_id)
    return sorted(regressions), sorted(improvements)


def _per_category_delta(run: RunResult, baseline_metrics: dict) -> dict[str, float]:
    """Per-category accuracy delta: current - baseline, rounded to 4 places.

    Computed from the stored aggregate counts (not baseline predictions), so it
    works even for runs stored before per-case history existed. Categories the
    baseline never saw are skipped - no baseline, no delta.
    """
    deltas: dict[str, float] = {}
    base_total = baseline_metrics.get("per_category_total", {}) or {}
    base_correct = baseline_metrics.get("per_category_correct", {}) or {}
    for category, total in run.metrics.per_category_total.items():
        b_total = base_total.get(category, 0)
        if not b_total:
            continue
        current_acc = run.metrics.per_category_correct.get(category, 0) / total
        baseline_acc = base_correct.get(category, 0) / b_total
        deltas[category] = round(current_acc - baseline_acc, 4)
    return deltas


def compare_to_baseline(
    run: RunResult,
    store: Store | None = None,
    baseline_key: str | None = None,
    threshold: float | None = None,
    warn_threshold: float = WARN_THRESHOLD,
    critical_threshold: float = CRITICAL_THRESHOLD,
) -> Verdict:
    """
    Decide: PASS, WARN, CRITICAL_REGRESSION, or BASELINE_CREATED - and say
    exactly which cases moved.

    Args:
        baseline_key: which baseline to compare against. Omit to use
            "<provider>:<prompt_version>:<dataset_version>" from the run
            itself — the default, because a baseline belongs to exactly one
            prompt version AND one dataset version (see BASELINE_KEY_FORMAT).
        threshold: LEGACY single cut-off. When given (not None), it replaces
            the warn/critical pair: any drop beyond it is CRITICAL_REGRESSION
            and nothing is ever WARN - the old binary behaviour, on demand.
        warn_threshold / critical_threshold: the Phase 3 two-level line,
            defaulting to the module constants. "Configurable" means both the
            constants and these arguments.

    Rule:
        no baseline yet                  -> BASELINE_CREATED
        drop <= warn_threshold           -> PASS
        drop <= critical_threshold       -> WARN
        drop >  critical_threshold       -> CRITICAL_REGRESSION
    """
    store = store or get_store()
    key = baseline_key or default_baseline_key(run)

    if threshold is not None:
        # Legacy mode: one strict line instead of the two-level verdict.
        warn_threshold = critical_threshold = threshold

    baseline = store.get_baseline(key)

    if baseline is None:
        store.set_baseline(key, _as_row(run))
        return Verdict(
            status="BASELINE_CREATED",
            run_id=run.run_id,
            provider=run.provider,
            prompt_version=run.prompt_version,
            dataset_version=run.dataset_version,
            current_accuracy=run.accuracy,
            baseline_accuracy=None,
            accuracy_delta=None,
            threshold=critical_threshold,
            warn_threshold=warn_threshold,
            details=[
                f"No baseline under key {key!r}. "
                f"Stored this run as the baseline ({run.accuracy:.4f}).",
                "Nothing to compare yet - first observation, not a verdict.",
            ],
            regression_detected=False,
        )

    baseline_accuracy = float(baseline["accuracy"])
    delta = round(run.accuracy - baseline_accuracy, 6)
    status = classify_run_status(delta, warn_threshold, critical_threshold)

    # Per-case diff: needs the baseline run's individual answers, which are
    # stored under the baseline row's run_id.
    baseline_predictions = store.get_predictions(baseline.get("run_id", ""))
    regressions, improvements = _flip_lists(run, baseline_predictions)
    per_cat_delta = _per_category_delta(run, baseline.get("metrics") or {})

    details = [
        f"Baseline {baseline['run_id']} pass rate {baseline_accuracy:.4f}.",
        f"Current  {run.run_id} pass rate {run.accuracy:.4f}.",
        f"Pass-rate delta {delta:+.4f} "
        f"(warn below -{warn_threshold:.4f}, critical below -{critical_threshold:.4f}).",
    ]
    if regressions:
        shown = ", ".join(regressions[:8])
        more = f" (+{len(regressions) - 8} more)" if len(regressions) > 8 else ""
        details.append(f"Regressions ({len(regressions)}): {shown}{more}")
    if improvements:
        shown = ", ".join(improvements[:8])
        more = f" (+{len(improvements) - 8} more)" if len(improvements) > 8 else ""
        details.append(f"Improvements ({len(improvements)}): {shown}{more}")
    for category, cat_delta in sorted(per_cat_delta.items()):
        if cat_delta < 0:
            details.append(f"Category {category!r}: {cat_delta:+.4f} vs baseline.")
    if status in ("WARN", "CRITICAL_REGRESSION"):
        details.extend(_explain_regressions(run, key, store))
        details.append(
            "Investigate: was the prompt version, model, or provider changed?"
        )

    return Verdict(
        status=status,
        run_id=run.run_id,
        provider=run.provider,
        prompt_version=run.prompt_version,
        dataset_version=run.dataset_version,
        current_accuracy=run.accuracy,
        baseline_accuracy=baseline_accuracy,
        accuracy_delta=delta,
        threshold=critical_threshold,
        warn_threshold=warn_threshold,
        details=details,
        regression_detected=status in ("WARN", "CRITICAL_REGRESSION"),
        pass_rate_delta=delta,
        per_category_delta=per_cat_delta,
        regressions=regressions,
        improvements=improvements,
    )


def _explain_regressions(
    run: RunResult, baseline_key: str, store: Store
) -> list[str]:
    """
    Name the categories that lost ground vs the baseline.

    An accuracy number alone says "something broke". Knowing WHICH class broke
    turns a two-hour hunt into a two-minute one.
    """
    notes: list[str] = []
    baseline_accuracy = store.get_baseline(baseline_key)
    if not baseline_accuracy:
        return notes

    for category, total in run.metrics.per_category_total.items():
        got = run.metrics.per_category_correct.get(category, 0)
        baseline_total = baseline_accuracy["metrics"].get("per_category_total", {}).get(
            category, total
        )
        baseline_got = baseline_accuracy["metrics"].get("per_category_correct", {}).get(
            category, got
        )
        if got < baseline_got:
            notes.append(
                f"Category {category!r}: {got}/{total} correct now, "
                f"was {baseline_got}/{baseline_total} at baseline."
            )
    return notes or ["No single category drove the drop; check per-sample diffs."]


# --------------------------------------------------------------------------
# Convenience: evaluate + store + compare, one call
# --------------------------------------------------------------------------


def run_evaluation(
    prompt: PromptConfig | None = None,
    client: ChatClient | None = None,
    store: Store | None = None,
    baseline_key: str | None = None,
    threshold: float | None = None,
    persist: bool = True,
    prompt_version: str | None = None,
    samples: list[dict] | None = None,
    classify: Callable[..., dict] | None = None,
    dataset_version: str | None = None,
    judge: bool = False,
) -> tuple[RunResult, Verdict]:
    """
    The whole MRDS loop in one call: score, persist, compare.

    Order matters: persist the run BEFORE comparing, so a crash during compare
    still leaves the raw predictions on disk to investigate.

    With `prompt_version="v2"` and no explicit `baseline_key`, the run is scored
    against the v2 baseline for THIS dataset (`mock:v2:<dataset_version>`),
    never v1's and never another dataset's — a different prompt is a different
    contract, and a different dataset is a different bar, neither is a
    regression.

    `threshold=None` (default) uses the Phase 3 warn/critical pair. Passing a
    number keeps the legacy single-line behaviour.
    """
    store = store or get_store()
    result = evaluate(
        prompt=prompt,
        client=client,
        prompt_version=prompt_version,
        samples=samples,
        classify=classify,
        dataset_version=dataset_version,
        judge=judge,
    )

    if persist:
        store.save_run(
            _as_row(result),
            [p.model_dump() for p in result.predictions],
        )

    verdict = compare_to_baseline(
        result,
        store=store,
        baseline_key=baseline_key,  # None -> derived from provider + prompt file
        threshold=threshold,
    )
    return result, verdict


def _as_row(run: RunResult) -> dict:
    """Flatten a RunResult into the dict shape storage.save_run expects."""
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