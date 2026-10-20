"""
contract.py — the typed boundary of MRDS.

Everything that crosses into or out of the "feature under test" is typed here,
and nowhere else:

    PromptSpec         the YAML prompt FILE schema (prompts/v1.yaml, v2.yaml)
    PromptConfig       THE CONTRACT: a prompt spec plus the rules the pipeline
                       consumes (allowed labels, prompt construction, validation)
    EmailInput         typed input: one support email
    EmailClassification typed output: category + summary + provenance

Naming note: this file was previously `PromptConfig` inside classifier.py and
meant "the YAML file schema". It is now `PromptSpec` — a file's contents. The
name `PromptConfig` belongs to the CONTRACT, which is what the eval pipeline
actually consumes. See decision.md D22.

Why a contract at all: without one, `classify_email` returned a bare dict and a
malformed model answer had to be caught by scattered `if` checks in every caller.
With one, an invalid shape cannot be constructed in the first place, and the
checks live in exactly one file.

The per-version label set is the interesting part. `Category` is a closed
Literal (billing | technical | account | general) so a typo or an invented label
cannot be constructed at all. `EmailClassification.allowed_categories` then
narrows that per prompt version: v1 allows all four, v2 deliberately allows only
three. The type says "a valid MRDS label"; the contract says "and this version
may use it".
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# --------------------------------------------------------------------------
# The closed label set. Type + runtime list, one source of truth.
# --------------------------------------------------------------------------

Category = Literal["billing", "technical", "account", "general"]

CATEGORIES: tuple[Category, ...] = ("billing", "technical", "account", "general")

# Marker separating our instructions from the customer's text in one user message.
# Shared with the mock provider so it can find the email body.
EMAIL_MARKER = "--- EMAIL ---"


class ContractViolationError(ValueError, RuntimeError):
    """
    The model, or a caller, produced something the contract forbids.

    Inherits from both ValueError and RuntimeError so older call sites that
    caught either still work. It is raised at the boundary, once, instead of
    being checked inline in five different places.
    """


# --------------------------------------------------------------------------
# Input side
# --------------------------------------------------------------------------


class EmailInput(BaseModel):
    """
    One support email, typed.

    A bare `str` would be simpler, but metadata then travels as a parallel dict
    (`email_id`, `subject`) that can drift out of sync with the text. Carrying
    subject and id inside the model means the pipeline can join predictions back
    to the dataset row without anyone passing two objects around together.
    """

    email_id: str = Field(default="", description="Dataset row id, for traceability.")
    subject: str = Field(default="", description="Email subject line, optional.")
    body: str = Field(description="Raw email text from the customer.")

    @field_validator("body", mode="after")
    @classmethod
    def _body_not_blank(cls, body: str) -> str:
        if not body.strip():
            raise ValueError("body is empty - nothing to classify.")
        return body


# --------------------------------------------------------------------------
# Prompt file schema (was called PromptConfig before this step)
# --------------------------------------------------------------------------


class PromptSpec(BaseModel):
    """
    The contents of ONE prompts/<version>.yaml file.

    Renamed from `PromptConfig` (decision.md D22) so the name `PromptConfig`
    can mean the contract. This class is only about describing a FILE; the
    pipeline consumes `PromptConfig`, which wraps one of these.
    """

    version: str = Field(
        description="Immutable version ID. Stored with every run so a metric "
        "change can be attributed to a specific prompt file."
    )
    # No trailing comma here on purpose: `Field(...)` followed by a comma would
    # make the default a TUPLE, and Pydantic would happily accept it instead of
    # treating the field as required. (Bug found in an earlier step.)
    created_at: str = Field(description="Explicit UTC timestamp written into the file.")
    system_prompt: str = Field(description="Rules and output contract, sent in the system turn.")
    categories: tuple[Category, ...] = Field(
        default=CATEGORIES, description="Closed set of allowed labels for this version."
    )
    few_shots: list[dict] = Field(
        default_factory=list,
        description="Worked examples. Sent in the SYSTEM turn, never mixed into the user turn.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=300, gt=0)

    @field_validator("categories", mode="after")
    @classmethod
    def _labels_must_be_unique(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(labels)) != len(labels):
            raise ValueError("categories contains duplicate labels")
        return labels

    @field_validator("categories", mode="after")
    @classmethod
    def _labels_must_be_known(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [label for label in labels if label not in CATEGORIES]
        if unknown:
            raise ValueError(
                f"categories {unknown} are not in MRDS's known label set {list(CATEGORIES)}"
            )
        return labels


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


class PromptConfig(BaseModel):
    """
    THE CONTRACT the eval pipeline consumes.

    A prompt spec (what the YAML file says) plus everything the pipeline must
    enforce: which labels this version may emit, how the prompt is assembled, and
    that the model's answer is valid before anyone sees it.

    One class, not several, because a spec without its enforcement rules is
    exactly the thing that caused the old scattered `if` checks.
    """

    spec: PromptSpec

    # -- labels ---------------------------------------------------------
    @property
    def version(self) -> str:
        """The prompt file's version ID. Never re-derived anywhere else."""
        return self.spec.version

    @property
    def allowed_categories(self) -> tuple[Category, ...]:
        """Labels THIS version may emit. v2 deliberately has fewer than v1."""
        return self.spec.categories

    @property
    def fallback_category(self) -> Category:
        """
        Where an out-of-version label goes.

        v1 lists `general` last, so unmatched mail becomes `general`. v2 has no
        catch-all, so the same mail falls into `account` - which is precisely the
        regression v2 exists to demonstrate.
        """
        return self.allowed_categories[-1]

    @classmethod
    def from_spec(cls, spec: PromptSpec) -> "PromptConfig":
        """Wrap a YAML file's contents into a contract."""
        return cls(spec=spec)

    # -- prompt assembly -------------------------------------------------
    def build_system_prompt(self) -> str:
        """
        Compose the SYSTEM turn: file's rules + allowed labels + few-shot examples.

        Few-shots live here, not in the user turn, for three reasons:
          1. Every provider API has a dedicated rules/examples channel (Anthropic
             `system`, Gemini `systemInstruction`, OpenAI system role), so nothing
             needs provider-specific handling.
          2. The user turn must contain only instructions + the customer's email,
             or the mock (and any prompt cache keyed on the user turn) would see
             a different input shape per version and break comparability.
          3. Examples are scaffolding, not user data. Mixing them into the user
             turn risks them being echoed into the summary.
        """
        parts = [
            self.spec.system_prompt.strip(),
            # The mock parses this line to learn which labels this version allows.
            f"ALLOWED_LABELS: {', '.join(self.allowed_categories)}",
        ]

        if self.spec.few_shots:
            lines = ["", "Examples:"]
            for shot in self.spec.few_shots:
                lines.append(f"- input: {shot.get('input', '')}")
                lines.append(f"  output: {shot.get('output', '')}")
            parts.append("\n".join(lines))

        return "\n".join(parts)

    def build_user_prompt(self, email: EmailInput) -> str:
        """
        Compose the single user message: instructions + the email itself.

        Shape is identical for every prompt version on purpose.
        """
        labels = ", ".join(self.allowed_categories)
        instructions = (
            "Task: classify the support email below into exactly one category "
            f"from: {labels}.\n"
            "Reply with ONLY this JSON shape, nothing else:\n"
            '{"category": "<one label>", "summary": "<one sentence>"}\n'
            "The summary must be a single sentence describing the customer's issue.\n"
        )
        subject = f"\n--- SUBJECT ---\n{email.subject}\n" if email.subject else ""
        return f"{instructions}\n{EMAIL_MARKER}\n{subject}{email.body}"

    # -- validation ------------------------------------------------------
    def parse_classification(
        self,
        raw: str,
        provider: str = "unknown",
    ) -> "EmailClassification":
        """
        Turn raw model text into a validated EmailClassification.

        ALL output validation for MRDS lives in this one method. Parsing
        (fences, chatter, wrong JSON) and label checking are the same problem:
        "did the model honour the contract?". Keeping them here means a new
        caller cannot accidentally skip the checks.

        Raises:
            ContractViolationError - unparseable text, wrong JSON shape, a label
                outside MRDS's closed set, or a label this version may not use.
        """
        payload = _extract_json_object(raw)

        summary = payload.get("summary", "")
        if not isinstance(summary, str) or not summary.strip():
            raise ContractViolationError(
                f"'summary' must be a non-empty string. Model output was: {raw!r}"
            )

        # A label outside MRDS's closed set (refund_request, "billing issue"...)
        # is STRUCTURAL: it means the model ignored the contract, so raise.
        # `category: Category` below makes that unconstructable.
        category = payload.get("category")
        if category not in CATEGORIES:
            raise ContractViolationError(
                f"category {category!r} is outside MRDS's closed label set "
                f"{list(CATEGORIES)}. The model ignored the contract."
            )

        # A valid MRDS label that THIS version may not use is RECOVERABLE:
        # fold it to the version's own fallback. v2 has no `general` catch-all,
        # so general feedback lands in `account` - exactly the quality loss v2
        # exists to demonstrate. One bad row must never kill a batch.
        allowed = self.allowed_categories
        if allowed and category not in allowed:
            category = self.fallback_category

        try:
            return EmailClassification(
                category=category,
                summary=summary.strip(),
                provider=provider,
                prompt_version=self.version,  # from the FILE, not re-derived
                allowed_categories=allowed,
            )
        except ValidationError as exc:
            # Translate Pydantic's error into the contract's own exception, so a
            # caller only ever has to catch ContractViolationError.
            raise ContractViolationError(
                f"Model output violates the {self.version!r} contract: {_short(exc)}"
            ) from exc


# --------------------------------------------------------------------------
# Output side
# --------------------------------------------------------------------------


class EmailClassification(BaseModel):
    """
    Typed output of the feature under test.

    `category` is a Literal, so `EmailClassification(category="refund_request")`
    raises at construction instead of flowing into the metrics. Construction is
    the validation: there is no way to hold an invalid one of these.

    Serialization stays a plain dict at the API edge (`.model_dump()`), which is
    what SQLite and JSON consume. `allowed_categories` is excluded from that
    dict: it is validation context, not an answer.
    """

    category: Category
    summary: str
    provider: str = "unknown"
    prompt_version: str = "unknown"
    allowed_categories: tuple[Category, ...] = Field(
        default=(),
        exclude=True,
        description="Per-version label set used to validate `category`. "
        "Excluded from serialization.",
    )

    @model_validator(mode="after")
    def _category_must_be_allowed(self) -> "EmailClassification":
        """
        Last line of defence for the per-version label set.

        PromptConfig.parse_classification folds out-of-version labels (that is
        recoverable). This validator RAINS, so direct construction - a test, a
        future caller, a hand-built payload - cannot quietly create a result
        this prompt version forbids.

        So: `Category` says "a valid MRDS label"; this says "and this version
        may use it".
        """
        if self.allowed_categories and self.category not in self.allowed_categories:
            raise ContractViolationError(
                f"category {self.category!r} is not allowed by prompt version "
                f"{self.prompt_version!r}, which allows "
                f"{list(self.allowed_categories)}"
            )
        return self


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _short(exc: ValidationError) -> str:
    """Pydantic errors -> 2 lines max. Beginner-readable failure output."""
    lines = [
        f"{'.'.join(str(p) for p in err['loc']) or '<output>'}: {err['msg']}"
        for err in exc.errors()[:2]
    ]
    if len(exc.errors()) > 2:
        lines.append(f"...and {len(exc.errors()) - 2} more")
    return " | ".join(lines)


def _extract_json_object(raw: str) -> dict:
    """
    Pull the JSON object out of a model's raw text.

    Live models wrap JSON in ```json fences or add a sentence of chatter.
    Strip fences, then take the outermost {...} block. Failing both, raise:
    silently substituting a value would corrupt the exact numbers MRDS exists to
    measure (decision.md D13 - degrade on recoverable, raise on structural).
    """
    text = (raw or "").strip()

    if "```" in text:
        for part in text.split("```"):
            candidate = part.removeprefix("json").strip()
            if candidate.startswith("{"):
                text = candidate
                break

    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ContractViolationError(
                f"Model did not return JSON. Raw output was: {raw!r}"
            )
        text = text[start : end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractViolationError(f"Model returned invalid JSON: {raw!r}") from exc

    if not isinstance(data, dict):
        raise ContractViolationError(
            f"Model returned JSON that is not an object: {raw!r}"
        )
    return data