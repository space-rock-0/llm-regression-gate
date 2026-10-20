"""
classifier.py — THE FEATURE UNDER TEST.

One function does the whole job:

    classify_email(email, prompt=None, client=None, prompt_version=None) -> dict

Why a single function matters for MRDS: a regression is a change in the OUTPUT
of this function. If the behaviour lived in 6 classes with hidden state, a diff
in behaviour would be impossible to attribute. One function, no hidden state,
pure input -> output is what makes the later "compare against baseline" step
trustworthy.

Where validation lives: in contract.py, not here. This module's only jobs are
  1. accept str or EmailInput (convenience),
  2. call the provider,
  3. hand the raw text to PromptConfig.parse_classification(),
  4. return a plain dict for storage.
Every check about what a valid answer looks like lives in the contract, so a
second caller cannot skip it.
"""

from __future__ import annotations

from contract import (
    ContractViolationError,
    EmailInput,
    PromptConfig,
    PromptSpec,
)
from providers import ChatClient, get_client


def classify_email(
    email_body: str | EmailInput,
    prompt: PromptConfig | PromptSpec | None = None,
    client: ChatClient | None = None,
    prompt_version: str | None = None,
) -> dict:
    """
    Classify one support email.

    Args:
        email_body:     raw email text, or an EmailInput carrying subject/id.
        prompt:         a PromptConfig (contract) or a bare PromptSpec. Omit to
                        load prompts/<prompt_version>.yaml.
        client:         LLM client. Omit to resolve one from env vars (mock default).
        prompt_version: which prompt file to load. Omit to use
                        prompt_registry.resolve_version() (arg > env > v1).

    Returns:
        dict ready for JSON / SQLite - the locked return-shape decision:
        {"category": str, "summary": str, "provider": str, "prompt_version": str}
        Built from the typed EmailClassification, so an invalid shape can never
        reach a caller.

    Raises:
        ContractViolationError - blank email, unparseable model output, a label
            outside the closed set, or a label this prompt version may not use.
        RuntimeError          - provider failed to start or is unreachable.
    """
    email = _as_email_input(email_body)
    config = _as_prompt_config(prompt, prompt_version)
    client = client or get_client()

    raw = client.complete(config.build_system_prompt(), config.build_user_prompt(email))

    # Single point of validation. Typed inside, serialized once at the edge.
    result = config.parse_classification(raw, provider=getattr(client, "name", "unknown"))
    return result.model_dump()


# --------------------------------------------------------------------------
# Internals (prefixed _ = not part of the public feature surface)
# --------------------------------------------------------------------------


def _as_email_input(value: str | EmailInput) -> EmailInput:
    """
    Accept a bare string for convenience, or a typed EmailInput.

    Both end up as the same model, so everything downstream sees one shape.
    The string path exists because `classify_email("text")` is the obvious thing
    to write; it is not a second input contract.
    """
    if isinstance(value, EmailInput):
        return value
    try:
        return EmailInput(body=value)
    except Exception as exc:  # blank email -> contract error
        raise ContractViolationError(f"email_body is empty - nothing to classify: {exc}") from exc


def _as_prompt_config(
    prompt: PromptConfig | PromptSpec | None, prompt_version: str | None
) -> PromptConfig:
    """
    Resolve whatever we were given into the contract object.

    Order: an explicit PromptConfig > a bare PromptSpec (wrapped) > the YAML file
    named by prompt_version. Imported lazily because prompt_registry imports this
    module's contract, and a top-level import would be a cycle.
    """
    if isinstance(prompt, PromptConfig):
        return prompt
    if isinstance(prompt, PromptSpec):
        return PromptConfig.from_spec(prompt)

    from prompt_registry import load_prompt_config

    return load_prompt_config(prompt_version)