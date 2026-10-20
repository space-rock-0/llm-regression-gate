"""
providers.py — every LLM call in MRDS goes through ONE small interface.

Why: later phases must be able to swap the model (regression watch needs to
compare model A vs model B). If the classifier called `openai` directly, you
would have to edit the classifier. Instead it calls `get_client()`, and only
this file knows which vendor is really being used.

All clients expose the same single method:

    complete(system_prompt: str, user_prompt: str) -> str

...and return RAW TEXT. Parsing the text into JSON is the classifier's job,
not the provider's job. That keeps every provider dumb and swappable.

Providers:
    MockClient      offline, deterministic, zero cost  <- DEFAULT
    OllamaClient    local models (llama3, mistral...) via OpenAI-compatible API
    OpenAIClient    api.openai.com
    AnthropicClient api.anthropic.com (Claude)
    GeminiClient    generativelanguage.googleapis.com (Gemini)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol

# --------------------------------------------------------------------------
# The interface every provider must satisfy
# --------------------------------------------------------------------------


class ChatClient(Protocol):
    """Anything that can turn (system, user) text into text."""

    name: str

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


# --------------------------------------------------------------------------
# Helper: read an env var, with a default
# --------------------------------------------------------------------------


def _env(key: str, default: str = "") -> str:
    """Read an environment variable as text, falling back to `default`."""
    return os.environ.get(key, default) or default


# --------------------------------------------------------------------------
# 1. Mock — default. Deterministic, offline, free.
# --------------------------------------------------------------------------
# Why mock first: a regression-detection system needs a ground truth that NEVER
# moves. A live LLM rephrases its answer between calls, which looks like a
# regression but is just noise. The mock answers with fixed rules, so if a
# metric changes later, the change came from YOUR code, not from the weather.


# Keyword -> category. First match wins in this exact order, so the result is
# stable forever. This is a tiny hand-written "model".
_MOCK_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "billing",
        (
            "invoice", "billing", "charged", "charge", "refund", "payment",
            "card", "subscription", "receipt", "overcharge", "price", "plan",
        ),
    ),
    (
        "technical",
        (
            "error", "bug", "crash", "500", "timeout", "server", "slow",
            "broken", "not working", "failed", "down", "api", "login page",
        ),
    ),
    (
        "account",
        (
            "password", "reset", "sign in", "signin", "login", "2fa", "mfa",
            "locked out", "suspend", "suspended", "delete my account",
            "change my email", "permission", "cancel my account",
        ),
    ),
]


class MockClient:
    """Fake LLM. Answers with fixed keyword rules. Never calls a network."""

    name = "mock"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # Match on the EMAIL ONLY. Our own instructions list the labels
        # ("billing", "technical"...) and would otherwise match first.
        email_text = _strip_prompt_instructions(user_prompt)
        body = email_text.lower()

        # Which labels does THIS prompt version allow? Read from the system
        # prompt's ALLOWED_LABELS line so the mock honours the prompt file
        # instead of a hardcoded list. Without this, v1 and v2 would produce
        # identical results and the whole A/B path would be theatre.
        allowed = _allowed_labels(system_prompt)

        chosen = allowed[-1]  # fallback bucket = the prompt's own last label
        for category, keywords in _MOCK_RULES:
            if category in allowed and any(word in body for word in keywords):
                chosen = category
                break

        summary = _first_sentence(email_text)
        return json.dumps({"category": chosen, "summary": summary})


def _allowed_labels(system_prompt: str) -> list[str]:
    """
    Read the `ALLOWED_LABELS: a, b, c` line the classifier writes into the
    system turn. Falls back to MRDS's full label set if the line is missing, so
    the mock still works when called directly in a test.
    """
    for line in system_prompt.splitlines():
        if line.startswith("ALLOWED_LABELS:"):
            labels = [p.strip() for p in line.split(":", 1)[1].split(",")]
            labels = [label for label in labels if label]
            if labels:
                return labels
    return ["billing", "technical", "account", "general"]


def _strip_prompt_instructions(user_prompt: str) -> str:
    """
    The classifier builds one user message that contains BOTH instructions and
    the email. The mock needs only the email text. Everything after the
    '--- EMAIL ---' marker is the email; anything before is our instructions.
    """
    marker = "--- EMAIL ---"
    if marker in user_prompt:
        text = user_prompt.split(marker, 1)[1]
        # The user turn may carry an optional subject block. It is prompt
        # scaffolding, not the customer's words, so it must not reach the summary.
        if "--- SUBJECT ---" in text:
            text = text.split("--- SUBJECT ---", 1)[1]
        return text
    return user_prompt


def _first_sentence(text: str, max_words: int = 20) -> str:
    """Return the first sentence of `text`, capped at `max_words` words."""
    cleaned = " ".join(text.split())  # collapse newlines + repeated spaces
    for stop in (". ", "? ", "! "):
        if stop in cleaned:
            cleaned = cleaned.split(stop, 1)[0] + stop.strip()
            break
    words = cleaned.split(" ")
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]) + "..."
    return cleaned.strip()


# --------------------------------------------------------------------------
# 2. Ollama — local models. Talks the OpenAI wire format, so it reuses the
#    OpenAI SDK pointed at http://localhost:11434/v1
# --------------------------------------------------------------------------


class OllamaClient:
    """Local Ollama models. No API key, no cost, no data leaves the machine."""

    name = "ollama"

    def __init__(self) -> None:
        from openai import OpenAI  # imported lazily: only needed if selected

        base_url = _env("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self._client = OpenAI(base_url=base_url, api_key="ollama")
        self._model = _env("OLLAMA_MODEL", "llama3.2")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""


# --------------------------------------------------------------------------
# 3. OpenAI
# --------------------------------------------------------------------------


class OpenAIClient:
    name = "openai"

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=_env("OPENAI_API_KEY"))
        self._model = _env("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""


# --------------------------------------------------------------------------
# 4. Anthropic (Claude)
# --------------------------------------------------------------------------


class AnthropicClient:
    name = "anthropic"

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=_env("ANTHROPIC_API_KEY"))
        self._model = _env("ANTHROPIC_MODEL", "claude-sonnet-5")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=_int_env("ANTHROPIC_MAX_TOKENS", 300),
            temperature=_float_env("ANTHROPIC_TEMPERATURE", 0.0),
            system=system_prompt,  # Anthropic takes system as its own field
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


# --------------------------------------------------------------------------
# 5. Gemini (Google)
# --------------------------------------------------------------------------
# Uses stdlib urllib on purpose: no extra dependency needed, and the REST call
# is 20 lines. Swapping in the google-genai SDK later is a one-file change.


class GeminiClient:
    name = "gemini"

    def __init__(self) -> None:
        self._api_key = _env("GEMINI_API_KEY", _env("GOOGLE_API_KEY"))
        self._model = _env("GEMINI_MODEL", "gemini-2.0-flash")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={self._api_key}"
        )
        body = json.dumps(
            {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": _float_env("GEMINI_TEMPERATURE", 0.0),
                    "maxOutputTokens": 300,
                },
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))

        parts = payload["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts)


# --------------------------------------------------------------------------
# Small typed env readers (env vars are ALWAYS strings)
# --------------------------------------------------------------------------


def _int_env(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _float_env(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


# --------------------------------------------------------------------------
# The factory — one place that decides which client the whole system uses
# --------------------------------------------------------------------------

_PROVIDERS = {
    "mock": MockClient,
    "ollama": OllamaClient,
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
}


def get_client(provider: str | None = None) -> ChatClient:
    """
    Build the LLM client.

    Resolution order for `provider`:
        1. explicit argument
        2. env var PROVIDER
        3. env var USE_MOCK ("false" -> not mock)
        4. mock  (the safe default)

    Why mock is the fallback: MRDS must run in CI with no keys and no network.
    A missing key should degrade to a deterministic run, not a crash.
    """
    name = (provider or _env("PROVIDER") or "").strip().lower()
    if not name:
        use_mock = _env("USE_MOCK", "true").strip().lower()
        name = "mock" if use_mock not in ("false", "0", "no") else "openai"

    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider {name!r}. Choose one of: {sorted(_PROVIDERS)}"
        )

    try:
        return _PROVIDERS[name]()
    except ImportError as exc:  # SDK not installed for that provider
        raise RuntimeError(
            f"Provider {name!r} needs its SDK. Install it, or use the mock: "
            f"set PROVIDER=mock (default). Original error: {exc}"
        ) from exc
    except Exception as exc:  # e.g. missing API key
        raise RuntimeError(
            f"Provider {name!r} failed to start: {exc}. Set the required API key "
            f"env var, or fall back to PROVIDER=mock."
        ) from exc


def available_providers() -> list[str]:
    """Names accepted by get_client(). Handy for the README and error messages."""
    return sorted(_PROVIDERS)