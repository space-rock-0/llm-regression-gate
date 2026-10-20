"""
prompt_registry.py — load versioned prompt files from disk.

The prompts live in `prompts/*.yaml`. They are CODE, not runtime config: each one
is an artifact CI evaluates against, reviewed in a diff, and pinned by version
ID. That is why they are separate files instead of strings inside classifier.py.

Four operations, nothing more:
    list_versions()       -> what is on disk
    load_prompt(id)       -> one validated PromptSpec
    load_prompt_config(id)-> that spec wrapped in the contract (PromptConfig)
    resolve_version()     -> explicit arg > env var > default

Failure policy (consistent with decision.md D13, degrade on recoverable, raise
on structural):
    unknown version ID        -> raise (a typo must not silently run v1)
    file missing required key -> raise (structural: the prompt IS the contract)
    extra unknown keys        -> ignored (recoverable: forward compatible)
    few_shots malformed       -> raise (they are sent to the model verbatim)
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from contract import CATEGORIES as VALID_CATEGORIES
from contract import PromptConfig, PromptSpec, _short

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Fallback when nothing is specified anywhere. v1 is the plain baseline, so a
# bare `py demo.py` always reproduces the known-good behaviour.
DEFAULT_VERSION = "v1"

# Env var naming the prompt to use when the caller passes no version.
PROMPT_VERSION_ENV = "PROMPT_VERSION"

class PromptNotFoundError(FileNotFoundError):
    """Requested version ID is not on disk."""


class PromptValidationError(ValueError):
    """A prompt file exists but does not satisfy the schema."""


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _prompt_files(limit_dir: str | Path | None = None) -> list[Path]:
    """All *.yaml prompt files, sorted by filename."""
    directory = Path(limit_dir) if limit_dir else PROMPTS_DIR
    if not directory.exists():
        raise PromptNotFoundError(f"Prompt directory does not exist: {directory}")
    return sorted(directory.glob("*.yaml"))


def list_versions(limit_dir: str | Path | None = None) -> list[str]:
    """Version IDs available on disk, in natural order."""
    return [_version_of(path) for path in _prompt_files(limit_dir)]


def resolve_version(requested: str | None = None) -> str:
    """
    Decide which prompt version to use.

    Order: explicit argument > PROMPT_VERSION env var > DEFAULT_VERSION.

    Why an env var at all, when the rule is "pin one exact version": it is for
    developer convenience (`PROMPT_VERSION=v2 py demo.py`). A regression check
    always passes the version explicitly, because a baseline belongs to one
    exact prompt version and must never drift with the environment.
    """
    return requested or os.environ.get(PROMPT_VERSION_ENV) or DEFAULT_VERSION


def load_prompt(version: str | None = None, limit_dir: str | Path | None = None) -> PromptSpec:
    """
    Load one prompt file into a validated PromptConfig.

    Args:
        version:   version ID such as "v1". Omit to use resolve_version().
        limit_dir: override the prompts directory (tests use this).

    Returns:
        PromptSpec whose `.version` is the ID from the FILE, not the argument.
        The file is the source of truth: if the filename and the `version:` field
        disagree, we raise rather than let the mismatch reach storage.
    """
    version_id = resolve_version(version)
    path = _find_file(version_id, limit_dir)

    if path is None:
        available = ", ".join(list_versions(limit_dir)) or "none"
        raise PromptNotFoundError(
            f"Unknown prompt version {version_id!r}. Available: {available}. "
            f"Add {PROMPTS_DIR / (version_id + '.yaml')} or pass a known version."
        )

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PromptValidationError(f"{path.name} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise PromptValidationError(
            f"{path.name} must contain a YAML mapping at the top level, "
            f"got {type(raw).__name__}."
        )

    try:
        spec = PromptSpec(**raw)
    except ValidationError as exc:
        raise PromptValidationError(
            f"{path.name} failed prompt schema validation:\n{_short(exc)}"
        ) from exc

    # The file must be self-consistent. A file named v1.yaml claiming version: v2
    # would store runs under the wrong ID and corrupt every comparison after it.
    if spec.version != _version_of(path):
        raise PromptValidationError(
            f"{path.name} declares version {spec.version!r} but its filename says "
            f"{_version_of(path)!r}. Keep the two identical."
        )
    return spec


def load_prompt_config(
    version: str | None = None, limit_dir: str | Path | None = None
) -> PromptConfig:
    """
    Load one prompt file and wrap it in the CONTRACT object.

    This is the default path for classify_email(): callers get the file plus the
    rules that must be enforced (allowed labels, prompt assembly, output
    validation) instead of a bare file description.
    """
    return PromptConfig.from_spec(load_prompt(version, limit_dir))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _find_file(version_id: str, limit_dir: str | Path | None) -> Path | None:
    for path in _prompt_files(limit_dir):
        if _version_of(path) == version_id:
            return path
    return None


def _version_of(path: Path) -> str:
    """Version ID for a filename: 'prompts/v2.yaml' -> 'v2'."""
    return path.stem


# Re-exported so callers can import the closed label set from one place.
__all__ = [
    "DEFAULT_VERSION",
    "PROMPT_VERSION_ENV",
    "PROMPTS_DIR",
    "PromptConfig",
    "PromptSpec",
    "load_prompt_config",
    "PromptNotFoundError",
    "PromptValidationError",
    "VALID_CATEGORIES",
    "list_versions",
    "load_prompt",
    "resolve_version",
]