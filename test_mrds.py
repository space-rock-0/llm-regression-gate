"""
test_mrds.py — all tests run on the MOCK provider (no key, no network, no cost).

    py -m pytest test_mrds.py -q
    # or, if pytest is missing:
    py test_mrds.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from contract import (  # noqa: E402
    CATEGORIES,
    EMAIL_MARKER,
    ContractViolationError,
    EmailClassification,
    EmailInput,
    PromptConfig,
    PromptSpec,
)
from classifier import classify_email  # noqa: E402
from evaluator import (  # noqa: E402
    GOLDEN_PATH,
    JudgeOutcome,
    Metrics,
    RunResult,
    Verdict,
    assemble_run,
    CaseResult,
    classify_run_status,
    compare_to_baseline,
    default_baseline_key,
    evaluate,
    heuristic_judge_score,
    judge_summary,
    judge_summary_async,
    load_golden_dataset,
    parse_judge_score,
    percentile,
    read_dataset_version,
    run_evaluation,
)
from prompt_registry import (  # noqa: E402
    PROMPTS_DIR,
    PromptNotFoundError,
    PromptValidationError,
    list_versions,
    load_prompt,
    load_prompt_config,
    resolve_version,
)
from providers import MockClient, available_providers, get_client  # noqa: E402
from runner import Completion, get_async_client, run_batch  # noqa: E402
from storage import SQLiteStore  # noqa: E402

def _prompt(version: str, **overrides) -> PromptSpec:
    """Build a minimal valid PromptSpec for a given version."""
    data = {
        "version": version,
        "created_at": "2026-01-01T00:00:00Z",
        "system_prompt": "classify",
        "categories": list(CATEGORIES),
    }
    data.update(overrides)
    return PromptSpec(**data)


def _contract(spec: PromptSpec) -> PromptConfig:
    return PromptConfig.from_spec(spec)

SAMPLES = [
    {"id": "a", "expected": "billing", "body": "You charged my card twice, I want a refund."},
    {"id": "b", "expected": "technical", "body": "The API returns error 500 and the server is down."},
    {"id": "c", "expected": "account", "body": "I cannot sign in, my password reset never arrives."},
    {"id": "d", "expected": "general", "body": "Sharing some feedback and a question about your roadmap."},
]


class AlwaysGeneral:
    name = "mock-broken"

    def complete(self, system_prompt, user_prompt):
        return json.dumps({"category": "general", "summary": "help"})


# --- providers ------------------------------------------------------------


def test_default_provider_is_mock():
    assert get_client().name == "mock"


def test_explicit_provider():
    assert get_client("mock").name == "mock"


def test_unknown_provider_raises():
    try:
        get_client("does-not-exist")
    except ValueError as exc:
        assert "Unknown provider" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_available_providers_lists_all_five():
    assert available_providers() == [
        "anthropic", "gemini", "mock", "ollama", "openai",
    ]


# --- classifier -----------------------------------------------------------


def test_classify_returns_valid_category():
    out = classify_email("I was overcharged, please refund my invoice.", client=MockClient())
    assert out["category"] == "billing"


def test_classify_shape_is_dict_with_expected_keys():
    out = classify_email("Server is broken, error 500.", client=MockClient())
    assert isinstance(out, dict)
    assert set(out) == {"category", "summary", "provider", "prompt_version"}


def test_summary_is_one_short_sentence():
    out = classify_email("Your invoice is wrong. Please refund it.", client=MockClient())
    assert out["summary"]
    assert len(out["summary"].split()) <= 21  # 20 words + optional "..."


def test_empty_email_raises():
    try:
        classify_email("   ", client=MockClient())
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_label_outside_closed_set_fails_at_contract():
    """
    An invented label is STRUCTURAL: it means the model ignored the contract, so
    the contract refuses to build it. Under the old inline checks this was
    silently folded to 'general'.
    """
    class Rude:
        name = "rude"

        def complete(self, s, u):
            return json.dumps({"category": "refund_request", "summary": "x"})

    try:
        classify_email("hello there", client=Rude())
    except ContractViolationError:
        pass
    else:
        raise AssertionError("expected ContractViolationError for an invented label")


def test_direct_construction_cannot_bypass_the_contract():
    """The output type itself is the guarantee, not just the factory."""
    try:
        EmailClassification(category="refund_request", summary="ok")  # type: ignore[arg-type]
    except Exception as exc:  # pydantic reports a literal_error
        assert "refund_request" in str(exc)
    else:
        raise AssertionError("expected failure for a label outside the closed set")

    try:
        EmailClassification(
            category="general",
            summary="ok",
            prompt_version="v2",
            allowed_categories=("billing", "technical", "account"),
        )
    except Exception as exc:  # ContractViolationError subclasses ValueError
        assert "not allowed by prompt version" in str(exc)
    else:
        raise AssertionError("expected failure for a label v2 forbids")


def test_prompt_is_a_parameter_not_hardcoded():
    """Two PromptConfigs, same email -> the config value reaches the output."""
    a = classify_email("hi", prompt=_prompt("vA"), client=MockClient())
    b = classify_email("hi", prompt=_prompt("vB"), client=MockClient())
    assert a["prompt_version"] == "vA"
    assert b["prompt_version"] == "vB"


def test_fenced_json_is_parsed():
    class Fenced:
        name = "fenced"

        def complete(self, s, u):
            return '```json\n{"category": "billing", "summary": "charge"}\n```'

    assert classify_email("x", client=Fenced())["category"] == "billing"


def test_non_json_response_raises():
    class Chatty:
        name = "chatty"

        def complete(self, s, u):
            return "Sure! Here is the email: hope you have a nice day."

    try:
        classify_email("x", client=Chatty())
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")


# --- evaluator ------------------------------------------------------------


def test_evaluate_on_labelled_samples():
    run = evaluate(client=MockClient(), samples=SAMPLES)
    assert run.n_samples == 4
    assert run.accuracy == 1.0
    assert run.metrics.n_correct == 4


def test_evaluate_is_deterministic():
    a = evaluate(client=MockClient(), samples=SAMPLES)
    b = evaluate(client=MockClient(), samples=SAMPLES)
    assert a.accuracy == b.accuracy
    assert [p.category for p in a.predictions] == [p.category for p in b.predictions]


def test_broken_client_scores_low():
    run = evaluate(client=AlwaysGeneral(), samples=SAMPLES)
    assert run.accuracy == 0.25  # only the 'general' sample is right


# --- storage + verdicts ---------------------------------------------------


def _tmp_store(tmp="mrds_test.sqlite3"):
    """Fresh store per call: delete any leftover file so tests are repeatable."""
    path = Path(__file__).parent / "data" / tmp
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return SQLiteStore(path)


def _pin_and_compare(store, run, key: str, threshold: float = 0.0):
    """
    Persist a run, then compare it against the named baseline.

    "Pinning" means: this run must land in the database before we judge it,
    otherwise a real incident would be visible in a verdict but not in the
    stored history. Three tests need exactly this sequence, so it lives here
    once instead of being pasted three times.
    """
    store.save_run(
        {
            "run_id": run.run_id, "created_at": run.created_at,
            "provider": run.provider, "prompt_version": run.prompt_version,
            "n_samples": run.n_samples, "accuracy": run.accuracy,
            "metrics": run.metrics.model_dump(),
        },
        [p.model_dump() for p in run.predictions],
    )
    return compare_to_baseline(run, store=store, baseline_key=key, threshold=threshold)


def test_baseline_created_then_pass_then_regression():
    store = _tmp_store()
    run1, v1 = run_evaluation(client=MockClient(), store=store, baseline_key="t1")
    assert v1.status == "BASELINE_CREATED"
    assert v1.regression_detected is False

    run2, v2 = run_evaluation(client=MockClient(), store=store, baseline_key="t1")
    assert v2.status == "PASS"
    assert v2.accuracy_delta == 0.0

    bad = evaluate(client=AlwaysGeneral(), samples=SAMPLES)
    v3 = _pin_and_compare(store, bad, "t1")
    # threshold=0.0 is the legacy single cut-off: any drop past it is the
    # critical verdict (Phase 3's binary mode), never a gentle WARN.
    assert v3.status == "CRITICAL_REGRESSION"
    assert v3.regression_detected is True
    assert v3.accuracy_delta == -0.75
    store.close()


def test_threshold_tolerates_small_drop():
    store = _tmp_store(tmp="mrds_test_thresh.sqlite3")
    run_evaluation(client=MockClient(), store=store, baseline_key="t2")
    bad = evaluate(client=AlwaysGeneral(), samples=SAMPLES)
    # Allowed drop of 1.0 -> a 0.75 drop is tolerated.
    v = _pin_and_compare(store, bad, "t2", threshold=1.0)
    assert v.status == "PASS"
    store.close()


def test_run_roundtrips_through_store():
    store = _tmp_store(tmp="mrds_test_rt.sqlite3")
    run, _ = run_evaluation(client=MockClient(), store=store, baseline_key="t3")
    loaded = store.get_run(run.run_id)
    assert loaded["accuracy"] == run.accuracy
    assert len(store.list_runs()) >= 1
    store.close()


def test_json_backend_parity():
    store = _tmp_store(tmp="mrds_test_parity.sqlite3")
    run_evaluation(client=MockClient(), store=store, baseline_key="t4")
    base = store.get_baseline("t4")
    assert base is not None and base["accuracy"] == 1.0


def test_verdict_is_valid_pydantic_payload():
    store = _tmp_store(tmp="mrds_test_verdict.sqlite3")
    _, v = run_evaluation(client=MockClient(), store=store, baseline_key="t5")
    assert Verdict.model_validate(json.loads(v.model_dump_json())) == v
    store.close()


# --- prompt registry (prompts/*.yaml) ------------------------------------


def test_registry_lists_both_versions():
    versions = list_versions()
    assert "v1" in versions and "v2" in versions


def test_load_prompt_returns_file_version():
    prompt = load_prompt("v1")
    assert prompt.version == "v1"
    assert prompt.created_at.startswith("20")
    assert "billing" in prompt.categories


def test_v1_and_v2_are_genuinely_different():
    v1, v2 = load_prompt("v1"), load_prompt("v2")
    assert v1.system_prompt != v2.system_prompt
    assert tuple(v1.categories) != tuple(v2.categories)
    assert len(v2.few_shots) > 0


def test_unknown_version_raises_loudly():
    try:
        load_prompt("v99")
    except PromptNotFoundError as exc:
        assert "v1" in str(exc)  # error lists what IS available
    else:
        raise AssertionError("expected PromptNotFoundError")


def test_invalid_prompt_file_raises(tmp_prompt_dir_factory=None):
    """A prompt missing required fields is structural -> raise, not degrade."""
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "v1.yaml").write_text(
            "version: v1\nsystem_prompt: hi\n", encoding="utf-8"
        )  # no created_at, no categories
        try:
            load_prompt("v1", limit_dir=folder)
        except PromptValidationError as exc:
            assert "created_at" in str(exc)
        else:
            raise AssertionError("expected PromptValidationError")


def test_broken_yaml_raises():
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "v1.yaml").write_text("version: [unclosed\n", encoding="utf-8")
        try:
            load_prompt("v1", limit_dir=folder)
        except PromptValidationError:
            pass
        else:
            raise AssertionError("expected PromptValidationError")


def test_filename_version_mismatch_raises():
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "v1.yaml").write_text(
            "version: v2\ncreated_at: '2026-01-01T00:00:00Z'\n"
            "system_prompt: hi\ncategories: [billing]\n",
            encoding="utf-8",
        )
        try:
            load_prompt("v1", limit_dir=folder)
        except PromptValidationError as exc:
            assert "declares version" in str(exc)
        else:
            raise AssertionError("expected PromptValidationError")


def test_unknown_label_in_prompt_raises():
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "v1.yaml").write_text(
            "version: v1\ncreated_at: '2026-01-01T00:00:00Z'\n"
            "system_prompt: hi\ncategories: [billing, unicorn]\n",
            encoding="utf-8",
        )
        try:
            load_prompt("v1", limit_dir=folder)
        except PromptValidationError:
            pass
        else:
            raise AssertionError("expected PromptValidationError")


def test_resolve_version_prefers_arg_then_env(monkeypatch=None):
    import os

    assert resolve_version("v2") == "v2"
    os.environ["PROMPT_VERSION"] = "v2"
    try:
        assert resolve_version() == "v2"
        assert resolve_version("v1") == "v1"  # explicit arg wins
    finally:
        os.environ.pop("PROMPT_VERSION", None)
    assert resolve_version() == "v1"  # default


# --- prompt plumbing: few-shots must not leak into the classified text -----


def test_few_shots_go_to_system_turn_not_user_turn():
    v2 = load_prompt("v2")
    system = _contract(v2).build_system_prompt()
    user = _contract(v2).build_user_prompt(EmailInput(body="Actual customer email text."))
    assert "Examples:" in system
    assert "Examples:" not in user
    assert "Actual customer email text." in user


def test_user_turn_shape_identical_across_versions():
    v1, v2 = load_prompt("v1"), load_prompt("v2")
    email = EmailInput(body="Same email for both prompts.")
    for spec in (v1, v2):
        user = _contract(spec).build_user_prompt(email)
        assert EMAIL_MARKER in user
        assert user.split(EMAIL_MARKER, 1)[1].strip() == email.body


def test_system_prompt_declares_allowed_labels():
    v2 = load_prompt("v2")
    assert "ALLOWED_LABELS: billing, technical, account" in _contract(v2).build_system_prompt()


def test_mock_summary_contains_only_email_not_few_shots():
    out = classify_email(
        "Your invoice is wrong. Please refund it.",
        prompt=load_prompt("v2"),
        client=MockClient(),
    )
    assert "Examples:" not in out["summary"]
    assert "invoice" in out["summary"].lower()


# --- v1 vs v2 must be distinguishable end to end -------------------------


def test_v1_and_v2_produce_different_accuracies():
    run1 = evaluate(client=MockClient(), samples=SAMPLES, prompt_version="v1")
    run2 = evaluate(client=MockClient(), samples=SAMPLES, prompt_version="v2")
    assert run1.prompt_version == "v1"
    assert run2.prompt_version == "v2"
    assert run1.accuracy == 1.0
    assert run2.accuracy < run1.accuracy  # the lever works


def test_baseline_keys_are_scoped_per_prompt_version():
    r1 = evaluate(client=MockClient(), samples=SAMPLES, prompt_version="v1")
    r2 = evaluate(client=MockClient(), samples=SAMPLES, prompt_version="v2")
    # samples/emails.json carries no dataset_version, so it reports "unknown" -
    # a sentinel that matches no real dataset and so can never collide with one.
    assert default_baseline_key(r1) == "mock:v1:unknown"
    assert default_baseline_key(r2) == "mock:v2:unknown"


# --- dataset_version in the baseline key ----------------------------------
#
# The bug this guards against: editing the golden set while old baselines are
# still keyed only on "<provider>:<prompt_version>". Every stored baseline then
# silently repoints at the new set, and the resulting accuracy change is
# reported as a model regression when really the BAR moved.


def test_baseline_keys_are_scoped_per_dataset_version():
    r04 = evaluate(client=MockClient(), samples=SAMPLES, prompt_version="v1",
                   dataset_version="0.4")
    r05 = evaluate(client=MockClient(), samples=SAMPLES, prompt_version="v1",
                   dataset_version="0.5")
    r04again = evaluate(client=MockClient(), samples=SAMPLES, prompt_version="v1",
                        dataset_version="0.4")
    assert default_baseline_key(r04) == "mock:v1:0.4"
    assert default_baseline_key(r05) == "mock:v1:0.5"
    assert default_baseline_key(r04) != default_baseline_key(r05)
    # Same provider + prompt + dataset always resolves to the same key, so a
    # repeat run compares against its own baseline rather than creating a new one.
    assert default_baseline_key(r04) == default_baseline_key(r04again)


def test_changing_dataset_gets_its_own_baseline_not_a_regression():
    """A model that scores 1.0 on 0.4 and 0.25 on 0.5 did not regress - the
    test set changed. Without dataset_version this reports REGRESSION."""
    store = _tmp_store(tmp="mrds_test_datasetscope.sqlite3")
    _, v04 = run_evaluation(client=MockClient(), store=store, samples=SAMPLES,
                            prompt_version="v1", dataset_version="0.4")
    assert v04.status == "BASELINE_CREATED"

    # Deliberately awful model, scored against a DIFFERENT dataset version.
    _, v05 = run_evaluation(client=AlwaysGeneral(), store=store, samples=SAMPLES,
                            prompt_version="v1", dataset_version="0.5")
    assert v05.status == "BASELINE_CREATED"
    assert v05.regression_detected is False

    # ...while the same awful model on the SAME dataset still trips its baseline.
    broken = evaluate(client=AlwaysGeneral(), samples=SAMPLES,
                      prompt_version="v1", dataset_version="0.4")
    worse = _pin_and_compare(store, broken, "mock:v1:0.4")
    assert worse.status == "CRITICAL_REGRESSION"
    store.close()


def test_read_dataset_version_from_file():
    # The golden set names itself...
    assert read_dataset_version(GOLDEN_PATH) == "0.4"
    # ...and the built-in samples file does not, so it reports the sentinel
    # rather than guessing a version it cannot know.
    assert read_dataset_version(Path(__file__).parent / "samples" / "emails.json") == "unknown"


# --- the golden dataset, loaded through the contract -----------------------


def _write_tmp_golden(doc: dict, name: str = "mrds_test_golden.json") -> Path:
    """Write a golden-shaped doc into the test data dir. Same convention as
    _tmp_store: no pytest fixtures, so the file is cleaned by overwrite."""
    path = Path(__file__).parent / "data" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_golden_dataset_loads_and_types_through_the_contract():
    rows, version = load_golden_dataset()
    assert version == "0.4"
    assert len(rows) >= 100
    assert len({r["id"] for r in rows}) == len(rows), "golden ids must be unique"

    for row in rows:
        assert row["expected"] in CATEGORIES
        assert row["body"].strip(), f"{row['id']} typed a blank body"
        # Re-type every row: this is the contract accepting the whole dataset.
        EmailInput(body=row["body"], subject=row["subject"], email_id=row["id"])


def test_golden_run_carries_dataset_version_into_its_baseline_key():
    rows, version = load_golden_dataset()
    run = evaluate(client=MockClient(), samples=rows, dataset_version=version)
    assert run.dataset_version == "0.4"
    assert default_baseline_key(run) == "mock:v1:0.4"


def test_unverified_golden_dataset_is_refused():
    """A set still under review is not a bar. Baselining on it would measure
    reviewer progress instead of the classifier."""
    path = _write_tmp_golden({
        "dataset_version": "9.9",
        "status": "awaiting_human_confirmation",
        "provenance": {"human_verified": False},
        "records": [{
            "id": "g1",
            "input": "My card was charged twice.",
            "human_verified_category": "billing",
            "human_verified_ideal_summary": "Duplicate charge.",
        }],
    })
    try:
        load_golden_dataset(path)
        raise AssertionError("unverified dataset should have been refused")
    except ValueError as exc:
        assert "NOT human-verified" in str(exc)

    # Inspection is still allowed when asked for explicitly - only building a
    # baseline from it is forbidden.
    rows, version = load_golden_dataset(path, require_verified=False)
    assert version == "9.9" and len(rows) == 1


def test_golden_row_with_inline_subject_is_split_not_duplicated():
    """`Subj:` lines move into the typed subject field; the body keeps
    everything else, so no reviewed text is dropped or counted twice."""
    path = _write_tmp_golden({
        "dataset_version": "9.9",
        "status": "human_verified",
        "provenance": {"human_verified": True},
        "records": [
            {"id": "g1", "input": "My card was charged twice.",
             "human_verified_category": "billing",
             "human_verified_ideal_summary": "Duplicate charge."},
            {"id": "g2", "input": "Subj: Refund please\nI want my money back.",
             "human_verified_category": "general",
             "human_verified_ideal_summary": "Customer wants a refund."},
        ],
    }, name="mrds_test_golden_subject.json")
    rows, _ = load_golden_dataset(path)

    assert rows[0]["subject"] == "" and rows[0]["body"] == "My card was charged twice."
    assert rows[1]["subject"] == "Refund please"
    assert rows[1]["body"] == "I want my money back."
    assert "Refund please" not in rows[1]["body"], "subject must not appear twice"


def test_golden_row_with_bad_category_is_refused():
    path = _write_tmp_golden({
        "dataset_version": "9.9",
        "status": "human_verified",
        "provenance": {"human_verified": True},
        "records": [{
            "id": "g1", "input": "Please refund me.",
            "human_verified_category": "refund_request",
            "human_verified_ideal_summary": "Customer wants a refund.",
        }],
    }, name="mrds_test_golden_badcat.json")
    try:
        load_golden_dataset(path)
        raise AssertionError("label outside the closed set should have been refused")
    except ValueError as exc:
        assert "invalid category" in str(exc)


def test_runs_stored_under_different_prompt_version_column():
    store = _tmp_store(tmp="mrds_test_promptver.sqlite3")
    run1, v1_verdict = run_evaluation(
        client=MockClient(), store=store, samples=SAMPLES, prompt_version="v1"
    )
    run2, v2_verdict = run_evaluation(
        client=MockClient(), store=store, samples=SAMPLES, prompt_version="v2"
    )
    stored1 = store.get_run(run1.run_id)
    stored2 = store.get_run(run2.run_id)
    assert stored1["prompt_version"] == "v1"
    assert stored2["prompt_version"] == "v2"
    # Each prompt version got its OWN first observation, not a cross-comparison.
    assert v1_verdict.status == "BASELINE_CREATED"
    assert v2_verdict.status == "BASELINE_CREATED"
    assert store.get_baseline("mock:v1:unknown") is not None
    assert store.get_baseline("mock:v2:unknown") is not None
    store.close()


def test_v2_does_not_trip_v1_baseline():
    """A different prompt is a different contract, never a v1 regression."""
    store = _tmp_store(tmp="mrds_test_no_cross.sqlite3")
    run_evaluation(client=MockClient(), store=store, samples=SAMPLES, prompt_version="v1")
    _, verdict = run_evaluation(
        client=MockClient(), store=store, samples=SAMPLES, prompt_version="v2"
    )
    assert verdict.status == "BASELINE_CREATED"
    assert verdict.regression_detected is False
    store.close()


def test_v2_degrades_against_its_own_baseline():
    """Pin v2's baseline, then break the model: v2 must report REGRESSION."""
    store = _tmp_store(tmp="mrds_test_v2reg.sqlite3")
    run_evaluation(client=MockClient(), store=store, samples=SAMPLES, prompt_version="v2")
    good = evaluate(client=MockClient(), samples=SAMPLES, prompt_version="v2")
    same = _pin_and_compare(store, good, "mock:v2:unknown")
    assert same.status == "PASS"
    broken = evaluate(client=AlwaysGeneral(), samples=SAMPLES, prompt_version="v2")
    # Baseline keys are "<provider>:<prompt_version>:<dataset_version>". The
    # broken client reports provider "mock-broken", so we pin the v2 mock
    # baseline explicitly — a different provider legitimately gets its own
    # baseline, and so does a different dataset.
    worse = _pin_and_compare(store, broken, "mock:v2:unknown")
    assert worse.status == "CRITICAL_REGRESSION"
    assert worse.accuracy_delta == -0.5
    store.close()


# --- the typed contract (contract.py) -------------------------------------


def test_valid_output_round_trips_through_the_contract():
    email = EmailInput(
        body="You charged my card twice. Refund please.", subject="Charge", email_id="e1"
    )
    config = _contract(load_prompt("v1"))
    typed = config.parse_classification(
        json.dumps({"category": "billing", "summary": "Duplicate charge."}),
        provider="mock",
    )
    assert typed.category == "billing"
    as_dict = typed.model_dump()  # what storage consumes
    assert as_dict == {
        "category": "billing",
        "summary": "Duplicate charge.",
        "provider": "mock",
        "prompt_version": "v1",
    }
    assert "allowed_categories" not in as_dict  # validation context, not an answer
    assert email.body.startswith("You charged")


def test_summary_of_wrong_type_fails_at_the_boundary():
    config = _contract(load_prompt("v1"))
    for bad in [{"category": "billing", "summary": 42},
                {"category": "billing", "summary": "   "},
                {"category": "billing"}]:
        try:
            config.parse_classification(json.dumps(bad))
        except ContractViolationError:
            pass
        else:
            raise AssertionError(f"expected ContractViolationError for {bad}")


def test_blank_email_body_fails_at_the_boundary():
    try:
        EmailInput(body="   ")
    except Exception:
        pass
    else:
        raise AssertionError("expected failure for a blank body")

    try:
        classify_email("", client=MockClient())
    except ContractViolationError:
        pass
    else:
        raise AssertionError("expected ContractViolationError for an empty email")


def test_v2_reduced_label_set_folds_recoverably():
    """v2 forbids `general` but it is a valid MRDS label -> fold, not crash."""
    config = _contract(load_prompt("v2"))
    assert config.allowed_categories == ("billing", "technical", "account")
    assert config.fallback_category == "account"
    folded = config.parse_classification(
        json.dumps({"category": "general", "summary": "feedback"})
    )
    assert folded.category == "account"  # v2's own last label
    assert config.parse_classification(
        json.dumps({"category": "billing", "summary": "refund"})
    ).category == "billing"


def test_v2_still_rejects_labels_outside_the_closed_set():
    config = _contract(load_prompt("v2"))
    try:
        config.parse_classification(
            json.dumps({"category": "refund_request", "summary": "x"})
        )
    except ContractViolationError as exc:
        assert "closed label set" in str(exc)
    else:
        raise AssertionError("expected ContractViolationError")


def test_prompt_config_wraps_a_prompt_spec():
    config = _contract(load_prompt("v2"))
    assert isinstance(config.spec, PromptSpec)
    assert config.version == config.spec.version  # never re-derived
    assert load_prompt_config("v2").version == "v2"


def test_classifier_accepts_str_or_typed_input():
    as_str = classify_email("Server returns error 500.", client=MockClient())
    as_model = classify_email(
        EmailInput(body="Server returns error 500."), client=MockClient()
    )
    assert as_str["category"] == as_model["category"] == "technical"


def test_classifier_accepts_a_bare_prompt_spec():
    out = classify_email("hi there", prompt=_prompt("hand-built"), client=MockClient())
    assert out["prompt_version"] == "hand-built"


def test_dict_output_still_round_trips_through_sqlite():
    store = _tmp_store(tmp="mrds_test_contract.sqlite3")
    run, _ = run_evaluation(
        client=MockClient(), store=store, samples=SAMPLES, prompt_version="v1"
    )
    loaded = store.get_run(run.run_id)
    assert loaded["prompt_version"] == "v1"
    assert loaded["accuracy"] == 1.0
    # The dict a prediction round-trips into storage unchanged.
    typed = EmailClassification(**run.predictions[0].model_dump())
    assert typed.category == run.predictions[0].category
    store.close()


def test_evaluator_rejects_a_classifier_that_breaks_the_contract():
    """The seam works both ways: a bad implementation fails at the boundary."""
    class BadImpl:
        name = "mock-bad"

        def complete(self, system_prompt, user_prompt):
            return json.dumps({"category": "billing"})  # no summary

    from evaluator import evaluate

    try:
        evaluate(client=BadImpl(), samples=SAMPLES, prompt_version="v1")
    except ContractViolationError:
        pass
    else:
        raise AssertionError("expected ContractViolationError")


# --- Phase 3: async runner, multi-dimensional scoring, diff, thresholds ----
#
# Every test below runs on mock / fake clients. No network, no keys, no cost -
# the async machinery is judged on its OWN behaviour (bounding, batching,
# measuring), not on what a live model happens to say today.


class CountingAsyncClient:
    """Fake async provider that records how many calls are in flight at once -
    the raw evidence that the semaphore actually bounds concurrency.
    Named 'mock', so the judge uses its offline heuristic (no calls counted)."""

    name = "mock"

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self._sync = MockClient()

    async def acomplete(self, system_prompt, user_prompt):
        self.active += 1
        self.calls += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(self.delay)  # hold the slot, like a real slow call
        text = self._sync.complete(system_prompt, user_prompt)
        self.active -= 1
        return Completion(text=text, prompt_tokens=7, completion_tokens=3)


class FakeLLMAsyncClient:
    """Async client that LOOKS like a real provider (name 'ollama'), so the
    LLM-as-Judge path runs through it too. Answers by prompt shape."""

    name = "ollama"

    async def acomplete(self, system_prompt, user_prompt):
        # Judge calls carry the rubric in the system turn and the marker in
        # the user turn; classification calls never contain this marker.
        if "SUMMARY TO GRADE" in user_prompt:
            return Completion(text='{"score": 4}', prompt_tokens=5, completion_tokens=2)
        return Completion(
            text=json.dumps({"category": "billing", "summary": "charged twice"}),
            prompt_tokens=11,
            completion_tokens=6,
        )


def test_async_runner_caps_concurrency_and_covers_every_case():
    config = load_prompt_config("v1")
    rows = [
        {"id": f"x{i}", "expected": "billing", "subject": "",
         "body": "You charged my card twice, I want a refund."}
        for i in range(9)
    ]
    client = CountingAsyncClient()
    cases, duration = asyncio.run(
        run_batch(rows, config, client, concurrency=3, judge=False)
    )
    assert client.calls == 9, "every case must run exactly once"
    # Parallelism happened, but never past the semaphore: max 3 in flight.
    assert 2 <= client.max_active <= 3, f"max in flight was {client.max_active}"
    assert len(cases) == 9
    assert duration > 0


def test_async_runner_reports_progress_for_every_case():
    config = load_prompt_config("v1")
    seen: list[tuple[int, int]] = []
    asyncio.run(
        run_batch(
            SAMPLES, config, CountingAsyncClient(), concurrency=2, judge=False,
            on_progress=lambda done, total: seen.append((done, total)),
        )
    )
    assert [d for d, _ in seen] == [1, 2, 3, 4]
    assert seen[-1] == (4, 4)


def test_runner_and_sync_evaluator_agree():
    """Drift guard: the async path builds prompts through the same PromptConfig
    and validates through the same contract as classify_email - so it must
    produce identical answers, not merely similar ones."""
    config = load_prompt_config("v1")
    async_client = get_async_client("mock")
    cases, _ = asyncio.run(run_batch(SAMPLES, config, async_client, judge=False))
    run_async = assemble_run(
        cases, provider="mock", prompt_version="v1", dataset_version="unknown"
    )
    run_sync = evaluate(client=MockClient(), samples=SAMPLES, prompt_version="v1")
    assert run_async.accuracy == run_sync.accuracy
    assert [(p.sample_id, p.category) for p in run_async.predictions] == [
        (p.sample_id, p.category) for p in run_sync.predictions
    ]


def test_runner_records_latency_tokens_and_per_case_metrics():
    config = load_prompt_config("v1")
    client = CountingAsyncClient()  # 7 prompt / 3 completion tokens per call
    cases, duration = asyncio.run(run_batch(SAMPLES, config, client, judge=True))
    run = assemble_run(
        cases, provider="mock", prompt_version="v1",
        dataset_version="unknown", duration_ms=duration,
    )
    tokens = run.metrics.tokens
    assert tokens.prompt_tokens == 7 * 4
    assert tokens.completion_tokens == 3 * 4
    assert tokens.total_tokens == 40
    lat = run.metrics.latency
    assert lat.n == 4 and lat.p50_ms > 0 and lat.p90_ms >= lat.p50_ms
    assert run.metrics.run_duration_ms > 0
    for p in run.predictions:
        assert p.case_metrics is not None
        assert p.case_metrics.total_tokens == 10
        assert p.case_metrics.latency_ms > 0
        assert p.case_metrics.judge_source == "heuristic"  # 'mock' -> offline


def test_llm_judge_path_records_score_and_its_own_tokens():
    config = load_prompt_config("v1")
    cases, _ = asyncio.run(
        run_batch(SAMPLES, config, FakeLLMAsyncClient(), concurrency=2, judge=True)
    )
    for c in cases:
        assert c.judge is not None
        assert c.judge.score == 4 and c.judge.source == "llm"
    run = assemble_run(
        cases, provider="ollama", prompt_version="v1", dataset_version="unknown"
    )
    assert run.metrics.judge.mean_score == 4.0
    assert run.metrics.judge.n_llm == 4 and run.metrics.judge.n_heuristic == 0
    # classification usage (11+6 per case) and judge usage tracked SEPARATELY -
    # a judge call is not a classification call, and conflating them would
    # make token trends unreadable when the judge is toggled on or off.
    assert run.metrics.tokens.prompt_tokens == 11 * 4
    assert run.metrics.tokens.completion_tokens == 6 * 4
    assert run.metrics.tokens.total_tokens == 17 * 4
    assert run.metrics.tokens.judge_prompt_tokens == 5 * 4
    assert run.metrics.tokens.judge_completion_tokens == 2 * 4
    assert run.metrics.tokens.judge_total_tokens == 7 * 4


def test_async_client_factory_follows_provider_rules():
    assert get_async_client("mock").name == "mock"
    assert get_async_client().name == "mock"  # offline default
    try:
        get_async_client("does-not-exist")
    except ValueError as exc:
        assert "Unknown provider" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_percentile_uses_nearest_rank():
    values = [float(i) for i in range(1, 11)]
    assert percentile(values, 50) == 5.0
    assert percentile(values, 90) == 9.0
    assert percentile([42.0], 90) == 42.0
    assert percentile([], 50) == 0.0


def test_exact_match_drives_the_pass_rate():
    run = evaluate(client=AlwaysGeneral(), samples=SAMPLES)
    assert run.accuracy == 0.25  # pass rate == exact-match fraction, by definition
    assert {p.sample_id: p.correct for p in run.predictions} == {
        "a": False, "b": False, "c": False, "d": True,
    }


def test_heuristic_judge_bands():
    ideal = (
        "The customer is requesting a refund for order 67001 "
        "after the laptop arrived damaged."
    )
    assert heuristic_judge_score(ideal, ideal, "any body") == 5  # fully grounded
    assert heuristic_judge_score("", ideal, "any body") == 1  # said nothing
    assert (
        heuristic_judge_score("Your account balance is five million", ideal, "refund")
        == 1
    )  # invented - nothing ties to the reference
    assert heuristic_judge_score("customer wants a pony", ideal, "") == 2  # weak link
    # no reference at all -> cannot judge -> middle of the scale, not a guess
    assert heuristic_judge_score("words here", None, "") == 3


def test_parse_judge_score_is_tolerant_but_bounded():
    assert parse_judge_score('{"score": 4}') == 4
    assert parse_judge_score('```json\n{"score": 5}\n```') == 5
    assert parse_judge_score('Here you go: {"score": 2}. Done.') == 2
    assert parse_judge_score('{"score": 9}') is None  # outside the rubric
    assert parse_judge_score('{"score": "four"}') is None
    assert parse_judge_score("looks great to me") is None
    assert parse_judge_score("") is None


def test_judge_falls_back_gracefully_when_offline_or_mocked():
    # mocked provider -> heuristic straight away, no model call
    out = judge_summary("refund for order 67001", "refund request", "body",
                        client=MockClient())
    assert out.source == "heuristic" and 1 <= out.score <= 5

    # provider that raises (server down) -> heuristic, no exception escapes
    class Offline:
        name = "ollama"

        def complete(self, s, u):
            raise ConnectionError("network down")

    out2 = judge_summary("refund for order 67001", "refund request", "body",
                         client=Offline())
    assert out2.source == "heuristic" and out2.score is not None

    # provider answers with chatter, not a score -> heuristic
    class Chatty:
        name = "ollama"

        def complete(self, s, u):
            return "That summary looks great to me!"

    out3 = judge_summary("refund for order 67001", "refund request", "body",
                         client=Chatty())
    assert out3.source == "heuristic"


def test_async_judge_disabled_reports_none_not_a_fake_score():
    off = asyncio.run(judge_summary_async("s", "ideal", "body", enabled=False))
    assert off.score is None and off.source == "none"
    on = asyncio.run(judge_summary_async("s", "ideal reference words", "body"))
    assert on.source == "heuristic" and on.score is not None  # no client -> offline


def test_sync_evaluate_judge_switch():
    run = evaluate(client=MockClient(), samples=SAMPLES, judge=True)
    stats = run.metrics.judge
    assert stats.n_scored == 4 and stats.n_heuristic == 4 and stats.n_llm == 0
    assert 1 <= stats.mean_score <= 5
    # judge defaults OFF on the sync path (keeps Phase 2 baseline.py timing)
    quiet = evaluate(client=MockClient(), samples=SAMPLES)
    assert quiet.metrics.judge.n_scored == 0
    assert all(p.case_metrics.judge_score is None for p in quiet.predictions)


def test_assemble_run_aggregates_all_four_axes():
    cases = [
        CaseResult(sample_id="a", expected="billing", category="billing",
                   summary="charged", latency_ms=10.0,
                   prompt_tokens=10, completion_tokens=5,
                   judge=JudgeOutcome(score=4, source="llm")),
        CaseResult(sample_id="b", expected="technical", category="general",
                   summary="hmm", latency_ms=30.0,
                   prompt_tokens=20, completion_tokens=5,
                   judge=JudgeOutcome(score=2, source="heuristic")),
    ]
    run = assemble_run(cases, provider="mock", prompt_version="v1",
                       dataset_version="0.4", duration_ms=1234.0)
    # (a) exact match
    assert run.accuracy == 0.5
    assert run.metrics.per_category_total == {"billing": 1, "technical": 1}
    # (c) latency
    assert run.metrics.latency.p50_ms == 10.0
    assert run.metrics.latency.p90_ms == 30.0
    assert run.metrics.run_duration_ms == 1234.0
    # (d) tokens
    assert run.metrics.tokens.total_tokens == 40
    # (b) judge
    assert run.metrics.judge.mean_score == 3.0
    assert run.metrics.judge.n_llm == 1 and run.metrics.judge.n_heuristic == 1
    assert run.predictions[1].case_metrics.judge_score == 2


# --- diffing + status classification ---------------------------------------


def _flat_run(acc: float, n: int = 100) -> RunResult:
    """Hand-built run with a chosen accuracy: verdict classification is a
    statement about NUMBERS, so it must be testable without a model."""
    correct = round(acc * n)
    return RunResult(
        run_id=f"fake_{acc}",
        created_at="2026-10-19T00:00:00+00:00",
        provider="mock",
        prompt_version="v1",
        dataset_version="0.4",
        n_samples=n,
        accuracy=correct / n,
        metrics=Metrics(accuracy=correct / n, n_samples=n, n_correct=correct),
    )


def _pin_accuracy(store, key: str, acc: float, n: int = 100) -> None:
    run = _flat_run(acc, n)
    store.set_baseline(key, {
        "run_id": run.run_id, "created_at": run.created_at,
        "provider": run.provider, "prompt_version": run.prompt_version,
        "n_samples": run.n_samples, "accuracy": run.accuracy,
        "metrics": run.metrics.model_dump(),
    })


def test_run_status_boundaries_are_exact():
    assert classify_run_status(+0.05) == "PASS"  # improvement
    assert classify_run_status(0.0) == "PASS"
    assert classify_run_status(-0.03) == "PASS"  # drop exactly AT warn line
    assert classify_run_status(-0.030001) == "WARN"
    assert classify_run_status(-0.08) == "WARN"  # drop exactly AT critical line
    assert classify_run_status(-0.080001) == "CRITICAL_REGRESSION"
    # thresholds are configurable, not hardcoded:
    assert classify_run_status(-0.05, warn_threshold=0.10,
                               critical_threshold=0.20) == "PASS"
    assert classify_run_status(-0.15, warn_threshold=0.10,
                               critical_threshold=0.20) == "WARN"
    assert classify_run_status(-0.25, warn_threshold=0.10,
                               critical_threshold=0.20) == "CRITICAL_REGRESSION"


def test_verdict_statuses_end_to_end_against_a_stored_baseline():
    store = _tmp_store(tmp="mrds_test_p3status.sqlite3")
    _pin_accuracy(store, "k", 0.50)
    v_pass = compare_to_baseline(_flat_run(0.48), store=store, baseline_key="k")
    v_warn = compare_to_baseline(_flat_run(0.45), store=store, baseline_key="k")
    v_crit = compare_to_baseline(_flat_run(0.40), store=store, baseline_key="k")
    assert [v.status for v in (v_pass, v_warn, v_crit)] == [
        "PASS", "WARN", "CRITICAL_REGRESSION",
    ]
    assert v_pass.regression_detected is False
    assert v_warn.regression_detected is True
    assert v_crit.regression_detected is True
    assert v_warn.pass_rate_delta == -0.05
    assert v_warn.warn_threshold == 0.03 and v_warn.threshold == 0.08
    store.close()


def test_legacy_threshold_argument_stays_binary():
    """Passing threshold=... keeps the pre-Phase-3 behaviour: one line, no WARN."""
    store = _tmp_store(tmp="mrds_test_p3legacy.sqlite3")
    _pin_accuracy(store, "k", 0.50)
    # n=1000 so a 0.1-point drop survives as an integer correct-count (499/1000)
    tiny = compare_to_baseline(_flat_run(0.499, n=1000), store=store,
                               baseline_key="k", threshold=0.0)
    assert tiny.status == "CRITICAL_REGRESSION"  # even a 0.1-point drop trips it
    assert tiny.warn_threshold == 0.0  # WARN unreachable in binary mode
    tolerant = compare_to_baseline(_flat_run(0.40), store=store, baseline_key="k",
                                   threshold=0.20)
    assert tolerant.status == "PASS"
    store.close()


def test_pass_rate_delta_and_per_category_deltas():
    store = _tmp_store(tmp="mrds_test_p3cat.sqlite3")
    base = evaluate(client=MockClient(), samples=SAMPLES, prompt_version="v1")
    _pin_and_compare(store, base, "p3cat")
    current = evaluate(client=AlwaysGeneral(), samples=SAMPLES, prompt_version="v1")
    verdict = compare_to_baseline(current, store=store, baseline_key="p3cat")
    assert verdict.pass_rate_delta == -0.75
    assert verdict.per_category_delta == {
        "account": -1.0, "billing": -1.0, "technical": -1.0, "general": 0.0,
    }
    assert verdict.status == "CRITICAL_REGRESSION"
    store.close()


def test_regressions_and_improvements_are_named_case_by_case():
    # Baseline: broken model (only 'general' right). Current: correct model.
    store = _tmp_store(tmp="mrds_test_p3diff.sqlite3")
    _pin_and_compare(
        store, evaluate(client=AlwaysGeneral(), samples=SAMPLES, prompt_version="v1"),
        "p3diff",
    )
    better = compare_to_baseline(
        evaluate(client=MockClient(), samples=SAMPLES, prompt_version="v1"),
        store=store, baseline_key="p3diff",
    )
    assert sorted(better.improvements) == ["a", "b", "c"]
    assert better.regressions == []
    assert better.status == "PASS"  # improvements are not a regression
    store.close()

    # Baseline: correct model. Current: one case flipped to the wrong category.
    store2 = _tmp_store(tmp="mrds_test_p3diff2.sqlite3")
    _pin_and_compare(
        store2, evaluate(client=MockClient(), samples=SAMPLES, prompt_version="v1"),
        "p3diff2",
    )

    def flip_c(email, prompt=None, client=None):
        out = classify_email(email, prompt=prompt, client=client)
        if email.email_id == "c":
            out["category"] = "billing"  # c expects 'account'
        return out

    worse = compare_to_baseline(
        evaluate(client=MockClient(), samples=SAMPLES, prompt_version="v1",
                 classify=flip_c),
        store=store2, baseline_key="p3diff2",
    )
    assert worse.status == "CRITICAL_REGRESSION"  # 4/4 -> 3/4 = 0.25 drop
    assert worse.regressions == ["c"]
    assert worse.improvements == []
    store2.close()


def test_case_metrics_roundtrip_through_both_backends():
    config = load_prompt_config("v1")
    cases, duration = asyncio.run(
        run_batch(SAMPLES, config, CountingAsyncClient(), judge=True)
    )
    run = assemble_run(cases, provider="mock", prompt_version="v1",
                       dataset_version="0.4", duration_ms=duration)
    row = {
        "run_id": run.run_id, "created_at": run.created_at,
        "provider": run.provider, "prompt_version": run.prompt_version,
        "dataset_version": run.dataset_version, "n_samples": run.n_samples,
        "accuracy": run.accuracy, "metrics": run.metrics.model_dump(),
    }
    preds = [p.model_dump() for p in run.predictions]
    store = _tmp_store(tmp="mrds_test_p3case.sqlite3")
    store.save_run(row, preds)
    loaded = store.get_predictions(run.run_id)
    assert len(loaded) == 4
    by_id = {p["sample_id"]: p for p in loaded}
    cm = by_id["a"]["case_metrics"]
    assert cm["total_tokens"] == 10
    assert cm["latency_ms"] > 0
    assert cm["judge_source"] in ("heuristic", "llm")
    store.close()


def test_get_predictions_for_unknown_run_is_empty():
    store = _tmp_store(tmp="mrds_test_p3empty.sqlite3")
    assert store.get_predictions("no-such-run") == []
    store.close()


def test_golden_rows_carry_the_ideal_summary_for_the_judge():
    """Also pins the golden-024 fix: its ideal summary must describe ITS email
    (refund for order 67001), not golden-023's stock enquiry."""
    rows, _ = load_golden_dataset()
    g24 = next(r for r in rows if r["id"] == "golden-024")
    assert g24["expected_summary"].startswith(
        "The customer is requesting a refund for order 67001"
    )


# --- plain-python runner (no pytest needed) -------------------------------


def _run_all_tests() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed} passed, {failed} failed, {len(tests)} total")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all_tests() else 0)