# MRDS — Decision Log (condensed)

## Core
- D1. Mock LLM is default provider; real providers opt-in via env/flag.
- D2. Single interface `complete(system, user) -> str` — providers swap without touching evaluator.
- D3. Prompt is a parameter (prompts/*.yaml), never hardcoded.
- D4. Pydantic for config/output; plain dict at classifier edge.
- D5/D25. Per-version closed label set enforced in contract only (D26).
- D6. Robust JSON parsing with loud failure.
- D7. SQLite stdlib + JSON fallback, no ORM.
- D8. Ground truth in samples/emails.json, not Python.
- D9. Deterministic verdict: PASS/WARN/CRITICAL_REGRESSION by delta vs baseline.
- D13. Degrade recoverable fields; raise on structural failures.
- D14. YAML prompts (PyYAML).
- D17. Few-shot in SYSTEM turn (stable user-turn shape).
- D19. v2 uses a reduced label set (folds categories) — design intent was "worse"; real Ollama run showed mixed: +0.1317 net, 15 regressions, 37 improvements, general collapsed (-0.20).
- D20. Mock honours prompt's label set.
- D21/D31.1. Baseline key = `<provider>:<prompt_version>:<dataset_version>`.
- D22. PromptConfig (file) vs PromptConfig (contract) renamed for clarity.
- D23/D24. Typed EmailInput in, EmailClassification (Literal) out; construction = validation.
- D27. Contract is the swap seam.
- D29. Prompt assembly lives in contract.

## Golden dataset
- D31.2. load_golden_dataset enforces provenance at load.
- D31.3. One file of record: golden_dataset/golden_dataset_v0.4_reviewed.json.
- D32.2/D33.1. golden-024 summary corrected (human approval).
- D32.3. Exclusions are data (excluded_from_v0.4.json), not deletions.

## Scoring & baselines
- D33.2. Async runner: asyncio + httpx, semaphore-bounded.
- D33.3. Scoring in one function (assemble_run).
- D33.4. LLM-as-Judge degrades, never fails; verdict driven by exact match.
- D33.5. PASS/WARN/CRITICAL_REGRESSION with named flips; configurable thresholds.
- D33.6. Per-case metrics in new table; tokens split classify vs judge.
- D33.7. Sync Ollama ignored temperature (SDK default 1.0); async sends it.
- D33.8. Baseline pinned ollama:v1:0.4 = 0.4910 (after 404 on default llama3.2).

## Alerting & reporting (Phase 4)
- Standalone stdlib HTML report (ui-ux-pro-max palette).
- Slack Block Kit; MockNotifier fallback when no webhook.
- Drift = secondary 7-run moving average; SLOW_DRIFT_WARNING.
- Validated on real Ollama data: v2 vs v1 = +0.1317, 15 regressions, 37 improvements.

## CI/CD & packaging (Phase 5)
- GitHub Actions on prompts/**; mock eval <30s; blocks on CRITICAL_REGRESSION.
- Dockerfile + docker-compose; thresholds from env.

## Cleanup
- JSONStore removed; alias shims/methods removed; requests dep removed.
- Golden_Data_Set renamed golden_dataset; source data archived.
- Tests: 79/79 green.
"""