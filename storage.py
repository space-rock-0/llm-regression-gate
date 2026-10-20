"""
storage.py — persistence for runs, predictions and baselines.

One backend: SQLiteStore — stdlib `sqlite3`. Real queries, real durability.

Why stdlib sqlite3 and no ORM: an ORM (SQLAlchemy) adds a dependency and a
layer of mapping code that a beginner then has to learn before understanding
the data. For 3 tables, plain SQL is clearer about what actually happens.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DB_FILENAME = "mrds.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    provider    TEXT NOT NULL,
    prompt_ver  TEXT NOT NULL,
    dataset_version TEXT NOT NULL DEFAULT 'unknown',
    n_samples   INTEGER NOT NULL,
    accuracy    REAL NOT NULL,
    metrics_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id     TEXT NOT NULL,
    sample_id  TEXT NOT NULL,
    category   TEXT NOT NULL,
    expected   TEXT NOT NULL,
    correct    INTEGER NOT NULL,
    summary    TEXT NOT NULL,
    PRIMARY KEY (run_id, sample_id)
);

CREATE TABLE IF NOT EXISTS baseline (
    key        TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    accuracy   REAL NOT NULL,
    metrics_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_metrics (
    run_id       TEXT NOT NULL,
    sample_id    TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    PRIMARY KEY (run_id, sample_id)
);
"""


class Store(Protocol):
    def save_run(self, run: dict, predictions: list[dict]) -> None: ...
    def get_run(self, run_id: str) -> dict | None: ...
    def list_runs(self) -> list[dict]: ...
    def get_predictions(self, run_id: str) -> list[dict]: ...
    def set_baseline(self, key: str, run: dict) -> None: ...
    def get_baseline(self, key: str) -> dict | None: ...


class SQLiteStore:
    def __init__(self, path=None) -> None:
        self.path = Path(path) if path else DEFAULT_DATA_DIR / DB_FILENAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        cols = {r["name"] for r in self._conn.execute(
            "PRAGMA table_info(runs)").fetchall()}
        if "dataset_version" not in cols:
            try:
                self._conn.execute(
                    "ALTER TABLE runs ADD COLUMN dataset_version TEXT NOT NULL DEFAULT 'unknown'"
                )
                self._conn.commit()
            except sqlite3.Error:
                pass
        self._conn.commit()

    def save_run(self, run, predictions):
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, created_at, provider, prompt_ver, dataset_version, n_samples, accuracy, metrics_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run["run_id"], run["created_at"], run["provider"],
                    run["prompt_version"], run.get("dataset_version", "unknown"),
                    run["n_samples"], run["accuracy"],
                    json.dumps(run["metrics"], sort_keys=True),
                ),
            )
            self._conn.executemany(
                "INSERT OR REPLACE INTO predictions "
                "(run_id, sample_id, category, expected, correct, summary)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(run["run_id"], p["sample_id"], p["category"], p["expected"],
                  int(p["correct"]), p["summary"]) for p in predictions],
            )
            metric_rows = [
                (run["run_id"], p["sample_id"], json.dumps(p["case_metrics"], sort_keys=True))
                for p in predictions if p.get("case_metrics")
            ]
            if metric_rows:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO case_metrics (run_id, sample_id, metrics_json)"
                    " VALUES (?, ?, ?)", metric_rows,
                )

    def get_run(self, run_id):
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_run(row) if row else None

    def list_runs(self):
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY created_at ASC").fetchall()
        return [self._row_to_run(r) for r in rows]

    def get_predictions(self, run_id):
        rows = self._conn.execute(
            "SELECT * FROM predictions WHERE run_id = ? ORDER BY sample_id",
            (run_id,)).fetchall()
        metrics = {
            row["sample_id"]: json.loads(row["metrics_json"])
            for row in self._conn.execute(
                "SELECT sample_id, metrics_json FROM case_metrics WHERE run_id = ?",
                (run_id,)).fetchall()
        }
        out = []
        for row in rows:
            item = dict(row)
            item["case_metrics"] = metrics.get(row["sample_id"])
            out.append(item)
        return out

    def set_baseline(self, key, run):
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO baseline "
                "(key, run_id, created_at, accuracy, metrics_json)"
                " VALUES (?, ?, ?, ?, ?)",
                (key, run["run_id"], run["created_at"], run["accuracy"],
                 json.dumps(run["metrics"], sort_keys=True)),
            )

    def get_baseline(self, key):
        row = self._conn.execute(
            "SELECT * FROM baseline WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        return {
            "key": row["key"], "run_id": row["run_id"],
            "created_at": row["created_at"], "accuracy": row["accuracy"],
            "metrics": json.loads(row["metrics_json"]),
        }

    @staticmethod
    def _row_to_run(row):
        return {
            "run_id": row["run_id"], "created_at": row["created_at"],
            "provider": row["provider"], "prompt_version": row["prompt_ver"],
            "dataset_version": row["dataset_version"] if "dataset_version" in row.keys() else "unknown",
            "n_samples": row["n_samples"], "accuracy": row["accuracy"],
            "metrics": json.loads(row["metrics_json"]),
        }

    def close(self):
        self._conn.close()


def get_store(backend=None):
    return SQLiteStore()