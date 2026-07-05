import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "pipeline.db"

STEPS = [
    {"index": 0, "name": "upload",     "label": "File Upload"},
    {"index": 1, "name": "understand", "label": "AI: Understand Fields"},
    {"index": 2, "name": "map",        "label": "AI: Map Schemas"},
    {"index": 3, "name": "rules",      "label": "AI: Suggest Rules"},
    {"index": 4, "name": "review",     "label": "Human Review"},
    {"index": 5, "name": "reconcile",  "label": "Run Reconciliation"},
    {"index": 6, "name": "results",    "label": "Results"},
]


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_pipeline_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'running',
                source_file TEXT,
                target_file TEXT,
                source_rows INTEGER,
                target_rows INTEGER,
                match_rate REAL,
                matched INTEGER,
                breaks INTEGER,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_steps (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                step_name TEXT NOT NULL,
                step_label TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                started_at TEXT,
                completed_at TEXT,
                duration_ms INTEGER,
                input_summary TEXT,
                output_summary TEXT,
                ai_reasoning TEXT,
                human_action TEXT,
                human_action_at TEXT,
                FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
            )
        """)
        conn.commit()


def create_run(
    source_file: str,
    target_file: str,
    source_rows: int,
    target_rows: int,
) -> str:
    run_id = str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO pipeline_runs
               (id, status, source_file, target_file, source_rows, target_rows, created_at)
               VALUES (?, 'running', ?, ?, ?, ?, ?)""",
            (run_id, source_file, target_file, source_rows, target_rows, _now()),
        )
        conn.commit()
    return run_id


def initialize_steps(run_id: str):
    with _get_conn() as conn:
        for step in STEPS:
            conn.execute(
                """INSERT INTO pipeline_steps
                   (id, run_id, step_index, step_name, step_label, status)
                   VALUES (?, ?, ?, ?, ?, 'pending')""",
                (str(uuid.uuid4()), run_id, step["index"], step["name"], step["label"]),
            )
        conn.commit()


def start_step(run_id: str, step_name: str, input_summary: str = ""):
    with _get_conn() as conn:
        conn.execute(
            """UPDATE pipeline_steps
               SET status = 'running', started_at = ?, input_summary = ?
               WHERE run_id = ? AND step_name = ?""",
            (_now(), input_summary, run_id, step_name),
        )
        conn.commit()


def complete_step(run_id: str, step_name: str, output_summary: str = "", ai_reasoning: str = ""):
    now = _now()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT started_at FROM pipeline_steps WHERE run_id = ? AND step_name = ?",
            (run_id, step_name),
        ).fetchone()
        duration_ms = None
        if row and row["started_at"]:
            try:
                started = datetime.fromisoformat(row["started_at"])
                completed = datetime.fromisoformat(now)
                duration_ms = int((completed - started).total_seconds() * 1000)
            except Exception:
                pass
        conn.execute(
            """UPDATE pipeline_steps
               SET status = 'done', completed_at = ?, duration_ms = ?,
                   output_summary = ?, ai_reasoning = ?
               WHERE run_id = ? AND step_name = ?""",
            (now, duration_ms, output_summary, ai_reasoning, run_id, step_name),
        )
        conn.commit()


def set_step_awaiting_human(run_id: str, step_name: str, output_summary: str = ""):
    with _get_conn() as conn:
        conn.execute(
            """UPDATE pipeline_steps
               SET status = 'awaiting_human', output_summary = ?
               WHERE run_id = ? AND step_name = ?""",
            (output_summary, run_id, step_name),
        )
        conn.commit()


def record_human_action(run_id: str, step_name: str, action_summary: str):
    now = _now()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT started_at FROM pipeline_steps WHERE run_id = ? AND step_name = ?",
            (run_id, step_name),
        ).fetchone()
        duration_ms = None
        if row and row["started_at"]:
            try:
                started = datetime.fromisoformat(row["started_at"])
                completed = datetime.fromisoformat(now)
                duration_ms = int((completed - started).total_seconds() * 1000)
            except Exception:
                pass
        conn.execute(
            """UPDATE pipeline_steps
               SET status = 'done', human_action = ?, human_action_at = ?,
                   completed_at = ?, duration_ms = ?
               WHERE run_id = ? AND step_name = ?""",
            (action_summary, now, now, duration_ms, run_id, step_name),
        )
        conn.commit()


def fail_step(run_id: str, step_name: str, error: str):
    now = _now()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT started_at FROM pipeline_steps WHERE run_id = ? AND step_name = ?",
            (run_id, step_name),
        ).fetchone()
        duration_ms = None
        if row and row["started_at"]:
            try:
                started = datetime.fromisoformat(row["started_at"])
                completed = datetime.fromisoformat(now)
                duration_ms = int((completed - started).total_seconds() * 1000)
            except Exception:
                pass
        conn.execute(
            """UPDATE pipeline_steps
               SET status = 'error', completed_at = ?, duration_ms = ?, output_summary = ?
               WHERE run_id = ? AND step_name = ?""",
            (now, duration_ms, error, run_id, step_name),
        )
        conn.execute(
            "UPDATE pipeline_runs SET status = 'failed' WHERE id = ?",
            (run_id,),
        )
        conn.commit()


def complete_run(run_id: str, match_rate: float, matched: int, breaks: int):
    with _get_conn() as conn:
        conn.execute(
            """UPDATE pipeline_runs
               SET status = 'completed', match_rate = ?, matched = ?, breaks = ?, completed_at = ?
               WHERE id = ?""",
            (match_rate, matched, breaks, _now(), run_id),
        )
        conn.commit()


def get_run_trace(run_id: str) -> dict:
    with _get_conn() as conn:
        run_row = conn.execute(
            "SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if not run_row:
            return {}
        step_rows = conn.execute(
            "SELECT * FROM pipeline_steps WHERE run_id = ? ORDER BY step_index",
            (run_id,),
        ).fetchall()
        return {
            "run": dict(run_row),
            "steps": [dict(r) for r in step_rows],
        }


def get_all_runs() -> list:
    with _get_conn() as conn:
        run_rows = conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        result = []
        for run_row in run_rows:
            run_dict = dict(run_row)
            step_rows = conn.execute(
                "SELECT step_name, step_label, status, duration_ms FROM pipeline_steps WHERE run_id = ? ORDER BY step_index",
                (run_row["id"],),
            ).fetchall()
            run_dict["steps"] = [dict(s) for s in step_rows]
            result.append(run_dict)
        return result
