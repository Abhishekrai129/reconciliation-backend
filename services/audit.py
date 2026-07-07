import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "audit.db"

_LLM_COLS = [
    ("run_id",       "TEXT"),
    ("step",         "TEXT"),
    ("prompt",       "TEXT"),
    ("raw_response", "TEXT"),
    ("tokens_in",    "INTEGER"),
    ("tokens_out",   "INTEGER"),
    ("latency_ms",   "INTEGER"),
]


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                llm_provider TEXT,
                reasoning TEXT
            )
        """)
        conn.commit()
        # Safe migration: add new columns if they don't exist yet
        existing = {row[1] for row in conn.execute("PRAGMA table_info(audit_log)").fetchall()}
        for col_name, col_type in _LLM_COLS:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE audit_log ADD COLUMN {col_name} {col_type}")
        conn.commit()


def log(action: str, details: dict, llm_provider: str = None, reasoning: str = None):
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_log (timestamp, action, details, llm_provider, reasoning) VALUES (?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                action,
                json.dumps(details),
                llm_provider,
                reasoning,
            ),
        )
        conn.commit()


def log_llm_call(
    *,
    run_id: str,
    step: str,
    provider: str,
    model: str,
    prompt: str,
    raw_response: str,
    tokens_in: int | None,
    tokens_out: int | None,
    latency_ms: int,
    error: str | None = None,
):
    """One row per LLM call — full prompt, full response, token counts, latency."""
    details = {"model": model}
    if error:
        details["error"] = error
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO audit_log
               (timestamp, action, details, llm_provider, reasoning,
                run_id, step, prompt, raw_response, tokens_in, tokens_out, latency_ms)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                "llm_call",
                json.dumps(details),
                provider,
                None,
                run_id or "",
                step or "",
                prompt,
                raw_response,
                tokens_in,
                tokens_out,
                latency_ms,
            ),
        )
        conn.commit()


def get_all() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT 500"
        ).fetchall()
        return [
            {
                "id":           r["id"],
                "timestamp":    r["timestamp"],
                "action":       r["action"],
                "details":      json.loads(r["details"]),
                "llm_provider": r["llm_provider"],
                "reasoning":    r["reasoning"],
                "run_id":       r["run_id"],
                "step":         r["step"],
                "tokens_in":    r["tokens_in"],
                "tokens_out":   r["tokens_out"],
                "latency_ms":   r["latency_ms"],
                "has_prompt":   bool(r["prompt"]),
            }
            for r in rows
        ]


def get_llm_calls(run_id: str | None = None) -> list[dict]:
    """Return only llm_call rows, optionally filtered by run_id."""
    with _get_conn() as conn:
        if run_id:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE action='llm_call' AND run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE action='llm_call' ORDER BY id DESC LIMIT 100"
            ).fetchall()
        return [
            {
                "id":           r["id"],
                "timestamp":    r["timestamp"],
                "run_id":       r["run_id"],
                "step":         r["step"],
                "llm_provider": r["llm_provider"],
                "details":      json.loads(r["details"]),
                "prompt":       r["prompt"],
                "raw_response": r["raw_response"],
                "tokens_in":    r["tokens_in"],
                "tokens_out":   r["tokens_out"],
                "latency_ms":   r["latency_ms"],
            }
            for r in rows
        ]
