import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "audit.db"


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


def get_all() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT 200"
        ).fetchall()
        return [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "action": r["action"],
                "details": json.loads(r["details"]),
                "llm_provider": r["llm_provider"],
                "reasoning": r["reasoning"],
            }
            for r in rows
        ]
