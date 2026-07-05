"""
Break Resolution RAG + Schema Rule Library

Stores historical breaks and their human-entered resolutions as embeddings.
At query time finds the top-k most similar historical breaks and surfaces
their resolutions — so the same pattern is never debugged twice.

Also maintains a schema-fingerprint keyed rule library: when the same source/
target column set is seen again the confirmed rules are retrieved instantly
without calling the LLM.

Embedding backend: OpenAI text-embedding-3-small (1536-d vectors).
Fallback when no API key: keyword overlap similarity on break field names.
Vector store: SQLite + JSON blobs + numpy cosine similarity in-process.
"""

import json
import math
import hashlib
import sqlite3
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "rag.db"


# ── DB init ────────────────────────────────────────────────────────────────────

def _conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_rag_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS break_library (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                description   TEXT    NOT NULL,
                source_fields TEXT    NOT NULL,
                target_fields TEXT    NOT NULL,
                break_fields  TEXT    NOT NULL,
                resolution    TEXT,
                resolution_type TEXT,
                embedding     TEXT,
                run_id        TEXT,
                created_at    TEXT    NOT NULL,
                resolved_at   TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS schema_rule_library (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                schema_fingerprint  TEXT    NOT NULL UNIQUE,
                source_schema       TEXT    NOT NULL,
                target_schema       TEXT    NOT NULL,
                confirmed_rules     TEXT    NOT NULL,
                confirmed_mappings  TEXT    NOT NULL,
                use_count           INTEGER DEFAULT 1,
                created_at          TEXT    NOT NULL,
                updated_at          TEXT    NOT NULL
            )
        """)
        c.commit()


# ── Embedding helpers ──────────────────────────────────────────────────────────

def _embed(text: str) -> Optional[list[float]]:
    try:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            return None
        client = OpenAI(api_key=key)
        r = client.embeddings.create(model="text-embedding-3-small", input=text[:8000])
        return r.data[0].embedding
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / mag if mag else 0.0


def _break_text(source: dict, target: dict, fields: list[str]) -> str:
    parts = []
    for f in fields:
        parts.append(f"field '{f}': source={source.get(f, '?')} target={target.get(f, '?')}")
    return "Reconciliation break — " + "; ".join(parts) if parts else "unknown break"


# ── Break library ──────────────────────────────────────────────────────────────

def store_break(
    source_fields: dict,
    target_fields: dict,
    break_fields: list[str],
    resolution: Optional[str] = None,
    resolution_type: Optional[str] = None,
    run_id: Optional[str] = None,
) -> int:
    desc = _break_text(source_fields, target_fields, break_fields)
    emb = _embed(desc)
    now = datetime.now(timezone.utc).isoformat()

    with _conn() as c:
        cur = c.execute(
            """INSERT INTO break_library
               (description, source_fields, target_fields, break_fields,
                resolution, resolution_type, embedding, run_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                desc,
                json.dumps(source_fields),
                json.dumps(target_fields),
                json.dumps(break_fields),
                resolution,
                resolution_type,
                json.dumps(emb) if emb else None,
                run_id,
                now,
            ),
        )
        c.commit()
        return cur.lastrowid


def resolve_break(break_id: int, resolution: str, resolution_type: str):
    with _conn() as c:
        c.execute(
            """UPDATE break_library
               SET resolution=?, resolution_type=?, resolved_at=?
               WHERE id=?""",
            (resolution, resolution_type, datetime.now(timezone.utc).isoformat(), break_id),
        )
        c.commit()


def find_similar_breaks(
    source_fields: dict,
    target_fields: dict,
    break_fields: list[str],
    top_k: int = 3,
) -> list[dict]:
    """Return top-k similar historical breaks that have been resolved."""
    desc = _break_text(source_fields, target_fields, break_fields)
    q_emb = _embed(desc)

    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM break_library WHERE resolution IS NOT NULL ORDER BY id DESC LIMIT 200"
        ).fetchall()

    if not rows:
        return []

    scored = []
    for row in rows:
        r = dict(row)
        if q_emb and r.get("embedding"):
            try:
                sim = _cosine(q_emb, json.loads(r["embedding"]))
            except Exception:
                sim = 0.0
        else:
            # Keyword fallback: Jaccard on break field names
            try:
                sf = set(json.loads(r.get("break_fields", "[]")))
                qf = set(break_fields)
                sim = len(sf & qf) / max(len(sf | qf), 1)
            except Exception:
                sim = 0.0
        r["similarity"] = round(sim, 3)
        scored.append(r)

    scored.sort(key=lambda x: x["similarity"], reverse=True)

    result = []
    for r in scored[:top_k]:
        for key in ("source_fields", "target_fields", "break_fields"):
            try:
                r[key] = json.loads(r[key])
            except Exception:
                pass
        r.pop("embedding", None)
        result.append(r)

    return result


def get_all_breaks(limit: int = 100) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, description, break_fields, resolution, resolution_type, run_id, created_at, resolved_at "
            "FROM break_library ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["break_fields"] = json.loads(d["break_fields"])
        except Exception:
            pass
        result.append(d)
    return result


# ── Schema / Rule Library ──────────────────────────────────────────────────────

def _fingerprint(source_cols: list[str], target_cols: list[str]) -> str:
    key = "|".join(sorted(source_cols)) + "||" + "|".join(sorted(target_cols))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def save_rule_library(
    source_cols: list[str],
    target_cols: list[str],
    confirmed_mappings: list[dict],
    confirmed_rules: list[dict],
) -> str:
    fp = _fingerprint(source_cols, target_cols)
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        exists = c.execute(
            "SELECT id FROM schema_rule_library WHERE schema_fingerprint=?", (fp,)
        ).fetchone()
        if exists:
            c.execute(
                """UPDATE schema_rule_library
                   SET confirmed_rules=?, confirmed_mappings=?,
                       use_count=use_count+1, updated_at=?
                   WHERE schema_fingerprint=?""",
                (json.dumps(confirmed_rules), json.dumps(confirmed_mappings), now, fp),
            )
        else:
            c.execute(
                """INSERT INTO schema_rule_library
                   (schema_fingerprint, source_schema, target_schema,
                    confirmed_rules, confirmed_mappings, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    fp,
                    json.dumps(source_cols),
                    json.dumps(target_cols),
                    json.dumps(confirmed_rules),
                    json.dumps(confirmed_mappings),
                    now,
                    now,
                ),
            )
        c.commit()
    return fp


def get_rule_library(source_cols: list[str], target_cols: list[str]) -> Optional[dict]:
    fp = _fingerprint(source_cols, target_cols)
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM schema_rule_library WHERE schema_fingerprint=?", (fp,)
        ).fetchone()
    if not row:
        return None
    r = dict(row)
    r["confirmed_rules"] = json.loads(r["confirmed_rules"])
    r["confirmed_mappings"] = json.loads(r["confirmed_mappings"])
    return r


def get_all_rule_library() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT schema_fingerprint, source_schema, target_schema, use_count, updated_at "
            "FROM schema_rule_library ORDER BY use_count DESC LIMIT 50"
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["source_schema"] = json.loads(d["source_schema"])
            d["target_schema"] = json.loads(d["target_schema"])
        except Exception:
            pass
        result.append(d)
    return result
