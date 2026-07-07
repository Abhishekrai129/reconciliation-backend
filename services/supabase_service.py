"""
Supabase persistence layer.
5 tables: pipeline_runs, break_records, field_mappings, recon_rules, audit_events
Sync functions used from FastAPI sync endpoints; async functions for async endpoints.
"""
import os
import httpx

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://aivgdszslsqsybugtrud.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

def _headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

def _url(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"

# ── Sync write (called from FastAPI sync endpoints) ───────────────────────────

def save_run_sync(source_file: str, target_file: str, use_case: str,
                  match_rate: float, matched: int, breaks: int,
                  total_source: int, total_target: int) -> str | None:
    payload = {
        "source_file": source_file, "target_file": target_file,
        "use_case": use_case, "match_rate": round(match_rate, 2),
        "matched_count": matched, "break_count": breaks,
        "total_source": total_source, "total_target": total_target,
        "status": "completed",
    }
    try:
        with httpx.Client(timeout=6) as c:
            r = c.post(_url("pipeline_runs"), headers=_headers(), json=payload)
            if r.status_code in (200, 201):
                data = r.json()
                return data[0]["id"] if isinstance(data, list) else data.get("id")
    except Exception:
        pass
    return None

def save_breaks_sync(run_id: str, break_list: list) -> None:
    if not run_id or not break_list:
        return
    rows = [{
        "run_id": run_id,
        "match_key": str(b.get("match_key", "")),
        "break_fields": b.get("break_reasons", []),
        "severity": b.get("severity", "warning"),
        "root_cause": b.get("root_cause", ""),
        "status": "open",
    } for b in break_list[:50]]
    try:
        with httpx.Client(timeout=6) as c:
            c.post(_url("break_records"), headers=_headers(), json=rows)
    except Exception:
        pass

def save_mappings_sync(mappings: list, use_case: str) -> None:
    if not mappings:
        return
    rows = [{
        "source_field": m.get("source_column", ""),
        "target_field": m.get("target_column", ""),
        "confidence": float(m.get("confidence", 0.85)),
        "match_type": m.get("match_type", "semantic"),
        "use_case": use_case,
        "confirmed_count": 0,
        "auto_apply": False,
    } for m in mappings if m.get("source_column") and m.get("target_column")]
    if not rows:
        return
    h = {**_headers(), "Prefer": "resolution=ignore-duplicates,return=minimal"}
    try:
        with httpx.Client(timeout=6) as c:
            c.post(_url("field_mappings"), headers=h, json=rows)
    except Exception:
        pass

def log_audit_sync(run_id: str | None, event_type: str, agent: str,
                   decision: str, field: str = "", reasoning: str = "") -> None:
    payload = {
        "run_id": run_id, "event_type": event_type, "agent": agent,
        "decision": decision, "field": field, "reasoning": reasoning,
    }
    try:
        with httpx.Client(timeout=6) as c:
            c.post(_url("audit_events"), headers=_headers(), json=payload)
    except Exception:
        pass

# ── Async write (for async endpoints) ────────────────────────────────────────

async def save_run(source_file: str, target_file: str, use_case: str,
                   match_rate: float, matched: int, breaks: int,
                   total_source: int, total_target: int) -> str | None:
    payload = {
        "source_file": source_file, "target_file": target_file,
        "use_case": use_case, "match_rate": round(match_rate, 2),
        "matched_count": matched, "break_count": breaks,
        "total_source": total_source, "total_target": total_target,
        "status": "completed",
    }
    async with httpx.AsyncClient(timeout=8) as c:
        r = await c.post(_url("pipeline_runs"), headers=_headers(), json=payload)
        if r.status_code in (200, 201):
            data = r.json()
            return data[0]["id"] if isinstance(data, list) else data.get("id")
    return None

async def save_breaks(run_id: str, break_list: list[dict]) -> None:
    if not run_id or not break_list:
        return
    rows = [{
        "run_id": run_id,
        "match_key": str(b.get("match_key", "")),
        "break_fields": b.get("break_reasons", []),
        "severity": b.get("severity", "warning"),
        "root_cause": b.get("root_cause", ""),
        "status": "open",
    } for b in break_list[:50]]   # cap at 50 per run
    async with httpx.AsyncClient(timeout=8) as c:
        await c.post(_url("break_records"), headers=_headers(), json=rows)

async def save_mappings(mappings: list[dict], use_case: str) -> None:
    if not mappings:
        return
    rows = [{
        "source_field": m.get("source_column", ""),
        "target_field": m.get("target_column", ""),
        "confidence": float(m.get("confidence", 0.85)),
        "match_type": m.get("match_type", "semantic"),
        "use_case": use_case,
        "confirmed_count": 0,
        "auto_apply": False,
    } for m in mappings if m.get("source_column") and m.get("target_column")]
    h = {**_headers(), "Prefer": "resolution=ignore-duplicates,return=minimal"}
    async with httpx.AsyncClient(timeout=8) as c:
        await c.post(_url("field_mappings"), headers=h, json=rows)

async def log_audit(run_id: str | None, event_type: str, agent: str,
                    decision: str, field: str = "", reasoning: str = "") -> None:
    payload = {
        "run_id": run_id, "event_type": event_type, "agent": agent,
        "decision": decision, "field": field, "reasoning": reasoning,
    }
    async with httpx.AsyncClient(timeout=8) as c:
        await c.post(_url("audit_events"), headers=_headers(), json=payload)

# ── Read (for Ask AI context) ─────────────────────────────────────────────────

async def get_runs(limit: int = 20) -> list[dict]:
    async with httpx.AsyncClient(timeout=8) as c:
        r = await c.get(
            f"{_url('pipeline_runs')}?order=created_at.desc&limit={limit}",
            headers=_headers()
        )
        return r.json() if r.status_code == 200 else []

async def get_open_breaks(limit: int = 100) -> list[dict]:
    async with httpx.AsyncClient(timeout=8) as c:
        r = await c.get(
            f"{_url('break_records')}?status=eq.open&order=created_at.desc&limit={limit}",
            headers=_headers()
        )
        return r.json() if r.status_code == 200 else []

async def get_mappings(use_case: str = "", limit: int = 50) -> list[dict]:
    q = f"order=confirmed_count.desc&limit={limit}"
    if use_case:
        q = f"use_case=eq.{use_case}&{q}"
    async with httpx.AsyncClient(timeout=8) as c:
        r = await c.get(f"{_url('field_mappings')}?{q}", headers=_headers())
        return r.json() if r.status_code == 200 else []

async def get_audit_events(run_id: str | None = None, limit: int = 50) -> list[dict]:
    q = f"order=created_at.desc&limit={limit}"
    if run_id:
        q = f"run_id=eq.{run_id}&{q}"
    async with httpx.AsyncClient(timeout=8) as c:
        r = await c.get(f"{_url('audit_events')}?{q}", headers=_headers())
        return r.json() if r.status_code == 200 else []
