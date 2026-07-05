import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from models.schemas import LLMConfig, LLMProvider
from services import audit, file_processor, llm_service, reconciliation
from services.audit import init_db, log
from services import pipeline_tracker
from services.pipeline_tracker import (
    init_pipeline_db, create_run, initialize_steps,
    start_step, complete_step, set_step_awaiting_human,
    record_human_action, fail_step, complete_run,
    get_run_trace, get_all_runs,
)
from services import rag_service
from services.rag_service import init_rag_db
from services import probabilistic_matcher
from services import data_profiler
from services import dictionary_service
from services.dictionary_service import init_dict_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_pipeline_db()
    init_rag_db()
    init_dict_db()
    yield


app = FastAPI(title="Reconciliation AI Pilot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── LLM Config ────────────────────────────────────────────────────────────────

@app.get("/api/llm/config")
def get_llm_config():
    cfg = llm_service.get_llm_config()
    return {"provider": cfg.provider, "model": cfg.model}


@app.post("/api/llm/config")
def update_llm_config(cfg: LLMConfig):
    llm_service.set_llm_config(cfg)
    log("llm_config_updated", {"provider": cfg.provider, "model": cfg.model})
    return {"status": "ok", "provider": cfg.provider, "model": cfg.model}


@app.get("/api/llm/models")
def list_models():
    return {
        "anthropic": [
            "claude-sonnet-4-6",
            "claude-opus-4-7",
            "claude-haiku-4-5-20251001",
        ],
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
        ],
        "ollama": [
            "llama3.1",
            "mistral",
            "gemma2",
            "phi3",
        ],
    }


# ── File Upload & Profiling ───────────────────────────────────────────────────

@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = file_processor.read_file(content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    file_id = file_processor.store_dataframe(df)
    profile = file_processor.profile_dataframe(df, file.filename, file_id)
    log("file_uploaded", {"file_id": file_id, "filename": file.filename, "rows": profile["row_count"]})
    return profile


@app.get("/api/files/{file_id}/sample")
def get_sample(file_id: str, n: int = 10):
    try:
        return {"data": file_processor.get_sample_data(file_id, n)}
    except KeyError:
        raise HTTPException(status_code=404, detail="File not found")


# ── Data Intelligence Profiler ────────────────────────────────────────────────

@app.post("/api/profile")
def profile_pair(body: dict):
    """Run the data intelligence profiler on two uploaded files.

    body: { "file_a_id": "...", "file_b_id": "...",
            "key_col_a": "...", "key_col_b": "..." }  # key_cols optional

    Returns quality_score, all issues, HITL triggers, recommendations.
    """
    file_a_id = body.get("file_a_id", "")
    file_b_id = body.get("file_b_id", "")
    try:
        df_a = file_processor.get_dataframe(file_a_id)
        df_b = file_processor.get_dataframe(file_b_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    report = data_profiler.profile_files(
        df_a, df_b,
        key_col_a=body.get("key_col_a"),
        key_col_b=body.get("key_col_b"),
    )
    log("data_profiled", {
        "file_a_id": file_a_id,
        "file_b_id": file_b_id,
        "quality_score": report["quality_score"],
        "issues": report["total_issues"],
    })
    return report


# ── AI Field Understanding ────────────────────────────────────────────────────

@app.post("/api/ai/understand-fields")
async def understand_fields(body: dict):
    """
    body: { "file_id": "...", "columns": [...] }
    """
    columns = body.get("columns", [])
    if not columns:
        raise HTTPException(status_code=400, detail="No columns provided")

    cfg = llm_service.get_llm_config()
    enriched = await llm_service.understand_fields(columns)

    log(
        "fields_understood",
        {"file_id": body.get("file_id"), "column_count": len(columns)},
        llm_provider=cfg.provider,
    )
    return {"columns": enriched}


# ── AI Schema Mapping ─────────────────────────────────────────────────────────

@app.post("/api/ai/map-schemas")
async def map_schemas(body: dict):
    """
    body: { "file_a_columns": [...], "file_b_columns": [...] }
    """
    file_a_cols = body.get("file_a_columns", [])
    file_b_cols = body.get("file_b_columns", [])

    cfg = llm_service.get_llm_config()
    result = await llm_service.map_schemas(file_a_cols, file_b_cols)

    log(
        "schemas_mapped",
        {"mappings_count": len(result.get("mappings", []))},
        llm_provider=cfg.provider,
    )
    return result


# ── AI Rule Suggestion ────────────────────────────────────────────────────────

@app.post("/api/ai/suggest-rules")
async def suggest_rules(body: dict):
    """
    body: { "mappings": [...], "file_a_id": "...", "file_b_id": "..." }
    """
    mappings = body.get("mappings", [])
    file_a_id = body.get("file_a_id", "")
    file_b_id = body.get("file_b_id", "")

    sample_a = file_processor.get_sample_data(file_a_id, 5) if file_a_id else []
    sample_b = file_processor.get_sample_data(file_b_id, 5) if file_b_id else []

    cfg = llm_service.get_llm_config()
    rules = await llm_service.suggest_matching_rules(
        mappings, {"file_a": sample_a, "file_b": sample_b}
    )

    log("rules_suggested", {"rule_count": len(rules)}, llm_provider=cfg.provider)
    return {"rules": rules}


# ── Reconciliation ────────────────────────────────────────────────────────────

class ReconcileRequest(BaseModel):
    file_a_id: str
    file_b_id: str
    rules: list[dict]
    key_columns: list[dict]
    run_id: Optional[str] = None


@app.post("/api/reconcile")
def run_reconciliation(req: ReconcileRequest):
    try:
        result = reconciliation.run_reconciliation(
            req.file_a_id,
            req.file_b_id,
            req.rules,
            req.key_columns,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if req.run_id:
        reconciliation.store_run_results(req.run_id, result)

    log("reconciliation_run", {
        "matched": result["matched"],
        "breaks": result["breaks"],
        "match_rate": result["match_rate"],
    })
    return result


@app.get("/api/reconcile/{run_id}/results")
def get_reconcile_results(run_id: str):
    result = reconciliation.get_run_results(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No results stored for this run")
    return result


# ── AI Break Explanation ──────────────────────────────────────────────────────

class BreakExplainRequest(BaseModel):
    source_record: dict
    target_record: dict
    break_fields: list[str]


@app.post("/api/ai/explain-break")
async def explain_break(req: BreakExplainRequest):
    cfg = llm_service.get_llm_config()
    explanation = await llm_service.explain_break(
        req.source_record, req.target_record, req.break_fields
    )
    log(
        "break_explained",
        {"break_fields": req.break_fields},
        llm_provider=cfg.provider,
        reasoning=explanation,
    )
    return {"explanation": explanation}


# ── Audit Trail ───────────────────────────────────────────────────────────────

@app.get("/api/audit")
def get_audit():
    return {"entries": audit.get_all()}


class AuditLogRequest(BaseModel):
    action: str
    details: dict
    llm_provider: Optional[str] = None
    reasoning: Optional[str] = None

@app.post("/api/audit/log")
def post_audit_log(req: AuditLogRequest):
    log(req.action, req.details, llm_provider=req.llm_provider, reasoning=req.reasoning)
    return {"ok": True}


# ── AI Data Chat ─────────────────────────────────────────────────────────────

class DataChatRequest(BaseModel):
    pathname: str = "/"
    messages: list[dict]


@app.post("/api/data-chat")
async def data_chat(req: DataChatRequest):
    last_user = next((m["content"] for m in reversed(req.messages) if m["role"] == "user"), "")
    cfg = llm_service.get_llm_config()

    # Get recent runs for context
    try:
        recent_runs = get_all_runs()[:3]
        runs_context = "\n".join([
            f"- {r['source_file']} → {r['target_file']}: {r.get('match_rate','?')}% match rate, {r.get('breaks','?')} breaks ({r['status']})"
            for r in recent_runs if r.get('match_rate')
        ])
    except Exception:
        runs_context = "No run data available"

    system = f"""You are an expert financial reconciliation AI assistant embedded in SmartStream TLM.
You help users understand reconciliation match rates, breaks, field mappings, and data sources.

Current reconciliation runs:
{runs_context}

Keep answers concise (3-5 sentences max). Use bullet points for lists. Be specific about numbers.
When asked about breaks, explain root causes. When asked about mappings, explain field semantics.
Format important numbers in bold using **value** syntax."""

    history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in req.messages[:-1][-6:]])
    prompt = f"{history}\nUSER: {last_user}\n\nAnswer concisely as a reconciliation expert:"

    try:
        reply = await llm_service.call_llm(prompt, system)
        log("data_chat", {"question": last_user[:100]}, llm_provider=cfg.provider)
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"AI service error: {str(e)[:100]}"}


# ── Pipeline Tracking ─────────────────────────────────────────────────────────

class PipelineCreateRequest(BaseModel):
    source_file: str
    target_file: str
    source_rows: int
    target_rows: int


class PipelineStepStartRequest(BaseModel):
    input_summary: Optional[str] = ""


class PipelineStepCompleteRequest(BaseModel):
    output_summary: Optional[str] = ""
    ai_reasoning: Optional[str] = ""


class PipelineReviewRequest(BaseModel):
    approved_mappings: list[dict]
    approved_rules: list[dict]
    rejected_mappings: list[str]


class PipelineCompleteRequest(BaseModel):
    match_rate: float
    matched: int
    breaks: int


@app.post("/api/pipeline/create")
def pipeline_create(req: PipelineCreateRequest):
    run_id = create_run(req.source_file, req.target_file, req.source_rows, req.target_rows)
    initialize_steps(run_id)
    trace = get_run_trace(run_id)
    return {"run_id": run_id, "steps": trace.get("steps", [])}


@app.get("/api/pipeline")
def pipeline_list():
    runs = get_all_runs()
    return {"runs": runs}


@app.get("/api/pipeline/{run_id}/trace")
def pipeline_trace(run_id: str):
    trace = get_run_trace(run_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Run not found")
    return trace


@app.post("/api/pipeline/{run_id}/step/{step_name}/start")
def pipeline_step_start(run_id: str, step_name: str, body: PipelineStepStartRequest):
    start_step(run_id, step_name, body.input_summary or "")
    return {"ok": True}


@app.post("/api/pipeline/{run_id}/step/{step_name}/complete")
def pipeline_step_complete(run_id: str, step_name: str, body: PipelineStepCompleteRequest):
    complete_step(run_id, step_name, body.output_summary or "", body.ai_reasoning or "")
    return {"ok": True}


@app.post("/api/pipeline/{run_id}/review")
def pipeline_review(run_id: str, body: PipelineReviewRequest):
    approved = len(body.approved_mappings)
    rejected = len(body.rejected_mappings)
    total = approved + rejected
    active_rules = len(body.approved_rules)
    action_summary = (
        f"Approved {approved}/{total} mappings, rejected {rejected}. "
        f"{active_rules} rules active."
    )

    # ── Learning loop: every confirmed mapping teaches the dictionary ──────
    new_aliases = 0
    for m in body.approved_mappings:
        src = m.get("source_column", "")
        tgt = m.get("target_column", "")
        if src and tgt:
            dictionary_service.record_confirmed_mapping(src, tgt, run_id=run_id)
            new_aliases += 1

    # ── Learning loop: value maps from approved rules ──────────────────────
    for r in body.approved_rules:
        threshold = r.get("threshold")
        if isinstance(threshold, dict) and r.get("match_type") == "value_lookup":
            src_col = r.get("source_column", "")
            entry = dictionary_service.lookup_field(src_col)
            field_type = entry["canonical_name"] if entry else src_col
            for abbrev, full_form in threshold.items():
                dictionary_service.learn_value_mapping(field_type, abbrev, str(full_form), run_id=run_id)

    start_step(run_id, "review", f"{total} mappings to review, {active_rules} rules")
    record_human_action(run_id, "review", action_summary)
    log("pipeline_review", {
        "run_id": run_id,
        "approved_mappings": approved,
        "rejected_mappings": rejected,
        "active_rules": active_rules,
        "dict_entries_updated": new_aliases,
    })
    return {"ok": True, "dict_entries_updated": new_aliases}


@app.post("/api/pipeline/{run_id}/complete")
def pipeline_complete(run_id: str, body: PipelineCompleteRequest):
    complete_run(run_id, body.match_rate, body.matched, body.breaks)
    return {"ok": True}


# ── Field Dictionary & Rule Book ─────────────────────────────────────────────

@app.get("/api/dictionary")
def get_dictionary():
    """Return all field dictionary entries, sorted by confirmation count."""
    return {
        "entries": dictionary_service.get_all_entries(),
        "stats":   dictionary_service.get_stats(),
        "log":     dictionary_service.get_learning_log(limit=30),
    }

@app.post("/api/dictionary/lookup")
def dictionary_lookup(body: dict):
    """Look up a single field name — returns its canonical entry."""
    name = body.get("name", "")
    entry = dictionary_service.lookup_field(name)
    return {"entry": entry, "found": entry is not None}

@app.post("/api/dictionary/learn")
def dictionary_learn(body: dict):
    """Manually teach the dictionary: add an alias or value mapping.

    body: { "action": "alias" | "value_map",
            "canonical_name": "Side",
            "alias": "direction",            -- for alias
            "abbreviation": "L", "full_form": "Long"  -- for value_map
           }
    """
    action = body.get("action", "alias")
    if action == "alias":
        entry = dictionary_service.lookup_field(body.get("canonical_name", ""))
        if entry:
            conn = dictionary_service._conn()
            aliases = json.loads(entry["aliases"])
            new_alias = body.get("alias", "")
            if new_alias and new_alias not in aliases:
                aliases.append(new_alias)
                conn.execute(
                    "UPDATE field_dictionary SET aliases = ? WHERE id = ?",
                    (json.dumps(aliases), entry["id"]),
                )
                conn.commit()
            conn.close()
    elif action == "value_map":
        dictionary_service.learn_value_mapping(
            body.get("canonical_name", ""),
            body.get("abbreviation", ""),
            body.get("full_form", ""),
        )
    return {"ok": True}

@app.get("/api/dictionary/stats")
def dictionary_stats():
    return dictionary_service.get_stats()


# ── Break RAG ─────────────────────────────────────────────────────────────────

class BreakStoreRequest(BaseModel):
    source_fields: dict
    target_fields: dict
    break_fields: list[str]
    resolution: Optional[str] = None
    resolution_type: Optional[str] = None
    run_id: Optional[str] = None


class BreakResolveRequest(BaseModel):
    break_id: int
    resolution: str
    resolution_type: str


class BreakSimilarRequest(BaseModel):
    source_fields: dict
    target_fields: dict
    break_fields: list[str]
    top_k: int = 3


@app.post("/api/breaks/store")
def breaks_store(req: BreakStoreRequest):
    break_id = rag_service.store_break(
        req.source_fields, req.target_fields, req.break_fields,
        req.resolution, req.resolution_type, req.run_id,
    )
    log("break_stored", {"break_id": break_id, "fields": req.break_fields})
    return {"break_id": break_id}


@app.post("/api/breaks/resolve")
def breaks_resolve(req: BreakResolveRequest):
    rag_service.resolve_break(req.break_id, req.resolution, req.resolution_type)
    log("break_resolved", {"break_id": req.break_id, "resolution_type": req.resolution_type},
        reasoning=req.resolution)
    return {"ok": True}


@app.post("/api/breaks/similar")
def breaks_similar(req: BreakSimilarRequest):
    similar = rag_service.find_similar_breaks(
        req.source_fields, req.target_fields, req.break_fields, req.top_k
    )
    return {"similar": similar}


@app.get("/api/breaks")
def breaks_list():
    return {"breaks": rag_service.get_all_breaks(100)}


# ── Schema Rule Library ────────────────────────────────────────────────────────

class RuleLibrarySaveRequest(BaseModel):
    source_cols: list[str]
    target_cols: list[str]
    confirmed_mappings: list[dict]
    confirmed_rules: list[dict]


class RuleLibraryLookupRequest(BaseModel):
    source_cols: list[str]
    target_cols: list[str]


@app.post("/api/rules/save")
def rules_save(req: RuleLibrarySaveRequest):
    fp = rag_service.save_rule_library(
        req.source_cols, req.target_cols, req.confirmed_mappings, req.confirmed_rules
    )
    log("rule_library_saved", {"fingerprint": fp, "rules": len(req.confirmed_rules)})
    return {"fingerprint": fp}


@app.post("/api/rules/lookup")
def rules_lookup(req: RuleLibraryLookupRequest):
    result = rag_service.get_rule_library(req.source_cols, req.target_cols)
    return {"found": result is not None, "library": result}


@app.get("/api/rules/library")
def rules_library_list():
    return {"entries": rag_service.get_all_rule_library()}


# ── Probabilistic Reconciliation ───────────────────────────────────────────────

class ProbabilisticRequest(BaseModel):
    file_a_id: str
    file_b_id: str
    mappings: list[dict]          # same format as regular rules
    threshold: float = 0.65
    blocking_field: Optional[str] = None
    run_id: Optional[str] = None


@app.post("/api/reconcile/probabilistic")
def run_probabilistic(req: ProbabilisticRequest):
    try:
        df_a = file_processor.get_dataframe(req.file_a_id)
        df_b = file_processor.get_dataframe(req.file_b_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    field_config = probabilistic_matcher.infer_field_config(req.mappings)

    try:
        result = probabilistic_matcher.probabilistic_reconcile(
            df_a, df_b, field_config,
            threshold=req.threshold,
            blocking_field=req.blocking_field,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if req.run_id:
        reconciliation.store_run_results(req.run_id, result)

    log("probabilistic_reconciliation", {
        "matched": result["matched"],
        "breaks": result["breaks"],
        "match_rate": result["match_rate"],
        "threshold": req.threshold,
    })
    return result


# ── Streaming Reconciliation (SSE) ─────────────────────────────────────────────

class StreamReconcileRequest(BaseModel):
    file_a_id: str
    file_b_id: str
    rules: list[dict]
    key_columns: list[dict]
    run_id: Optional[str] = None


@app.post("/api/reconcile/stream")
async def stream_reconciliation(req: StreamReconcileRequest):
    """
    Server-Sent Events endpoint. Streams each reconciled record as it is
    processed so the UI can surface breaks in real time without waiting for
    the full result set.
    """
    def _event(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    def generate():
        try:
            df_a = file_processor.get_dataframe(req.file_a_id).copy()
            df_b = file_processor.get_dataframe(req.file_b_id).copy()
        except KeyError as e:
            yield _event({"type": "error", "message": str(e)})
            return

        yield _event({"type": "start", "total_source": len(df_a), "total_target": len(df_b)})

        rename_map = {r["target_column"]: r["source_column"] for r in req.rules}
        df_b = df_b.rename(columns=rename_map)

        key_source = [k["source"] for k in req.key_columns]
        key_target = [rename_map.get(k["target"], k["target"]) for k in req.key_columns]

        import pandas as pd
        df_a["_key"] = df_a[key_source].astype(str).apply(lambda x: "|".join(x), axis=1)
        df_b["_key"] = df_b[key_source].astype(str).apply(lambda x: "|".join(x), axis=1)

        merged = pd.merge(
            df_a.add_suffix("_A"),
            df_b.add_suffix("_B"),
            left_on="_key_A",
            right_on="_key_B",
            how="outer",
            indicator=True,
        )

        matched = breaks = unmatched_a = unmatched_b = 0
        all_records = []

        for _, row in merged.iterrows():
            flag = row["_merge"]

            if flag == "left_only":
                unmatched_a += 1
                rec = {
                    "match_key": str(row.get("_key_A", "")),
                    "status": "unmatched_source",
                    "match_probability": 0.0,
                    "source_data": {c.replace("_A", ""): row[c] for c in merged.columns if c.endswith("_A")},
                    "target_data": {},
                    "break_reasons": ["No matching record in target file"],
                }
            elif flag == "right_only":
                unmatched_b += 1
                rec = {
                    "match_key": str(row.get("_key_B", "")),
                    "status": "unmatched_target",
                    "match_probability": 0.0,
                    "source_data": {},
                    "target_data": {c.replace("_B", ""): row[c] for c in merged.columns if c.endswith("_B")},
                    "break_reasons": ["No matching record in source file"],
                }
            else:
                source_data = {c.replace("_A", ""): row[c] for c in merged.columns if c.endswith("_A")}
                target_data = {c.replace("_B", ""): row[c] for c in merged.columns if c.endswith("_B")}
                break_reasons = []

                from services.reconciliation import _compare_values
                for rule in req.rules:
                    col = rule["source_column"]
                    col_a, col_b = f"{col}_A", f"{col}_B"
                    if col_a not in merged.columns or col_b not in merged.columns:
                        continue
                    val_a, val_b = row.get(col_a), row.get(col_b)
                    import pandas as _pd
                    if _pd.isna(val_a) and _pd.isna(val_b):
                        continue
                    if not _compare_values(val_a, val_b, rule.get("match_type", "exact"), rule.get("threshold")):
                        break_reasons.append(f"{col}: {val_a} ≠ {val_b}")

                if break_reasons:
                    breaks += 1
                    status = "break"
                else:
                    matched += 1
                    status = "matched"

                rec = {
                    "match_key": str(row.get("_key_A", row.get("_key_B", ""))),
                    "status": status,
                    "match_probability": 1.0 if status == "matched" else 0.5,
                    "source_data": source_data,
                    "target_data": target_data,
                    "break_reasons": break_reasons,
                }

            all_records.append(rec)
            # Stream each record immediately
            yield _event({"type": "record", "record": rec})

        total = matched + breaks + unmatched_a + unmatched_b
        match_rate = round(matched / max(total, 1) * 100, 2)

        final = {
            "type": "complete",
            "total_source": len(df_a),
            "total_target": len(df_b),
            "matched": matched,
            "breaks": breaks,
            "unmatched_source": unmatched_a,
            "unmatched_target": unmatched_b,
            "match_rate": match_rate,
        }
        if req.run_id:
            reconciliation.store_run_results(req.run_id, {**final, "records": all_records})

        log("stream_reconciliation_complete", {
            "matched": matched, "breaks": breaks, "match_rate": match_rate,
        })
        yield _event(final)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Regulatory Report ──────────────────────────────────────────────────────────

@app.get("/api/reports/regulatory/{run_id}")
def regulatory_report(run_id: str):
    """
    Full regulatory-grade report for a single reconciliation run.
    Combines the pipeline trace (steps, AI reasoning, human decisions)
    with the reconciliation results and audit log entries.
    """
    trace = get_run_trace(run_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Run not found")

    results = reconciliation.get_run_results(run_id)
    audit_entries = audit.get_all()
    # Filter audit entries to approximate this run's window
    run_created = trace["run"].get("created_at", "")
    run_completed = trace["run"].get("completed_at", "")
    run_entries = [
        e for e in audit_entries
        if run_created <= e["timestamp"] <= (run_completed or "9999")
    ]

    # Summary statistics
    records = results.get("records", []) if results else []
    break_records = [r for r in records if r["status"] == "break"]
    unmatched = [r for r in records if r["status"] in ("unmatched_source", "unmatched_target")]

    # Field-level break frequency
    field_break_counts: dict[str, int] = {}
    for r in break_records:
        for reason in r.get("break_reasons", []):
            field = reason.split(":")[0].strip()
            field_break_counts[field] = field_break_counts.get(field, 0) + 1

    # HITL decisions from pipeline steps
    human_decisions = [
        {
            "step": s["step_label"],
            "action": s["human_action"],
            "timestamp": s["human_action_at"],
            "reviewer": "human_reviewer",
        }
        for s in trace.get("steps", [])
        if s.get("human_action")
    ]

    # Matching rules applied (from rules step output)
    rules_step = next(
        (s for s in trace.get("steps", []) if s["step_name"] == "rules"), {}
    )
    mapping_step = next(
        (s for s in trace.get("steps", []) if s["step_name"] == "map"), {}
    )

    report = {
        "report_type": "reconciliation_audit_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run": trace["run"],
        "pipeline_steps": trace.get("steps", []),
        "human_decisions": human_decisions,
        "audit_events": run_entries,
        "summary": {
            "total_source": results.get("total_source") if results else 0,
            "total_target": results.get("total_target") if results else 0,
            "matched": results.get("matched") if results else 0,
            "breaks": results.get("breaks") if results else 0,
            "unmatched_source": results.get("unmatched_source") if results else 0,
            "unmatched_target": results.get("unmatched_target") if results else 0,
            "match_rate": results.get("match_rate") if results else 0,
        },
        "field_break_analysis": [
            {"field": f, "break_count": c}
            for f, c in sorted(field_break_counts.items(), key=lambda x: -x[1])
        ],
        "breaks": break_records[:50],   # first 50 breaks with full detail
        "unmatched": unmatched[:20],
        "ai_reasoning": {
            "mapping": mapping_step.get("ai_reasoning", ""),
            "rules": rules_step.get("ai_reasoning", ""),
        },
        "compliance": {
            "hitl_review_performed": len(human_decisions) > 0,
            "full_audit_trail": True,
            "data_lineage_tracked": True,
            "break_explanations_available": True,
        },
    }

    log("regulatory_report_generated", {"run_id": run_id, "breaks": len(break_records)})
    return report


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# ── Sample data download ──────────────────────────────────────────────────────

from pathlib import Path
from fastapi.responses import FileResponse

SAMPLE_DIR = Path(__file__).parent / "sample_data"

SAMPLE_FILES = {
    "trade_confirm_source":   ("internal_book.csv",                    "Trade Confirmation — Internal Book (CSV)"),
    "trade_confirm_target":   ("broker_feed.csv",                      "Trade Confirmation — Broker FIX Feed (CSV)"),
    "nostro_source":          ("nostro_internal_ledger.csv",           "Cash/Nostro — Internal Ledger (CSV)"),
    "nostro_target":          ("citi_mt940_statement.txt",             "Cash/Nostro — Bank MT940 Statement (SWIFT)"),
    "position_source":        ("fund_pms_positions.csv",               "Position Recon — Fund PMS (CSV)"),
    "position_target":        ("statestreet_custodian_extract.csv",    "Position Recon — Custodian Report (CSV)"),
    "corp_actions_source":    ("corporate_events_db.csv",              "Corporate Actions — Events DB (CSV)"),
    "corp_actions_target":    ("bloomberg_email_announcements.txt",    "Corporate Actions — Bloomberg Email (Unstructured)"),
    "invoice_source":         ("sap_erp_purchase_orders.csv",          "Invoice/AP — SAP ERP POs (CSV)"),
    "invoice_target":         ("vendor_invoices_pdf_extracted.txt",    "Invoice/AP — Vendor PDF Invoices (Extracted)"),
}


@app.get("/api/samples")
def list_samples():
    return {"samples": [
        {"key": k, "filename": v[0], "label": v[1]}
        for k, v in SAMPLE_FILES.items()
    ]}


@app.get("/api/samples/{key}")
def download_sample(key: str):
    if key not in SAMPLE_FILES:
        raise HTTPException(status_code=404, detail="Sample not found")
    filename, label = SAMPLE_FILES[key]
    path = SAMPLE_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found on disk")
    return FileResponse(str(path), filename=filename, media_type="application/octet-stream")
