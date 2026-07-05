import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_pipeline_db()
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
    # Mark review step started if not already
    start_step(run_id, "review", f"{total} mappings to review, {active_rules} rules")
    record_human_action(run_id, "review", action_summary)
    log("pipeline_review", {
        "run_id": run_id,
        "approved_mappings": approved,
        "rejected_mappings": rejected,
        "active_rules": active_rules,
    })
    return {"ok": True}


@app.post("/api/pipeline/{run_id}/complete")
def pipeline_complete(run_id: str, body: PipelineCompleteRequest):
    complete_run(run_id, body.match_rate, body.matched, body.breaks)
    return {"ok": True}


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
