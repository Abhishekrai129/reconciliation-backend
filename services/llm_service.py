"""
LLM Service — wraps Anthropic / OpenAI / Ollama with dictionary-augmented prompts.

Key design:
  1. Every schema-mapping call receives the Field Dictionary as context.
     This gives the LLM ground truth for aliases (bs_ind=Side) and value maps (B→Buy).
  2. LLM responses are normalized: source_column / target_column are mapped back to
     the ACTUAL column names present in the files (not inferred labels).
  3. suggest_matching_rules falls back to heuristic rules if LLM returns empty.
"""

import json
import os
import re
import time
from typing import Any
from models.schemas import LLMConfig, LLMProvider


_config = LLMConfig()

# Current run context — set once per pipeline run so all LLM calls are tagged
_current_run_id: str = ""
_current_step: str = ""


def set_llm_config(config: LLMConfig):
    global _config
    _config = config


def get_llm_config() -> LLMConfig:
    return _config


def set_run_context(run_id: str, step: str = ""):
    """Call at the start of each pipeline step so LLM traces are tagged."""
    global _current_run_id, _current_step
    _current_run_id = run_id or ""
    _current_step = step or ""


# ── LLM call router ───────────────────────────────────────────────────────────

async def call_llm(prompt: str, system: str = "", step: str = "") -> str:
    cfg = get_llm_config()
    effective_step = step or _current_step

    if cfg.provider == LLMProvider.anthropic:
        return await _call_anthropic(prompt, system, cfg, effective_step)
    elif cfg.provider == LLMProvider.openai:
        return await _call_openai(prompt, system, cfg, effective_step)
    elif cfg.provider == LLMProvider.ollama:
        return await _call_ollama(prompt, system, cfg, effective_step)

    raise ValueError(f"Unknown provider: {cfg.provider}")


async def _call_anthropic(prompt: str, system: str, cfg: LLMConfig, step: str) -> str:
    from services import audit
    import anthropic
    key = cfg.api_key or os.getenv("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key)
    raw_response = ""
    tokens_in = tokens_out = None
    t0 = time.perf_counter()
    error = None
    try:
        msg = client.messages.create(
            model=cfg.model,
            max_tokens=4096,
            system=system or "You are an expert financial data analyst specializing in reconciliation.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw_response = msg.content[0].text
        tokens_in  = getattr(msg.usage, "input_tokens",  None)
        tokens_out = getattr(msg.usage, "output_tokens", None)
        return raw_response
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        audit.log_llm_call(
            run_id=_current_run_id, step=step, provider="anthropic", model=cfg.model,
            prompt=prompt, raw_response=raw_response,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            error=error,
        )


async def _call_openai(prompt: str, system: str, cfg: LLMConfig, step: str) -> str:
    from services import audit
    from openai import OpenAI
    key = cfg.api_key or os.getenv("OPENAI_API_KEY", "") or os.getenv("OPENROUTER_API_KEY", "")
    base_url = cfg.base_url or os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1"
    client = OpenAI(api_key=key, **({"base_url": base_url} if base_url else {}))
    raw_response = ""
    tokens_in = tokens_out = None
    t0 = time.perf_counter()
    error = None
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system or "You are an expert financial data analyst."},
                {"role": "user", "content": prompt},
            ],
        )
        raw_response = resp.choices[0].message.content
        if resp.usage:
            tokens_in  = resp.usage.prompt_tokens
            tokens_out = resp.usage.completion_tokens
        return raw_response
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        audit.log_llm_call(
            run_id=_current_run_id, step=step, provider=str(cfg.provider), model=cfg.model,
            prompt=prompt, raw_response=raw_response,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            error=error,
        )


async def _call_ollama(prompt: str, system: str, cfg: LLMConfig, step: str) -> str:
    from services import audit
    import ollama
    base_url = cfg.base_url or "http://localhost:11434"
    client = ollama.Client(host=base_url)
    raw_response = ""
    tokens_in = tokens_out = None
    t0 = time.perf_counter()
    error = None
    try:
        resp = client.chat(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system or "You are an expert financial data analyst."},
                {"role": "user", "content": prompt},
            ],
        )
        raw_response = resp["message"]["content"]
        tokens_in  = resp.get("prompt_eval_count")
        tokens_out = resp.get("eval_count")
        return raw_response
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        audit.log_llm_call(
            run_id=_current_run_id, step=step, provider="ollama", model=cfg.model,
            prompt=prompt, raw_response=raw_response,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            error=error,
        )


# ── JSON parsing helper ───────────────────────────────────────────────────────

def parse_json_from_llm(text: str) -> Any:
    """Robustly extract JSON from LLM response (handles markdown fences)."""
    cleaned = text.strip()
    # Strip markdown code fences
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            if part.startswith("json"):
                part = part[4:]
            try:
                return json.loads(part.strip())
            except Exception:
                continue
    # Try direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Try to extract first {...} or [...] block
    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', cleaned)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    raise ValueError("No valid JSON found in LLM response")


# ── Field understanding ───────────────────────────────────────────────────────

async def understand_fields(columns: list[dict]) -> list[dict]:
    """Use LLM + dictionary to understand what each column represents."""
    from services.dictionary_service import get_context_for_mapping

    cfg = get_llm_config()
    col_names = [c.get("name", str(c)) for c in columns]

    if cfg.privacy_mode == "headers_only":
        columns_for_ai = [{"name": c.get("name", c)} for c in columns]
        privacy_note = "\nNote: only column names are provided (privacy mode). Infer types from naming conventions."
    else:
        columns_for_ai = columns
        privacy_note = ""

    dict_context = get_context_for_mapping(col_names, [])

    prompt = f"""You are analyzing financial data columns for a reconciliation system.

{dict_context}

For each column below, determine:
1. A human-readable label matching one of the canonical field names in the dictionary above where possible
2. The semantic type (date, price, quantity, identifier, currency, side, text)
3. Confidence score (0.0 to 1.0)
4. Brief reasoning (1 sentence)

Columns to analyze:
{json.dumps(columns_for_ai, indent=2)}{privacy_note}

IMPORTANT: Use the EXACT "name" field from each column object as the "name" in your response.
Do NOT rename or substitute the name field.

Respond ONLY with a valid JSON array:
[
  {{
    "name": "trd_dt",
    "inferred_label": "Trade Date",
    "inferred_type": "date",
    "confidence": 0.97,
    "reasoning": "trd_dt is a known alias for Trade Date per the field dictionary."
  }}
]"""

    response = await call_llm(prompt)
    try:
        result = parse_json_from_llm(response)
        if isinstance(result, list):
            # Ensure name field is preserved from original columns
            name_map = {c.get("name"): c for c in columns}
            for item in result:
                orig_name = item.get("name")
                if orig_name not in name_map:
                    # LLM swapped the name — restore it
                    for col in columns:
                        if col.get("name", "").lower() == str(orig_name).lower():
                            item["name"] = col["name"]
                            break
            return result
    except Exception:
        pass
    return columns


# ── Schema mapping ────────────────────────────────────────────────────────────

async def map_schemas(file_a_columns: list[dict], file_b_columns: list[dict]) -> dict:
    """Use LLM + dictionary context to map columns across two files.

    CRITICAL: source_column and target_column in the response MUST be the actual
    column names (the 'name' field), not inferred labels. This function normalizes
    the LLM output back to actual column names after parsing.
    """
    from services.dictionary_service import get_context_for_mapping, normalize_column_name, get_rejection_context

    # Extract actual column names for normalization after LLM call
    names_a = [c.get("name", str(c)) for c in file_a_columns]
    names_b = [c.get("name", str(c)) for c in file_b_columns]

    dict_context      = get_context_for_mapping(names_a, names_b)
    rejection_context = get_rejection_context(names_a, names_b)

    # Build a compact column list showing name + label side by side
    def _col_summary(cols: list[dict]) -> str:
        rows = []
        for c in cols:
            name = c.get("name", "?")
            label = c.get("inferred_label", name)
            dtype = c.get("inferred_type", c.get("dtype", ""))
            sample = c.get("sample_values", [])
            sample_str = f"  sample={json.dumps(sample[:3])}" if sample else ""
            rows.append(f'  "{name}" ({label}, {dtype}){sample_str}')
        return "\n".join(rows)

    rejection_block = f"\n{rejection_context}\n" if rejection_context else ""

    prompt = f"""You are mapping fields between two financial data sources for reconciliation.

{dict_context}
{rejection_block}

FILE A columns (name | inferred_label | type):
{_col_summary(file_a_columns)}

FILE B columns (name | inferred_label | type):
{_col_summary(file_b_columns)}

For each column in File A, find the best matching column in File B.
Use the Field Dictionary above to identify synonyms and value normalisation needs.

CRITICAL RULES:
1. "source_column" MUST be the exact name from FILE A (the first part before the parenthesis)
2. "target_column" MUST be the exact name from FILE B (the first part before the parenthesis)
3. Do NOT use the inferred label — use the raw name exactly as shown
4. match_type: "value_lookup" when the Side/Dr-Cr field needs B→Buy normalisation

Respond ONLY with valid JSON:
{{
  "mappings": [
    {{
      "source_column": "trd_dt",
      "target_column": "Trade_Date",
      "confidence": 0.97,
      "match_type": "exact",
      "tolerance": null,
      "reasoning": "Both are Trade Date. Values match exactly."
    }}
  ],
  "unmapped_source": [],
  "unmapped_target": []
}}

Match types: exact | fuzzy | numeric_tolerance | date_tolerance | value_lookup"""

    response = await call_llm(prompt)
    try:
        result = parse_json_from_llm(response)

        # Handle LLM returning array directly instead of wrapped object
        if isinstance(result, list):
            result = {"mappings": result, "unmapped_source": [], "unmapped_target": []}

        mappings = result.get("mappings", [])

        # NORMALIZE: ensure source_column / target_column are actual file column names
        normalized = []
        for m in mappings:
            src = normalize_column_name(str(m.get("source_column", "")), names_a)
            tgt = normalize_column_name(str(m.get("target_column", "")), names_b)
            normalized.append({
                "source_column": src,
                "target_column": tgt,
                "confidence": float(m.get("confidence", 0.8)),
                "match_type": m.get("match_type", "exact"),
                "tolerance": m.get("tolerance"),
                "reasoning": m.get("reasoning", ""),
            })

        result["mappings"] = normalized
        return result

    except Exception:
        return {"mappings": [], "unmapped_source": names_a, "unmapped_target": names_b}


# ── Agent Planner ─────────────────────────────────────────────────────────────

async def plan_reconciliation(file_a_cols: list[dict], file_b_cols: list[dict]) -> dict:
    """Agent planner: decompose the reconciliation task into a strategy before any mapping."""
    names_a = [c.get("name", str(c)) for c in file_a_cols]
    names_b = [c.get("name", str(c)) for c in file_b_cols]

    prompt = f"""You are a reconciliation planning agent. Analyze these two datasets and produce a structured plan before any mapping begins.

SOURCE columns: {json.dumps(names_a)}
TARGET columns: {json.dumps(names_b)}

Respond with a JSON plan:
{{
  "dataset_assessment": "1-2 sentence description of what each dataset is",
  "complexity": "low|medium|high",
  "complexity_reason": "why",
  "concept_groups": [
    {{"concept": "Trade Identity", "source_cols": ["col1","col2"], "target_cols": ["colA"], "notes": "..."}}
  ],
  "recommended_strategy": "exact|fuzzy|probabilistic|hybrid",
  "key_fields": ["list of likely join key fields"],
  "risk_flags": ["list of potential issues: format mismatch, encoding differences, etc."],
  "agent_steps": [
    {{"step": "Understand Fields", "agent": "Profiler Agent", "goal": "..."}},
    {{"step": "Map Schemas", "agent": "Maker Agent", "goal": "..."}},
    {{"step": "Validate Mappings", "agent": "Checker Agent", "goal": "..."}},
    {{"step": "Suggest Rules", "agent": "Rules Agent", "goal": "..."}},
    {{"step": "Human Review", "agent": "Human", "goal": "..."}}
  ]
}}"""

    response = await call_llm(prompt, step="plan")
    try:
        return parse_json_from_llm(response)
    except Exception:
        return {
            "dataset_assessment": "Could not analyze — AI unavailable",
            "complexity": "medium",
            "complexity_reason": "Unable to assess",
            "concept_groups": [],
            "recommended_strategy": "exact",
            "key_fields": [],
            "risk_flags": ["AI planning unavailable — using heuristics"],
            "agent_steps": [],
        }


# ── Maker-Checker Validation ──────────────────────────────────────────────────

async def validate_mappings_checker(
    proposed_mappings: list[dict],
    file_a_cols: list[dict],
    file_b_cols: list[dict],
) -> dict:
    """Checker Agent: independently reviews the Maker Agent's proposed mappings."""
    names_a = [c.get("name", str(c)) for c in file_a_cols]
    names_b = [c.get("name", str(c)) for c in file_b_cols]

    prompt = f"""You are the CHECKER AGENT in a Maker-Checker framework. The Maker Agent proposed these schema mappings.
Your job is to independently verify each mapping, flag any errors, and suggest corrections.

SOURCE columns available: {json.dumps(names_a)}
TARGET columns available: {json.dumps(names_b)}

MAKER AGENT'S PROPOSED MAPPINGS:
{json.dumps(proposed_mappings, indent=2)}

For each mapping, assess:
1. Is this mapping semantically correct?
2. Are there better alternatives?
3. Is the confidence appropriate?

Respond ONLY with JSON:
{{
  "verdict": "approved|approved_with_changes|rejected",
  "overall_confidence": 0.0-1.0,
  "checker_notes": "Overall assessment in 1-2 sentences",
  "mapping_reviews": [
    {{
      "source_column": "col_name",
      "target_column": "col_name",
      "checker_verdict": "approved|flagged|rejected",
      "checker_note": "brief reason",
      "suggested_correction": null
    }}
  ],
  "flagged_count": 0,
  "approved_count": 0
}}"""

    response = await call_llm(prompt, step="validate_mappings")
    try:
        return parse_json_from_llm(response)
    except Exception:
        approved = len(proposed_mappings)
        return {
            "verdict": "approved",
            "overall_confidence": 0.75,
            "checker_notes": "Checker AI unavailable — mappings passed without validation",
            "mapping_reviews": [
                {
                    "source_column": m.get("source_column"),
                    "target_column": m.get("target_column"),
                    "checker_verdict": "approved",
                    "checker_note": "Auto-approved (checker unavailable)",
                    "suggested_correction": None,
                }
                for m in proposed_mappings
            ],
            "flagged_count": 0,
            "approved_count": approved,
        }


# ── Break Resolution with Tool Calls ─────────────────────────────────────────

async def resolve_break_with_actions(
    source_record: dict,
    target_record: dict,
    break_fields: list[str],
    context: str = "",
) -> dict:
    """Agentic break resolver: explains break AND proposes concrete tool-call actions."""
    prompt = f"""You are a break resolution agent. A reconciliation break was found. Analyze it and propose concrete resolution actions.

Source record: {json.dumps(source_record, indent=2)}
Target record: {json.dumps(target_record, indent=2)}
Fields with differences: {json.dumps(break_fields)}
{f'Context: {context}' if context else ''}

Respond ONLY with JSON:
{{
  "root_cause": "1-2 sentence explanation of why this break occurred",
  "severity": "low|medium|high|critical",
  "auto_resolvable": true|false,
  "auto_resolution": "what the system can fix automatically, or null",
  "proposed_actions": [
    {{
      "tool": "slack_message|erp_journal_entry|email_counterparty|flag_for_review|apply_tolerance_rule",
      "label": "Human-readable action label",
      "description": "What this action would do",
      "payload": {{}},
      "requires_approval": true|false
    }}
  ],
  "historical_pattern": "Is this a known recurring break pattern? Describe if so."
}}"""

    response = await call_llm(prompt, step="resolve_break")
    try:
        return parse_json_from_llm(response)
    except Exception:
        return {
            "root_cause": "Break analysis unavailable — AI offline",
            "severity": "medium",
            "auto_resolvable": False,
            "auto_resolution": None,
            "proposed_actions": [
                {
                    "tool": "flag_for_review",
                    "label": "Flag for Manual Review",
                    "description": "Escalate to operations team for manual investigation",
                    "payload": {"fields": break_fields},
                    "requires_approval": True,
                }
            ],
            "historical_pattern": "Unable to check historical patterns — AI offline",
        }


# ── Break explanation ─────────────────────────────────────────────────────────

async def explain_break(source_record: dict, target_record: dict, break_fields: list[str]) -> str:
    """Use LLM to explain why two records didn't match."""
    prompt = f"""A reconciliation break was found. Explain why these two records don't match and suggest resolution.

Source record: {json.dumps(source_record, indent=2)}
Target record: {json.dumps(target_record, indent=2)}
Fields with differences: {break_fields}

Provide a concise explanation (2-3 sentences) of:
1. What specifically differs
2. Why this might have occurred (rounding, timezone, system difference, etc.)
3. Suggested action to resolve

Be specific about the actual values."""

    return await call_llm(prompt)


# ── Rule suggestion ───────────────────────────────────────────────────────────

_HEURISTIC_RULES = {
    "side":       ("value_lookup", None,  {"B": "Buy", "S": "Sell", "D": "Debit", "C": "Credit"}),
    "price":      ("numeric_tolerance", 0.01, None),
    "numeric":    ("numeric_tolerance", 0.01, None),
    "quantity":   ("numeric_tolerance", 1.0,  None),
    "date":       ("date_tolerance",    0,    None),
    "identifier": ("exact",             None, None),
    "currency":   ("exact",             None, None),
    "text":       ("fuzzy",             0.85, None),
}


def _heuristic_rule(source_col: str, target_col: str, inferred_type: str = "text") -> dict:
    match_type, threshold, lookup = _HEURISTIC_RULES.get(inferred_type, _HEURISTIC_RULES["text"])
    rule: dict = {
        "source_column": source_col,
        "target_column": target_col,
        "match_type": match_type,
        "threshold": lookup if lookup else threshold,
        "reasoning": f"Heuristic rule for {inferred_type} field",
    }
    return rule


async def suggest_matching_rules(mappings: list[dict], sample_data: dict) -> list[dict]:
    """Use LLM + dictionary to suggest matching rules for each mapped field pair.

    Falls back to type-based heuristic rules if LLM returns empty or fails.
    """
    from services.dictionary_service import get_context_for_mapping

    if not mappings:
        return []

    src_cols = [m.get("source_column", "") for m in mappings]
    tgt_cols = [m.get("target_column", "") for m in mappings]
    dict_context = get_context_for_mapping(src_cols, tgt_cols)

    prompt = f"""Based on these field mappings, sample data, and the field dictionary, suggest the best matching rule for each pair.

{dict_context}

Mappings to create rules for:
{json.dumps(mappings, indent=2)}

Sample data (first 5 rows each side):
{json.dumps(sample_data, indent=2)}

For each mapping, choose:
- match_type: exact | fuzzy | numeric_tolerance | date_tolerance | value_lookup
- threshold: null for exact, 0-1 for fuzzy similarity, numeric for tolerance, OR a dict for value_lookup
- For value_lookup fields (Side, Dr/Cr) use the value_map from the dictionary above

IMPORTANT: "source_column" and "target_column" must exactly match the mapping input values.

Respond ONLY with valid JSON array:
[
  {{
    "source_column": "exec_px",
    "target_column": "Execution_Price",
    "match_type": "numeric_tolerance",
    "threshold": 0.01,
    "reasoning": "Price fields may differ by rounding — 0.01 covers 2 decimal places."
  }},
  {{
    "source_column": "bs_ind",
    "target_column": "Side",
    "match_type": "value_lookup",
    "threshold": {{"B": "Buy", "S": "Sell", "BUY": "Buy", "SELL": "Sell"}},
    "reasoning": "Side codes need normalisation: B/BUY→Buy, S/SELL→Sell."
  }}
]"""

    response = await call_llm(prompt)
    try:
        result = parse_json_from_llm(response)
        rules = result if isinstance(result, list) else result.get("rules", [])

        if rules:
            # Normalize column names in rules to match actual mapping names
            src_map = {m.get("source_column", "").lower(): m.get("source_column") for m in mappings}
            tgt_map = {m.get("target_column", "").lower(): m.get("target_column") for m in mappings}
            for r in rules:
                src = r.get("source_column", "")
                tgt = r.get("target_column", "")
                r["source_column"] = src_map.get(src.lower(), src)
                r["target_column"] = tgt_map.get(tgt.lower(), tgt)
            return rules

    except Exception:
        pass

    # ── Heuristic fallback: derive rules from mappings + dictionary ──────────
    from services.dictionary_service import lookup_field
    heuristic_rules = []
    for m in mappings:
        src = m.get("source_column", "")
        tgt = m.get("target_column", "")
        # Infer type from dictionary lookup
        entry = lookup_field(src) or lookup_field(tgt)
        dtype = entry["data_type"] if entry else "text"
        heuristic_rules.append(_heuristic_rule(src, tgt, dtype))
    return heuristic_rules
