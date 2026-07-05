import json
import os
from typing import Any
from models.schemas import LLMConfig, LLMProvider


_config = LLMConfig()


def set_llm_config(config: LLMConfig):
    global _config
    _config = config


def get_llm_config() -> LLMConfig:
    return _config


async def call_llm(prompt: str, system: str = "") -> str:
    cfg = get_llm_config()

    if cfg.provider == LLMProvider.anthropic:
        return await _call_anthropic(prompt, system, cfg)
    elif cfg.provider == LLMProvider.openai:
        return await _call_openai(prompt, system, cfg)
    elif cfg.provider == LLMProvider.ollama:
        return await _call_ollama(prompt, system, cfg)

    raise ValueError(f"Unknown provider: {cfg.provider}")


async def _call_anthropic(prompt: str, system: str, cfg: LLMConfig) -> str:
    import anthropic
    key = cfg.api_key or os.getenv("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=cfg.model,
        max_tokens=4096,
        system=system or "You are an expert financial data analyst specializing in reconciliation.",
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


async def _call_openai(prompt: str, system: str, cfg: LLMConfig) -> str:
    from openai import OpenAI
    key = cfg.api_key or os.getenv("OPENAI_API_KEY", "")
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system or "You are an expert financial data analyst."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


async def _call_ollama(prompt: str, system: str, cfg: LLMConfig) -> str:
    import ollama
    base_url = cfg.base_url or "http://localhost:11434"
    client = ollama.Client(host=base_url)
    resp = client.chat(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system or "You are an expert financial data analyst."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp["message"]["content"]


async def understand_fields(columns: list[dict]) -> list[dict]:
    """Use LLM to understand what each column represents."""
    prompt = f"""You are analyzing financial data columns for a reconciliation system.

For each column below, determine:
1. A human-readable label (e.g., "Trade Date", "Quantity", "Price")
2. The semantic type (date, price, quantity, identifier, currency, side, text)
3. Confidence score (0.0 to 1.0)
4. Brief reasoning (1 sentence)

Columns to analyze:
{json.dumps(columns, indent=2)}

Respond ONLY with a valid JSON array. Example:
[
  {{
    "name": "trd_dt",
    "inferred_label": "Trade Date",
    "inferred_type": "date",
    "confidence": 0.97,
    "reasoning": "Column name is an abbreviation of trade date and values follow ISO date format."
  }}
]"""

    response = await call_llm(prompt)
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned)
    except Exception:
        return columns


async def map_schemas(file_a_columns: list[dict], file_b_columns: list[dict]) -> dict:
    """Use LLM to map columns across two files."""
    prompt = f"""You are mapping fields between two financial data sources for reconciliation.

FILE A columns (with understood labels):
{json.dumps(file_a_columns, indent=2)}

FILE B columns (with understood labels):
{json.dumps(file_b_columns, indent=2)}

For each column in File A, find the best matching column in File B.
Consider: semantic meaning, data type, value patterns, financial domain knowledge.

Respond ONLY with valid JSON:
{{
  "mappings": [
    {{
      "source_column": "Trade Date",
      "target_column": "trd_dt",
      "confidence": 0.97,
      "match_type": "exact",
      "tolerance": null,
      "reasoning": "Both represent trade execution date. Values match exactly."
    }}
  ],
  "unmapped_source": ["column_name"],
  "unmapped_target": ["column_name"]
}}

Match types: exact (identical values), fuzzy (string similarity), tolerance (numeric within range), semantic (different format, same meaning)"""

    response = await call_llm(prompt)
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned)
    except Exception:
        return {"mappings": [], "unmapped_source": [], "unmapped_target": []}


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


async def suggest_matching_rules(mappings: list[dict], sample_data: dict) -> list[dict]:
    """Use LLM to suggest matching rules for each mapped field pair."""
    prompt = f"""Based on these field mappings and sample data, suggest the best matching rule for each pair.

Mappings: {json.dumps(mappings, indent=2)}
Sample data (first 5 rows): {json.dumps(sample_data, indent=2)}

For each mapping, suggest:
- match_type: exact | levenshtein | jaro_winkler | numeric_tolerance | date_tolerance
- threshold: null for exact, 0-1 for string similarity, numeric value for tolerance

Respond ONLY with valid JSON array:
[
  {{
    "source_column": "Price",
    "target_column": "exec_px",
    "match_type": "numeric_tolerance",
    "threshold": 0.01,
    "reasoning": "Price fields may differ by rounding — 0.01 tolerance covers 2 decimal place differences."
  }}
]"""

    response = await call_llm(prompt)
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned)
    except Exception:
        return []
