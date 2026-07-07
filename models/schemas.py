import os
from pydantic import BaseModel
from typing import Any, Optional
from enum import Enum

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LLMProvider(str, Enum):
    anthropic = "anthropic"
    openai = "openai"
    ollama = "ollama"


def _default_provider() -> LLMProvider:
    v = os.getenv("LLM_PROVIDER", "openai").lower()
    return LLMProvider(v) if v in ("anthropic", "openai", "ollama") else LLMProvider.openai


def _default_model() -> str:
    return os.getenv("LLM_MODEL", "openai/gpt-4o-mini")


class LLMConfig(BaseModel):
    provider: LLMProvider = None          # type: ignore[assignment]
    model: str = None                     # type: ignore[assignment]
    api_key: Optional[str] = None
    base_url: Optional[str] = None        # for Ollama / OpenRouter / private cloud
    privacy_mode: str = "full"            # "full" | "headers_only"

    def model_post_init(self, __context: Any) -> None:
        if self.provider is None:
            object.__setattr__(self, "provider", _default_provider())
        if self.model is None:
            object.__setattr__(self, "model", _default_model())


class ColumnProfile(BaseModel):
    name: str
    sample_values: list[Any]
    dtype: str
    null_count: int
    unique_count: int
    inferred_label: Optional[str] = None
    inferred_type: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class FileProfile(BaseModel):
    file_id: str
    filename: str
    format: str
    row_count: int
    columns: list[ColumnProfile]


class FieldMapping(BaseModel):
    source_file: str
    source_column: str
    target_file: str
    target_column: str
    confidence: float
    reasoning: str
    match_type: str  # exact, fuzzy, tolerance
    tolerance: Optional[str] = None
    user_confirmed: bool = False
    user_overridden: bool = False


class MappingResult(BaseModel):
    mappings: list[FieldMapping]
    unmapped_source: list[str]
    unmapped_target: list[str]


class ReconciliationRule(BaseModel):
    source_column: str
    target_column: str
    match_type: str  # exact, levenshtein, jaro_winkler, numeric_tolerance
    threshold: Optional[float] = None


class ReconciliationConfig(BaseModel):
    file_a_id: str
    file_b_id: str
    rules: list[ReconciliationRule]
    key_columns: list[str]


class ReconciliationRecord(BaseModel):
    match_key: str
    status: str  # matched, break, unmatched_a, unmatched_b
    match_probability: Optional[float] = None
    source_data: dict
    target_data: dict
    break_reasons: list[str] = []
    ai_explanation: Optional[str] = None


class ReconciliationResult(BaseModel):
    total_source: int
    total_target: int
    matched: int
    breaks: int
    unmatched_source: int
    unmatched_target: int
    match_rate: float
    records: list[ReconciliationRecord]


class AuditEntry(BaseModel):
    timestamp: str
    action: str
    details: dict
    llm_provider: Optional[str] = None
    reasoning: Optional[str] = None
