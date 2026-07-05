"""
Probabilistic Record Matching — Fellegi-Sunter model (no external deps).

Each field pair is scored 0.0–1.0 based on its semantic type.
Field scores are combined into a composite match probability.
Records are greedily assigned (best score first, 1:1 constraint).

Designed for cases where no clean join key exists:
  - Nostro: bank reference ≠ internal reference
  - Invoice: vendor name spelling differs, PO reference formatted differently
  - Position: custodian ISIN vs PMS ticker (cross-referenced via lookup)

Usage:
    results = probabilistic_reconcile(df_a, df_b, field_config, threshold=0.65)

field_config = [
  {"source": "ref_no", "target": "reference", "type": "text",    "weight": 2.0},
  {"source": "amount",  "target": "amount",    "type": "numeric", "weight": 1.5},
  {"source": "date",    "target": "value_date","type": "date",    "weight": 1.0},
]
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd


# ── Field-level match probability ─────────────────────────────────────────────

def _score_field(val_a: Any, val_b: Any, field_type: str) -> float:
    """Return P(match | field values) in [0, 1]."""
    try:
        a = str(val_a).strip() if val_a is not None and not _is_na(val_a) else ""
        b = str(val_b).strip() if val_b is not None and not _is_na(val_b) else ""

        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0

        if field_type == "exact":
            return 1.0 if a.lower() == b.lower() else 0.0

        if field_type == "numeric":
            fa = _parse_num(a)
            fb = _parse_num(b)
            if fa is None or fb is None:
                return 0.0
            denom = max(abs(fa), abs(fb), 1e-9)
            rel_diff = abs(fa - fb) / denom
            # Smooth decay: 0% diff → 1.0, 1% diff → 0.9, 10% diff → ~0.0
            return max(0.0, 1.0 - rel_diff * 10)

        if field_type == "date":
            da = pd.to_datetime(a, errors="coerce", dayfirst=True)
            db = pd.to_datetime(b, errors="coerce", dayfirst=True)
            if pd.isna(da) or pd.isna(db):
                return 0.5
            delta = abs((da - db).days)
            return 1.0 if delta == 0 else max(0.0, 1.0 - delta / 5)

        if field_type == "text":
            # Jaro-Winkler approximation: SequenceMatcher on lowercased strings
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        if field_type == "lookup":
            # Abbreviation expansion: "B"→"Buy", "GS"→"Goldman Sachs" etc.
            return 1.0 if _lookup_normalize(a) == _lookup_normalize(b) else 0.0

        # Default: hybrid exact + fuzzy
        if a.lower() == b.lower():
            return 1.0
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    except Exception:
        return 0.5


def _is_na(v: Any) -> bool:
    try:
        import math
        return math.isnan(float(v))
    except Exception:
        return False


def _parse_num(s: str) -> float | None:
    try:
        return float(re.sub(r"[^\d.\-]", "", s))
    except Exception:
        return None


# Bidirectional lookup table for common financial abbreviations
_LOOKUP: dict[str, str] = {
    "b": "buy", "s": "sell", "buy": "buy", "sell": "sell",
    "gs": "goldman sachs", "goldman sachs": "goldman sachs",
    "ms": "morgan stanley", "morgan stanley": "morgan stanley",
    "jpm": "jpmorgan", "jpmorgan": "jpmorgan",
    "c": "credit", "d": "debit", "credit": "credit", "debit": "debit",
}


def _lookup_normalize(v: str) -> str:
    return _LOOKUP.get(v.strip().lower(), v.strip().lower())


# ── Composite scoring ──────────────────────────────────────────────────────────

def _score_pair(
    row_a: pd.Series,
    row_b: pd.Series,
    field_config: list[dict],
) -> tuple[float, dict[str, float]]:
    """Return (composite_score, {field: score}) for a record pair."""
    total_weight = 0.0
    weighted_sum = 0.0
    field_scores: dict[str, float] = {}

    for fc in field_config:
        src = fc.get("source", "")
        tgt = fc.get("target", src)
        ftype = fc.get("type", "exact")
        weight = float(fc.get("weight", 1.0))

        va = row_a.get(src) if src in row_a.index else None
        vb = row_b.get(tgt) if tgt in row_b.index else None

        score = _score_field(va, vb, ftype)
        field_scores[src] = round(score, 3)
        weighted_sum += score * weight
        total_weight += weight

    composite = weighted_sum / total_weight if total_weight > 0 else 0.0
    return round(composite, 3), field_scores


# ── Main reconciliation function ───────────────────────────────────────────────

def probabilistic_reconcile(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    field_config: list[dict],
    threshold: float = 0.65,
    blocking_field: str | None = None,
) -> dict:
    """
    Probabilistic reconciliation of df_a vs df_b.

    Returns the same shape as reconciliation.run_reconciliation():
    { total_source, total_target, matched, breaks, unmatched_source,
      unmatched_target, match_rate, records }

    Each record additionally carries 'field_scores' and 'match_probability'.
    """
    df_a = df_a.copy().reset_index(drop=True)
    df_b = df_b.copy().reset_index(drop=True)

    # Build candidate pairs (with optional blocking to cut O(n²))
    pairs: list[tuple[int, int]] = _build_pairs(df_a, df_b, blocking_field)

    # Score all pairs
    scored: list[tuple[float, int, int, dict[str, float]]] = []
    for ia, ib in pairs:
        comp, fscores = _score_pair(df_a.iloc[ia], df_b.iloc[ib], field_config)
        scored.append((comp, ia, ib, fscores))

    # Sort descending; greedy 1:1 assignment
    scored.sort(key=lambda x: x[0], reverse=True)

    matched_a: set[int] = set()
    matched_b: set[int] = set()
    records: list[dict] = []
    matched = 0
    breaks = 0

    for comp, ia, ib, fscores in scored:
        if ia in matched_a or ib in matched_b:
            continue
        if comp < threshold:
            break  # remaining scores are lower; skip

        matched_a.add(ia)
        matched_b.add(ib)

        src_data = _row_to_dict(df_a.iloc[ia])
        tgt_data = _row_to_dict(df_b.iloc[ib])

        # Break fields: any field scoring below 0.9
        break_reasons = [
            f"{f}: {src_data.get(f)} ≠ {tgt_data.get(fscores_target(f, field_config))} "
            f"(match={s:.0%})"
            for f, s in fscores.items()
            if s < 0.9
        ]

        status = "matched" if not break_reasons else "break"
        if status == "matched":
            matched += 1
        else:
            breaks += 1

        records.append({
            "match_key": f"prob_{ia}_{ib}",
            "status": status,
            "match_probability": comp,
            "source_data": src_data,
            "target_data": tgt_data,
            "break_reasons": break_reasons,
            "field_scores": fscores,
            "match_method": "probabilistic",
        })

    # Unmatched source
    unmatched_source = 0
    for ia in range(len(df_a)):
        if ia not in matched_a:
            unmatched_source += 1
            records.append({
                "match_key": f"src_{ia}",
                "status": "unmatched_source",
                "match_probability": 0.0,
                "source_data": _row_to_dict(df_a.iloc[ia]),
                "target_data": {},
                "break_reasons": ["No probabilistic match in target (threshold not met)"],
                "field_scores": {},
                "match_method": "probabilistic",
            })

    # Unmatched target
    unmatched_target = 0
    for ib in range(len(df_b)):
        if ib not in matched_b:
            unmatched_target += 1
            records.append({
                "match_key": f"tgt_{ib}",
                "status": "unmatched_target",
                "match_probability": 0.0,
                "source_data": {},
                "target_data": _row_to_dict(df_b.iloc[ib]),
                "break_reasons": ["No probabilistic match in source (threshold not met)"],
                "field_scores": {},
                "match_method": "probabilistic",
            })

    total = matched + breaks + unmatched_source + unmatched_target
    match_rate = round(matched / max(total, 1) * 100, 2)

    return {
        "total_source": len(df_a),
        "total_target": len(df_b),
        "matched": matched,
        "breaks": breaks,
        "unmatched_source": unmatched_source,
        "unmatched_target": unmatched_target,
        "match_rate": match_rate,
        "records": records,
        "match_method": "probabilistic",
        "threshold_used": threshold,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_pairs(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    blocking_field: str | None,
) -> list[tuple[int, int]]:
    """Return index pairs to compare. Blocking reduces O(n²) to O(n×k)."""
    if blocking_field and blocking_field in df_a.columns and blocking_field in df_b.columns:
        pairs = []
        b_groups: dict[str, list[int]] = {}
        for ib, rb in df_b.iterrows():
            key = str(rb[blocking_field]).strip().lower()
            b_groups.setdefault(key, []).append(ib)
        for ia, ra in df_a.iterrows():
            key = str(ra[blocking_field]).strip().lower()
            for ib in b_groups.get(key, []):
                pairs.append((ia, ib))
        # Fall back to full cross-product if blocking produced nothing
        if not pairs:
            return [(ia, ib) for ia in range(len(df_a)) for ib in range(len(df_b))]
        return pairs
    else:
        # Full cross-product — only safe for small files (< 1000 rows each)
        MAX = 500
        if len(df_a) > MAX or len(df_b) > MAX:
            # Take top rows as representative sample
            df_a = df_a.head(MAX)
            df_b = df_b.head(MAX)
        return [(ia, ib) for ia in range(len(df_a)) for ib in range(len(df_b))]


def _row_to_dict(row: pd.Series) -> dict:
    return {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in row.items()
    }


def fscores_target(source_field: str, field_config: list[dict]) -> str:
    for fc in field_config:
        if fc.get("source") == source_field:
            return fc.get("target", source_field)
    return source_field


# ── Auto-infer field config from column profiles ───────────────────────────────

def infer_field_config(mappings: list[dict]) -> list[dict]:
    """
    Convert the existing mapping format into field_config for probabilistic matching.
    mappings: [{"source_column": "...", "target_column": "...", "match_type": "...", ...}]
    """
    TYPE_MAP = {
        "exact":             "exact",
        "numeric_tolerance": "numeric",
        "date_tolerance":    "date",
        "levenshtein":       "text",
        "value_lookup":      "lookup",
        "fuzzy":             "text",
        "tolerance":         "numeric",
        "semantic":          "text",
    }
    config = []
    for m in mappings:
        config.append({
            "source": m.get("source_column", ""),
            "target": m.get("target_column", ""),
            "type":   TYPE_MAP.get(m.get("match_type", "exact"), "exact"),
            "weight": 2.0 if m.get("confidence", 0) > 0.9 else 1.0,
        })
    return config
