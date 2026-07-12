"""
Data Intelligence Profiler — scans both files before reconciliation and surfaces:

  • Schema problems   (missing columns, mismatched names, no join key)
  • Type problems     (mixed types, date format inconsistency)
  • Value problems    (nulls, duplicates, outliers, enum inconsistency, whitespace)
  • Cross-file issues (row count mismatch, zero key overlap, scale mismatch, date range gap)
  • Parsing challenges (format detected, encoding problems)

Returns a structured report consumed by the UI confidence panel and HITL gateway.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd
import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def _issue(severity: str, category: str, message: str, detail: str = "") -> dict:
    return {"severity": severity, "category": category, "message": message, "detail": detail}


DATE_PATTERNS = [
    ("ISO",       re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    ("DD/MM/YYYY",re.compile(r"^\d{2}/\d{2}/\d{4}$")),
    ("MM/DD/YYYY",re.compile(r"^\d{2}/\d{2}/\d{4}$")),
    ("DD-MM-YYYY",re.compile(r"^\d{2}-\d{2}-\d{4}$")),
    ("DDMONYYYY",  re.compile(r"^\d{2}[A-Z]{3}\d{4}$")),
    ("YYYYMMDD",  re.compile(r"^\d{8}$")),
]

KNOWN_ENUMS = {
    # side codes
    frozenset({"b", "buy"}), frozenset({"s", "sell"}),
    frozenset({"b", "s", "buy", "sell"}),
    # debit/credit
    frozenset({"d", "debit"}), frozenset({"c", "credit"}),
    frozenset({"d", "c", "debit", "credit"}),
    # long/short
    frozenset({"l", "long"}), frozenset({"s", "short"}),
}

CURRENCY_SYMS = re.compile(r"^[\$€£¥₹]")


# ── Single-file profiling ──────────────────────────────────────────────────────

def _profile_single(df: pd.DataFrame, label: str) -> list[dict]:
    issues: list[dict] = []
    n = len(df)

    if n == 0:
        return [_issue("error", "Empty File", f"{label} has no data rows",
                       "Cannot reconcile an empty file")]

    col_names = list(df.columns)

    # Check for completely unnamed columns (Excel artefact)
    unnamed = [c for c in col_names if str(c).startswith("Unnamed:")]
    if unnamed:
        issues.append(_issue("warning", "Schema", f"{len(unnamed)} unnamed column(s) found in {label}",
                             f"Likely a header row parsing issue: {unnamed[:3]}"))

    for col in col_names:
        series = df[col]
        non_null = series.dropna()
        null_rate = 1 - len(non_null) / n if n else 0

        # ── Null checks ─────────────────────────────────────────────────────
        if null_rate == 1.0:
            issues.append(_issue("error", "Empty Column",
                                 f"[{label}] '{col}' is entirely empty",
                                 "All rows are null — column cannot participate in matching"))
        elif null_rate > 0.5:
            issues.append(_issue("warning", "High Nulls",
                                 f"[{label}] '{col}' is {null_rate:.0%} empty",
                                 f"{int(null_rate*n)} of {n} rows are null"))
        elif null_rate > 0.1:
            issues.append(_issue("info", "Nulls",
                                 f"[{label}] '{col}' has {null_rate:.0%} missing values",
                                 f"{int(null_rate*n)} rows"))

        if len(non_null) == 0:
            continue

        str_vals = non_null.astype(str).str.strip()

        # ── Duplicate key check ─────────────────────────────────────────────
        # isin/cusip are securities codes (repeat across many trades) — not trade-level unique keys
        key_hints = {"_id", "ref", "key", "account", "order", "number", "num",
                     "reference", "transaction", "trxn", "uetr",
                     "trade_id", "tradeid", "trade_ref", "traderef"}
        col_lower = col.lower().replace(" ", "_")
        if any(kh in col_lower for kh in key_hints):
            dup = non_null.duplicated().sum()
            if dup > 0:
                issues.append(_issue("error", "Duplicate Keys",
                                     f"[{label}] '{col}' has {dup} duplicate value(s)",
                                     "Duplicate join keys produce incorrect M:N matches"))

        # ── Date format inconsistency ───────────────────────────────────────
        sample = str_vals.head(100)
        found_formats: set[str] = set()
        for v in sample:
            for name, pat in DATE_PATTERNS:
                if pat.match(v.upper()):
                    found_formats.add(name)
                    break
        if len(found_formats) > 1:
            issues.append(_issue("warning", "Date Format Inconsistency",
                                 f"[{label}] '{col}' has mixed date formats",
                                 f"Found: {', '.join(sorted(found_formats))} — normalised at compare time"))

        # ── Enum inconsistency (abbreviation + full form) ───────────────────
        if non_null.dtype == object:
            unique_lower = {str(v).strip().lower() for v in non_null.unique()}
            unique_orig  = {str(v).strip()       for v in non_null.unique()}

            if len(unique_orig) != len(unique_lower):
                issues.append(_issue("info", "Case Variation",
                                     f"[{label}] '{col}' has case inconsistency",
                                     f"e.g. {sorted(list(unique_orig))[:4]}"))

            short = [v for v in unique_orig if len(v) <= 3]
            long_ = [v for v in unique_orig if len(v) >  3]
            if short and long_:
                issues.append(_issue("warning", "Value Encoding Mismatch",
                                     f"[{label}] '{col}' mixes abbreviations and full names",
                                     f"Short: {short[:3]}  Full: {long_[:3]} — use lookup table to normalise"))

            # Encoding / mojibake
            has_bad = any(
                unicodedata.category(ch) in ("Cs", "Co", "Cn")
                for v in non_null.head(50).astype(str)
                for ch in v
            )
            if has_bad:
                issues.append(_issue("warning", "Encoding Issue",
                                     f"[{label}] '{col}' contains non-printable / mojibake characters",
                                     "Likely encoding mismatch (UTF-8 vs Windows-1252)"))

            # Leading/trailing whitespace
            has_ws = (str_vals.head(50) != non_null.astype(str).head(50)).any()
            if has_ws:
                issues.append(_issue("info", "Whitespace",
                                     f"[{label}] '{col}' has leading/trailing spaces",
                                     "Auto-stripped at comparison time"))

            # Currency symbol inconsistency
            has_sym  = str_vals.str.match(CURRENCY_SYMS).any()
            has_bare = (~str_vals.str.match(CURRENCY_SYMS)).any()
            if has_sym and has_bare and pd.to_numeric(str_vals.str.replace(r"[^\d.]", "", regex=True), errors="coerce").notna().mean() > 0.7:
                issues.append(_issue("warning", "Currency Symbol",
                                     f"[{label}] '{col}' mixes currency symbols with plain numbers",
                                     "e.g. '$1,250.00' vs '1250' — stripped before numeric compare"))

        # ── Mixed type detection ────────────────────────────────────────────
        if non_null.dtype == object:
            numeric_frac = pd.to_numeric(non_null, errors="coerce").notna().mean()
            if 0.1 < numeric_frac < 0.9:
                issues.append(_issue("warning", "Mixed Types",
                                     f"[{label}] '{col}' is {numeric_frac:.0%} numeric, {1-numeric_frac:.0%} text",
                                     "Column has mixed data types — may indicate parsing error"))

        # ── Outlier detection for numeric columns ───────────────────────────
        if pd.api.types.is_numeric_dtype(series):
            try:
                clean = pd.to_numeric(non_null, errors="coerce").dropna()
                if len(clean) > 10:
                    mean, std = float(clean.mean()), float(clean.std())
                    if std > 0:
                        outliers = int(((clean - mean).abs() > 4 * std).sum())
                        if outliers > 0:
                            issues.append(_issue("info", "Outliers",
                                                 f"[{label}] '{col}' has {outliers} statistical outlier(s)",
                                                 f"Values > 4σ from mean ({mean:,.2f} ± {std:,.2f})"))
            except Exception:
                pass

    return issues


# ── Cross-file profiling ───────────────────────────────────────────────────────

def _profile_cross(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    key_col_a: str | None = None,
    key_col_b: str | None = None,
) -> list[dict]:
    issues: list[dict] = []
    na, nb = len(df_a), len(df_b)

    # ── Row count mismatch ─────────────────────────────────────────────────
    if max(na, nb) > 0:
        ratio = abs(na - nb) / max(na, nb)
        if ratio > 0.5:
            issues.append(_issue("warning", "Row Count Mismatch",
                                 f"Files differ significantly: {na:,} vs {nb:,} rows",
                                 f"{ratio:.0%} size difference — expect many unmatched records"))
        elif ratio > 0.2:
            issues.append(_issue("info", "Row Count Difference",
                                 f"Files have different sizes: {na:,} vs {nb:,} rows",
                                 f"{ratio:.0%} difference"))

    # ── Key overlap ────────────────────────────────────────────────────────
    if key_col_a and key_col_b:
        if key_col_a in df_a.columns and key_col_b in df_b.columns:
            set_a = set(df_a[key_col_a].dropna().astype(str))
            set_b = set(df_b[key_col_b].dropna().astype(str))
            if set_a and set_b:
                overlap = len(set_a & set_b) / max(len(set_a), len(set_b))
                if overlap == 0.0:
                    issues.append(_issue("error", "Zero Key Overlap",
                                         f"Keys '{key_col_a}' ↔ '{key_col_b}' share 0% overlap",
                                         "Reconciliation will produce only unmatched records — verify correct files are loaded"))
                elif overlap < 0.3:
                    issues.append(_issue("warning", "Low Key Overlap",
                                         f"Only {overlap:.0%} of keys match between files",
                                         f"{int(overlap * max(len(set_a), len(set_b)))} common key(s)"))

    # ── Schema drift (columns only in one file) ────────────────────────────
    cols_a = {c.lower().strip() for c in df_a.columns}
    cols_b = {c.lower().strip() for c in df_b.columns}
    only_a = sorted(cols_a - cols_b)
    only_b = sorted(cols_b - cols_a)
    if only_a:
        issues.append(_issue("info", "Schema Drift",
                             f"{len(only_a)} column(s) only in source file",
                             ", ".join(only_a)))
    if only_b:
        issues.append(_issue("info", "Schema Drift",
                             f"{len(only_b)} column(s) only in target file",
                             ", ".join(only_b)))

    # ── Numeric scale mismatch ─────────────────────────────────────────────
    num_a = df_a.select_dtypes(include="number")
    num_b = df_b.select_dtypes(include="number")
    for col_a in num_a.columns:
        for col_b in num_b.columns:
            a_low, b_low = col_a.lower(), col_b.lower()
            if not (a_low in b_low or b_low in a_low):
                continue
            med_a = float(num_a[col_a].median())
            med_b = float(num_b[col_b].median())
            if med_a > 0 and med_b > 0:
                ratio = max(med_a, med_b) / min(med_a, med_b)
                if ratio > 100:
                    issues.append(_issue("error", "Numeric Scale Mismatch",
                                         f"'{col_a}' vs '{col_b}' differ by {ratio:,.0f}×",
                                         f"Source median: {med_a:,.0f}  Target median: {med_b:,.0f} — one file may be in thousands"))

    # ── Date range gap ─────────────────────────────────────────────────────
    for col_a in df_a.columns:
        for col_b in df_b.columns:
            a_low, b_low = col_a.lower(), col_b.lower()
            if not (a_low in b_low or b_low in a_low):
                continue
            try:
                dates_a = pd.to_datetime(df_a[col_a], errors="coerce", dayfirst=True).dropna()
                dates_b = pd.to_datetime(df_b[col_b], errors="coerce", dayfirst=True).dropna()
                if len(dates_a) < 2 or len(dates_b) < 2:
                    continue
                if dates_a.min() > dates_b.max() or dates_b.min() > dates_a.max():
                    issues.append(_issue("error", "Date Range Gap",
                                         f"Date columns '{col_a}' / '{col_b}' have no overlapping period",
                                         f"Source: {dates_a.min().date()} – {dates_a.max().date()} | "
                                         f"Target: {dates_b.min().date()} – {dates_b.max().date()}"))
            except Exception:
                pass

    return issues


# ── Overall quality score & HITL triggers ─────────────────────────────────────

def _quality_score(issues: list[dict], avg_mapping_confidence: float = 0.0) -> int:
    # Schema Drift is expected in cross-system reconciliation (OMS vs broker FIX feed
    # will never share column names). Info items are observations, not quality problems —
    # deducting for them penalises normal data. Only errors and warnings cost points.
    deductions = {"error": 20, "warning": 8, "info": 0}
    base = 100 - sum(deductions.get(i["severity"], 0) for i in issues)
    # Bonus: average semantic similarity across AI-mapped field pairs (0–10 pts).
    # High-confidence mapping (e.g. 0.90 avg) → full +10. No mappings → 0.
    if avg_mapping_confidence > 0:
        base += round(avg_mapping_confidence * 10)
    return max(0, min(100, base))


def _hitl_triggers(issues: list[dict], quality_score: int) -> list[str]:
    triggers: list[str] = []
    cats = {i["category"] for i in issues if i["severity"] == "error"}

    if quality_score < 50:
        triggers.append("Data quality score is critically low — human must review before running reconciliation")
    if "Duplicate Keys" in cats:
        triggers.append("Duplicate join keys — matching results will be incorrect without human de-duplication")
    if "Zero Key Overlap" in cats:
        triggers.append("Zero key overlap — human must verify the correct pair of files was uploaded")
    if "Numeric Scale Mismatch" in cats:
        triggers.append("Numeric scale mismatch — human must confirm unit convention (actuals vs thousands)")
    if "Date Range Gap" in cats:
        triggers.append("Date ranges don't overlap — human must confirm the correct reporting period")
    if "Empty Column" in cats:
        triggers.append("One or more columns are entirely empty — human must investigate data extraction")

    return triggers


def _recommendations(issues: list[dict]) -> list[str]:
    recs: list[str] = []
    cats = {i["category"] for i in issues}

    if "Date Format Inconsistency" in cats:
        recs.append("Normalise all dates to ISO 8601 (YYYY-MM-DD) — AI does this automatically at compare time")
    if "Value Encoding Mismatch" in cats:
        recs.append("Use value lookup table to map abbreviations (B→Buy, S→Sell, GS→Goldman Sachs)")
    if "Case Variation" in cats:
        recs.append("Apply case-insensitive matching for all text fields — AI enables this by default")
    if "Numeric Scale Mismatch" in cats:
        recs.append("Apply 1,000× scale factor to bring numeric fields to the same unit before comparing")
    if "Whitespace" in cats:
        recs.append("Strip leading/trailing spaces from all fields — AI does this automatically")
    if "Currency Symbol" in cats:
        recs.append("Strip currency symbols ($, £, €) before numeric comparison — AI handles this")
    if "Mixed Types" in cats:
        recs.append("Cast mixed-type columns explicitly — AI uses numeric tolerance when one side is numeric")
    if "High Nulls" in cats or "Nulls" in cats:
        recs.append("Consider treating null values as a break rather than a match — configure in tolerance rules")

    return recs


# ── Public API ─────────────────────────────────────────────────────────────────

def profile_files(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    key_col_a: str | None = None,
    key_col_b: str | None = None,
    format_a: str = "unknown",
    format_b: str = "unknown",
) -> dict:
    """Run the full intelligence profiler on both dataframes.

    Returns a structured report with quality_score (0-100), issue lists,
    HITL triggers, and human-readable recommendations.
    """
    issues_a = _profile_single(df_a, "Source")
    issues_b = _profile_single(df_b, "Target")
    cross    = _profile_cross(df_a, df_b, key_col_a, key_col_b)

    all_issues = issues_a + issues_b + cross
    quality_score = _quality_score(all_issues)

    # Parsing challenge summary — what the AI had to handle to even read the files
    parsing_challenges: list[str] = []
    if format_a in ("SWIFT MT940", "Fixed-Width/CSV or SWIFT MT940"):
        parsing_challenges.append(f"Source: SWIFT MT940 tag format parsed (:61: / :86: transaction blocks)")
    if format_b in ("SWIFT MT940", "Fixed-Width/CSV or SWIFT MT940"):
        parsing_challenges.append(f"Target: SWIFT MT940 tag format parsed")
    if format_a == "PDF" or format_b == "PDF":
        parsing_challenges.append("PDF: GPT-4o vision used to extract structured fields from unstructured document")

    # Schema mismatch summary for UI
    exact_col_matches = len(set(c.lower() for c in df_a.columns) & set(c.lower() for c in df_b.columns))
    total_a_cols = len(df_a.columns)
    total_b_cols = len(df_b.columns)
    schema_match_pct = round(exact_col_matches / max(total_a_cols, total_b_cols, 1) * 100)

    return {
        "quality_score":     quality_score,
        "total_issues":      len(all_issues),
        "error_count":       sum(1 for i in all_issues if i["severity"] == "error"),
        "warning_count":     sum(1 for i in all_issues if i["severity"] == "warning"),
        "info_count":        sum(1 for i in all_issues if i["severity"] == "info"),
        "file_a_issues":     issues_a,
        "file_b_issues":     issues_b,
        "cross_file_issues": cross,
        "hitl_triggers":     _hitl_triggers(all_issues, quality_score),
        "recommendations":   _recommendations(all_issues),
        "parsing_challenges":parsing_challenges,
        "schema_overlap": {
            "exact_col_matches": exact_col_matches,
            "source_cols":       total_a_cols,
            "target_cols":       total_b_cols,
            "match_pct":         schema_match_pct,
        },
        "row_counts": {
            "source": len(df_a),
            "target": len(df_b),
        },
    }


# ── Record-level confidence scoring ───────────────────────────────────────────

def score_field_match(val_a: Any, val_b: Any, match_type: str, threshold: Any = None) -> float:
    """Return a 0.0–1.0 confidence score for a single field comparison.

    Unlike the boolean _compare_values, this returns a gradient:
      1.00  — exact match
      0.85–0.99 — within tolerance (closer = higher score)
      0.60–0.84 — fuzzy / lookup match
      0.00  — clear mismatch
    """
    try:
        if pd.isna(val_a) and pd.isna(val_b):
            return 1.0
        if pd.isna(val_a) or pd.isna(val_b):
            return 0.0

        a = str(val_a).strip()
        b = str(val_b).strip()

        if match_type == "exact":
            return 1.0 if a.lower() == b.lower() else 0.0

        if match_type == "numeric_tolerance":
            def _to_f(v: str) -> float:
                return float(re.sub(r"[^\d.\-]", "", v))
            fa, fb = _to_f(a), _to_f(b)
            tol = float(threshold) if threshold else 0.01
            diff = abs(fa - fb)
            if diff <= 1e-9:
                return 1.0
            if diff <= tol:
                return 0.85 + 0.15 * (1 - diff / tol)
            return max(0.0, 0.85 - (diff - tol) / (tol + 1e-9) * 0.5)

        if match_type in ("levenshtein", "fuzzy", "jaro_winkler", "similarity"):
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
            min_ratio = float(threshold) if isinstance(threshold, (int, float)) else 0.7
            if ratio >= 1.0:
                return 1.0
            if ratio >= min_ratio:
                return 0.60 + 0.35 * ((ratio - min_ratio) / (1.0 - min_ratio + 1e-9))
            # Partial match — still give partial credit not 0
            return max(0.0, ratio * 0.5)

        if match_type == "value_lookup":
            lookup: dict = threshold if isinstance(threshold, dict) else {}
            fwd = {k.strip().lower(): v.strip().lower() for k, v in lookup.items()}
            def _norm(v: str) -> str:
                s = v.strip().lower()
                return fwd.get(s, s)
            return 1.0 if _norm(a) == _norm(b) else 0.0

        if match_type == "date_tolerance":
            da = pd.to_datetime(a, errors="coerce", dayfirst=True)
            db = pd.to_datetime(b, errors="coerce", dayfirst=True)
            if pd.isna(da) or pd.isna(db):
                return 0.0
            days = abs((da - db).days)
            max_days = int(threshold) if threshold else 0
            if days == 0:
                return 1.0
            if days <= max_days:
                return 0.85 + 0.15 * (1 - days / (max_days + 1))
            return max(0.0, 0.7 - days * 0.1)

        # Fallback: exact string
        return 1.0 if a == b else 0.0

    except Exception:
        return 0.0 if str(val_a) != str(val_b) else 1.0


def confidence_band(score: float) -> str:
    """Classify a 0-1 confidence score into a human-readable band."""
    if score >= 0.90:
        return "high"
    if score >= 0.70:
        return "medium"
    return "low"


def hitl_required(score: float, has_break: bool) -> bool:
    """Determine if a human must review this record."""
    return score < 0.70 or (has_break and score < 0.85)
