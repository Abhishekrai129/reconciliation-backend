import json
import sqlite3
from pathlib import Path
import pandas as pd
from typing import Any
from services.file_processor import get_dataframe
from services.data_profiler import score_field_match, confidence_band, hitl_required as _hitl_required

_DB_PATH = Path(__file__).parent.parent / "pipeline.db"


def _results_conn():
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reconciliation_results (
            run_id TEXT PRIMARY KEY,
            results_json TEXT NOT NULL,
            saved_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def store_run_results(run_id: str, results: dict):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    trimmed = {k: v for k, v in results.items() if k != "records"}
    trimmed["records"] = results.get("records", [])[:500]  # cap at 500 to keep DB lean
    with _results_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO reconciliation_results (run_id, results_json, saved_at) VALUES (?, ?, ?)",
            (run_id, json.dumps(trimmed), now),
        )
        conn.commit()


def get_run_results(run_id: str) -> dict | None:
    try:
        with _results_conn() as conn:
            row = conn.execute(
                "SELECT results_json FROM reconciliation_results WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row:
                return json.loads(row["results_json"])
    except Exception:
        pass
    return None


def run_reconciliation(
    file_a_id: str,
    file_b_id: str,
    rules: list[dict],
    key_columns: list[dict],  # [{"source": "Trade_ID", "target": "ref_no"}]
) -> dict:
    df_a = get_dataframe(file_a_id).copy()
    df_b = get_dataframe(file_b_id).copy()

    # Rename File B columns to match File A using rules
    rename_map = {r["target_column"]: r["source_column"] for r in rules}
    df_b = df_b.rename(columns=rename_map)

    # Build key for joining
    key_source = [k["source"] for k in key_columns]
    key_target = [rename_map.get(k["target"], k["target"]) for k in key_columns]

    df_a["_key"] = df_a[key_source].astype(str).apply(lambda x: "|".join(x), axis=1)
    df_b["_key"] = df_b[key_source].astype(str).apply(lambda x: "|".join(x), axis=1)

    records = []

    # Full outer join on key
    merged = pd.merge(
        df_a.add_suffix("_A"),
        df_b.add_suffix("_B"),
        left_on="_key_A",
        right_on="_key_B",
        how="outer",
        indicator=True,
    )

    matched = 0
    breaks = 0
    unmatched_a = 0
    unmatched_b = 0

    for _, row in merged.iterrows():
        merge_flag = row["_merge"]

        if merge_flag == "left_only":
            unmatched_a += 1
            records.append({
                "match_key": row.get("_key_A", ""),
                "status": "unmatched_source",
                "match_probability": 0.0,
                "source_data": {c.replace("_A", ""): row[c] for c in merged.columns if c.endswith("_A")},
                "target_data": {},
                "break_reasons": ["No matching record in target file"],
            })
            continue

        if merge_flag == "right_only":
            unmatched_b += 1
            records.append({
                "match_key": row.get("_key_B", ""),
                "status": "unmatched_target",
                "match_probability": 0.0,
                "source_data": {},
                "target_data": {c.replace("_B", ""): row[c] for c in merged.columns if c.endswith("_B")},
                "break_reasons": ["No matching record in source file"],
            })
            continue

        # Both sides present — check field-level rules
        source_data = {c.replace("_A", ""): row[c] for c in merged.columns if c.endswith("_A")}
        target_data = {c.replace("_B", ""): row[c] for c in merged.columns if c.endswith("_B")}
        break_reasons = []

        field_scores: dict[str, float] = {}
        for rule in rules:
            col = rule["source_column"]
            col_a = f"{col}_A"
            col_b = f"{col}_B"

            if col_a not in merged.columns or col_b not in merged.columns:
                continue
            val_a = row.get(col_a)
            val_b = row.get(col_b)

            if pd.isna(val_a) and pd.isna(val_b):
                field_scores[col] = 1.0
                continue

            match_type = rule.get("match_type", "exact")
            threshold = rule.get("threshold")

            # Real confidence score per field (0.0–1.0)
            score = score_field_match(val_a, val_b, match_type, threshold)
            field_scores[col] = score

            if score < 0.5:
                break_reasons.append(f"{col}: {val_a} ≠ {val_b}  [{score:.0%} confidence]")

        # Weighted overall confidence
        if field_scores:
            weights = {r["source_column"]: float(r.get("weight", 1.0)) for r in rules}
            total_w = sum(weights.get(c, 1.0) for c in field_scores)
            overall = sum(field_scores[c] * weights.get(c, 1.0) for c in field_scores) / max(total_w, 1e-9)
        else:
            overall = 1.0

        has_break = bool(break_reasons)
        if has_break:
            breaks += 1
            status = "break"
        else:
            matched += 1
            status = "matched"

        records.append({
            "match_key": row.get("_key_A", row.get("_key_B", "")),
            "status": status,
            "match_probability": round(overall, 4),
            "confidence_band": confidence_band(overall),
            "hitl_required": _hitl_required(overall, has_break),
            "field_scores": {k: round(v, 4) for k, v in field_scores.items()},
            "source_data": source_data,
            "target_data": target_data,
            "break_reasons": break_reasons,
        })

    total_a = len(df_a)
    total_b = len(df_b)
    total_compared = matched + breaks + unmatched_a + unmatched_b
    match_rate = round(matched / max(total_compared, 1) * 100, 2)

    return {
        "total_source": total_a,
        "total_target": total_b,
        "matched": matched,
        "breaks": breaks,
        "unmatched_source": unmatched_a,
        "unmatched_target": unmatched_b,
        "match_rate": match_rate,
        "records": records,
    }


def _compare_values(val_a: Any, val_b: Any, match_type: str, threshold: Any) -> bool:
    try:
        if match_type == "exact":
            return str(val_a).strip().lower() == str(val_b).strip().lower()

        elif match_type == "numeric_tolerance":
            # Strip currency symbols, commas before parsing
            def _to_float(v):
                return float(str(v).replace(",", "").replace("$", "").strip())
            a = _to_float(val_a)
            b = _to_float(val_b)
            tol = float(threshold) if threshold else 0.01
            # Use epsilon to handle floating-point representation errors
            return abs(a - b) <= tol + 1e-9

        elif match_type == "value_lookup":
            # threshold is a dict mapping abbreviations → full names (e.g. {"B":"Buy"})
            # Merge with the dictionary's richer value_map at runtime so all known
            # aliases (L→Buy, P→Buy, 1→Buy, etc.) are always covered.
            lookup: dict = threshold if isinstance(threshold, dict) else {}
            try:
                from services.dictionary_service import lookup_field
                for raw in [str(val_a), str(val_b)]:
                    entry = lookup_field(raw) or lookup_field(str(val_a))
                    if entry:
                        import json as _json
                        vm = _json.loads(entry.get("value_map", "{}") if isinstance(entry.get("value_map"), str) else "{}")
                        for k, v in vm.items():
                            if k not in lookup:
                                lookup[k] = v
                        break
            except Exception:
                pass
            forward = {k.strip().lower(): v.strip().lower() for k, v in lookup.items()}

            def _normalize(v: str) -> str:
                s = str(v).strip().lower()
                return forward.get(s, s)

            return _normalize(str(val_a)) == _normalize(str(val_b))

        elif match_type == "levenshtein":
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, str(val_a), str(val_b)).ratio()
            return ratio >= (float(threshold) if threshold else 0.8)

        elif match_type == "date_tolerance":
            import pandas as pd
            da = pd.to_datetime(val_a)
            db = pd.to_datetime(val_b)
            delta = abs((da - db).days)
            return delta <= (int(threshold) if threshold else 0)

        return str(val_a).strip() == str(val_b).strip()
    except Exception:
        return str(val_a) == str(val_b)
