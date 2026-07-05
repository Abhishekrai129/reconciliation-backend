import pandas as pd
from typing import Any
from services.file_processor import get_dataframe

# In-memory results store keyed by run_id
_results_store: dict[str, dict] = {}

def store_run_results(run_id: str, results: dict):
    _results_store[run_id] = results

def get_run_results(run_id: str) -> dict | None:
    return _results_store.get(run_id)


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

        for rule in rules:
            col = rule["source_column"]
            col_a = f"{col}_A"
            col_b = f"{col}_B"

            if col_a not in merged.columns or col_b not in merged.columns:
                continue
            val_a = row.get(col_a)
            val_b = row.get(col_b)

            if pd.isna(val_a) and pd.isna(val_b):
                continue

            match_type = rule.get("match_type", "exact")
            threshold = rule.get("threshold")

            is_match = _compare_values(val_a, val_b, match_type, threshold)
            if not is_match:
                break_reasons.append(
                    f"{col}: {val_a} ≠ {val_b}"
                )

        if break_reasons:
            breaks += 1
            status = "break"
        else:
            matched += 1
            status = "matched"

        records.append({
            "match_key": row.get("_key_A", row.get("_key_B", "")),
            "status": status,
            "match_probability": 1.0 if status == "matched" else 0.5,
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
            # threshold is a dict mapping abbreviations → full names (e.g. {"B":"Buy","GS":"Goldman Sachs"})
            # Normalize both sides to the full-name canonical form
            lookup: dict = threshold if isinstance(threshold, dict) else {}
            forward = {k.strip().lower(): v.strip().lower() for k, v in lookup.items()}

            def _normalize(v: str) -> str:
                s = str(v).strip().lower()
                return forward.get(s, s)  # abbrev → full name; full name stays as-is

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
