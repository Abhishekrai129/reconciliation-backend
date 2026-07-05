import io
import json
import re
import uuid
from pathlib import Path
from typing import Any
import pandas as pd


SUPPORTED_FORMATS = {
    ".csv": "CSV",
    ".tsv": "TSV",
    ".xlsx": "Excel",
    ".xls": "Excel",
    ".json": "JSON",
    ".xml": "XML",
    ".parquet": "Parquet",
    ".txt": "Fixed-Width/CSV or SWIFT MT940",
}


def detect_format(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return SUPPORTED_FORMATS.get(ext, "Unknown")


def _is_swift_mt940(content: bytes) -> bool:
    text = content.decode("utf-8", errors="replace").strip()
    return text.startswith(":20:") or "\n:61:" in text


def _is_bloomberg_email(content: bytes) -> bool:
    text = content.decode("utf-8", errors="replace")
    return "bloomberg" in text.lower() and "ISIN:" in text and "EX-DATE:" in text


def _is_pdf_extraction(content: bytes) -> bool:
    text = content.decode("utf-8", errors="replace")
    return "PDF EXTRACTION RESULT" in text and "TOTAL DUE:" in text


def parse_bloomberg_email(content: bytes) -> pd.DataFrame:
    """Parse Bloomberg corporate action email announcements into a DataFrame.

    Each announcement block (separated by ===) becomes one row.
    Key fields: isin, event_type, ex_date, record_date, amount_per_share, currency
    """
    text = content.decode("utf-8", errors="replace")
    blocks = re.split(r"={4,}", text)
    rows = []
    for block in blocks:
        if "ISIN:" not in block:
            continue

        def _get(label: str) -> str:
            m = re.search(rf"^{label}:\s*(.+)$", block, re.MULTILINE | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        isin = _get("ISIN")
        if not isin:
            continue

        # Normalise ex_date: "14-FEB-2024" → "2024-02-14"
        raw_ex = _get("EX-DATE")
        try:
            ex_date = pd.to_datetime(raw_ex, dayfirst=True).strftime("%Y-%m-%d")
        except Exception:
            ex_date = raw_ex

        raw_record = _get("RECORD DATE")
        try:
            record_date = pd.to_datetime(raw_record, dayfirst=True).strftime("%Y-%m-%d")
        except Exception:
            record_date = raw_record

        raw_pay = _get("PAY DATE")
        try:
            pay_date = pd.to_datetime(raw_pay, dayfirst=True).strftime("%Y-%m-%d")
        except Exception:
            pay_date = raw_pay

        # Amount: "USD 0.5023" or "0.00"
        raw_amt = _get("AMOUNT PER SHARE")
        amt_match = re.search(r"[\d.]+", raw_amt)
        amount = float(amt_match.group()) if amt_match else 0.0

        rows.append({
            "isin":             isin,
            "ticker":           _get("TICKER").split()[0] if _get("TICKER") else "",
            "event_type":       _get("ACTION TYPE"),
            "ex_date":          ex_date,
            "record_date":      record_date,
            "pay_date":         pay_date,
            "amount_per_share": amount,
            "currency":         _get("CURRENCY") or "USD",
            "source":           "bloomberg_email",
        })

    if not rows:
        raise ValueError("No Bloomberg announcement blocks found in email file")
    return pd.DataFrame(rows)


def parse_pdf_invoices(content: bytes) -> pd.DataFrame:
    """Parse pdfplumber-extracted invoice text into a DataFrame.

    Each === PDF EXTRACTION === block becomes one row.
    Key fields: po_reference, vendor_name, amount_total, invoice_date
    """
    text = content.decode("utf-8", errors="replace")
    blocks = re.split(r"={4,}", text)
    rows = []

    for block in blocks:
        if "TOTAL DUE:" not in block and "PO NUMBER:" not in block:
            continue

        def _get(label: str) -> str:
            m = re.search(rf"^{label}:\s*(.+)$", block, re.MULTILINE | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        po_ref = _get("PO NUMBER")
        if not po_ref:
            continue

        vendor = _get("VENDOR")
        if not vendor:
            vendor = _get("ISSUER")

        # Amount: "$89,375.00" or "89375.00"
        raw_amt = _get("TOTAL DUE")
        amt_clean = re.sub(r"[^\d.]", "", raw_amt)
        try:
            amount = float(amt_clean)
        except Exception:
            amount = 0.0

        # Date normalisation
        raw_date = _get("INVOICE DATE")
        try:
            invoice_date = pd.to_datetime(raw_date).strftime("%Y-%m-%d")
        except Exception:
            invoice_date = raw_date

        rows.append({
            "po_reference":  po_ref,
            "vendor_name":   vendor,
            "amount_total":  amount,
            "invoice_date":  invoice_date,
            "payment_terms": _get("PAYMENT TERMS"),
            "invoice_number":_get("INVOICE NUMBER"),
            "source":        "pdf_extracted",
        })

    if not rows:
        raise ValueError("No invoice blocks found in PDF extraction file")
    return pd.DataFrame(rows)


def _parse_amount_mt940(raw: str) -> float:
    """Convert MT940 European amount '15000,00' → 15000.0"""
    return float(raw.replace(",", "."))


def _parse_date_mt940(yymmdd: str) -> str:
    """Convert YYMMDD → YYYY-MM-DD"""
    if len(yymmdd) == 6:
        return f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"
    return yymmdd


def parse_swift_mt940(content: bytes) -> pd.DataFrame:
    """Parse a SWIFT MT940 bank statement into a DataFrame.

    Each :61: transaction line becomes one row.
    The following :86: narrative is attached to that row.
    """
    text = content.decode("utf-8", errors="replace")
    rows = []

    # Split into lines, strip CRs
    lines = [l.rstrip() for l in text.splitlines()]

    # Collect :61: / :86: pairs
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(":61:"):
            raw61 = line[4:]  # strip ":61:"
            # Format: YYMMDD[YYMMDD][D|C|RD|RC]amount,centNTRF[ref[//bank_ref]]
            m = re.match(
                r"(\d{6})(\d{4})?"     # value_date + optional entry_date
                r"([DC](?:[RC])?)"     # D/C flag (D, C, RD, RC)
                r"([0-9,]+)"           # amount with comma decimal
                r"([A-Z]{4})"         # transaction type (NTRF, NCHK, etc.)
                r"([^/\r\n]*)"        # customer reference
                r"(?://(.*))?$",       # optional bank reference
                raw61,
            )
            if m:
                value_date = _parse_date_mt940(m.group(1))
                dc_raw = m.group(3)
                dr_cr = "C" if dc_raw.startswith("C") else "D"
                amount = _parse_amount_mt940(m.group(4))
                ref = m.group(6).strip() if m.group(6) else ""
                bank_ref = m.group(7).strip() if m.group(7) else ""

                # Look ahead for :86: narrative
                narrative = ""
                if i + 1 < len(lines) and lines[i + 1].startswith(":86:"):
                    narrative = lines[i + 1][4:].strip()

                rows.append({
                    "reference": ref,
                    "value_date": value_date,
                    "dr_cr": dr_cr,
                    "amount": amount,
                    "narrative": narrative,
                    "bank_reference": bank_ref,
                })
        i += 1

    if not rows:
        raise ValueError("No :61: transaction lines found in MT940 file")

    return pd.DataFrame(rows)


def read_file(content: bytes, filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()

    if ext == ".txt" and _is_swift_mt940(content):
        return parse_swift_mt940(content)

    if ext == ".txt" and _is_bloomberg_email(content):
        return parse_bloomberg_email(content)

    if ext == ".txt" and _is_pdf_extraction(content):
        return parse_pdf_invoices(content)

    if ext in (".csv", ".txt"):
        # Try comma first, then pipe, then tab
        for sep in [",", "|", "\t", ";"]:
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep, encoding="utf-8-sig")
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
        return pd.read_csv(io.BytesIO(content))

    elif ext == ".tsv":
        return pd.read_csv(io.BytesIO(content), sep="\t")

    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(content))

    elif ext == ".json":
        data = json.loads(content)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try records key or flatten
            for key in ["data", "records", "rows", "items"]:
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            return pd.DataFrame([data])

    elif ext == ".xml":
        import xmltodict
        data = xmltodict.parse(content)
        # Flatten first list found
        def find_list(d):
            for v in d.values():
                if isinstance(v, list):
                    return v
                elif isinstance(v, dict):
                    result = find_list(v)
                    if result:
                        return result
            return None
        rows = find_list(data)
        if rows:
            return pd.DataFrame(rows)
        return pd.json_normalize(data)

    elif ext == ".parquet":
        return pd.read_parquet(io.BytesIO(content))

    raise ValueError(f"Unsupported file format: {ext}")


def profile_dataframe(df: pd.DataFrame, filename: str, file_id: str) -> dict:
    columns = []
    for col in df.columns:
        series = df[col].dropna()
        sample_values = series.head(5).tolist()
        # Convert numpy types to native Python
        sample_values = [
            v.item() if hasattr(v, "item") else v for v in sample_values
        ]
        columns.append({
            "name": col,
            "sample_values": sample_values,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique()),
        })

    return {
        "file_id": file_id,
        "filename": filename,
        "format": detect_format(filename),
        "row_count": len(df),
        "columns": columns,
    }


# In-memory file store (replace with DB/S3 in production)
_file_store: dict[str, pd.DataFrame] = {}


def store_dataframe(df: pd.DataFrame) -> str:
    file_id = str(uuid.uuid4())
    _file_store[file_id] = df
    return file_id


def get_dataframe(file_id: str) -> pd.DataFrame:
    if file_id not in _file_store:
        raise KeyError(f"File {file_id} not found")
    return _file_store[file_id]


def get_sample_data(file_id: str, n: int = 5) -> list[dict]:
    df = get_dataframe(file_id)
    return df.head(n).to_dict(orient="records")
