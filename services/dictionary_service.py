"""
Financial Field Dictionary & Value Registry — the AI's living knowledge base.

Three layers of knowledge, stored in SQLite and growing with every human confirmation:

  Layer 1 — Field Dictionary   : canonical field names + all known aliases
  Layer 2 — Value Registry     : abbreviation → full-form lookup tables per field type
  Layer 3 — Confirmed Mappings : human-verified source→target pairs with match rules

The LLM is injected with relevant dictionary context before each mapping call.
This turns vague guessing into structured lookup — like giving a new analyst the team's notes.

Learning loop:
  Human confirms mapping → confirmed_count increments → dictionary entry gets promoted
  Human adds a new alias → stored for all future runs on similar schemas
  Human corrects a value mapping (B→Buy) → value_registry updated
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

DB_PATH = os.getenv("DICT_DB_PATH", "dictionary.db")

# ── Pre-seeded financial domain knowledge ─────────────────────────────────────
# Format: (canonical_name, domain, data_type, aliases_list, value_map_dict)

SEED_ENTRIES: list[tuple[str, str, str, list[str], dict]] = [
    # ── Trade identifiers ──────────────────────────────────────────────────
    ("Trade ID", "all", "identifier",
     ["trade_id", "TradeID", "trd_id", "ref_no", "reference", "trade_ref",
      "TRADEID", "trx_id", "txn_id", "confirmation_id", "ConfirmationID",
      "order_id", "OrderID", "deal_id", "DealID", "ticket_id", "FIX_ID",
      "trade_number", "trd_num", "TRDNUM", "booking_ref"],
     {}),

    ("ISIN", "all", "identifier",
     ["isin", "ISIN", "security_id", "sec_id", "instrument_id", "SecurityID",
      "securityidentifier", "ric", "RIC", "bloomberg_id", "BBGID"],
     {}),

    ("CUSIP", "all", "identifier",
     ["cusip", "CUSIP", "security_no", "sec_no", "SecurityNo"],
     {}),

    ("SEDOL", "all", "identifier",
     ["sedol", "SEDOL"],
     {}),

    ("LEI", "all", "identifier",
     ["lei", "LEI", "legal_entity_id", "LegalEntityID"],
     {}),

    ("UETR", "payments", "identifier",
     ["uetr", "UETR", "unique_end_to_end_id", "e2e_ref", "end_to_end_id"],
     {}),

    # ── Dates ─────────────────────────────────────────────────────────────
    ("Trade Date", "all", "date",
     ["trade_date", "trd_dt", "TradeDate", "TRDDT", "td", "execution_date",
      "deal_date", "booking_date", "order_date", "transaction_date",
      "TRADEDATE", "trd_date", "trade_dt"],
     {}),

    ("Settlement Date", "all", "date",
     ["settlement_date", "sttl_dt", "SettlementDate", "settle_dt", "STTLDT",
      "value_date", "ValueDate", "delivery_date", "maturity_date",
      "settlement_dt", "SETTLEDATE", "stl_dt"],
     {}),

    ("Value Date", "nostro", "date",
     ["value_date", "val_dt", "ValueDate", "VALDT", "posting_date",
      "effective_date", "credit_date"],
     {}),

    ("Expiry Date", "all", "date",
     ["expiry_date", "expiration_date", "exp_date", "EXPDT", "maturity"],
     {}),

    ("Pay Date", "corporate_actions", "date",
     ["pay_date", "payment_date", "PayDate", "PAYDT"],
     {}),

    ("Ex Date", "corporate_actions", "date",
     ["ex_date", "ex_dividend_date", "ExDate", "EXDT"],
     {}),

    # ── Prices & Amounts ──────────────────────────────────────────────────
    ("Execution Price", "trade_confirm", "price",
     ["exec_px", "price", "Price", "execution_price", "ExecutionPrice",
      "px", "PRICE", "exec_price", "trade_price", "fill_price",
      "avg_price", "AvgPrice", "deal_price", "DealPrice", "rate"],
     {}),

    ("Notional", "all", "numeric",
     ["notional", "NOTIONAL", "ntnl", "notional_amount", "NotionalAmount",
      "face_value", "nominal", "principal", "gross_amount", "GrossAmount",
      "consideration", "trade_amount", "amount"],
     {}),

    ("Quantity", "all", "numeric",
     ["qty", "quantity", "Quantity", "QTY", "shares", "units", "lots",
      "QTYM", "nominal_qty", "face_amount", "contract_size",
      "settled_qty", "pending_qty", "position"],
     {}),

    ("Amount", "all", "numeric",
     ["amount", "Amount", "AMOUNT", "value", "Value", "gross", "net",
      "total", "Total", "sum", "Sum"],
     {}),

    ("Net Amount", "all", "numeric",
     ["net_amount", "NetAmount", "net", "NET", "net_value", "net_settlement"],
     {}),

    ("Amount Per Share", "corporate_actions", "numeric",
     ["amount_per_share", "AmountPerShare", "div_amount", "dividend_amount",
      "distribution_amount", "rate_per_share"],
     {}),

    ("Rate", "all", "rate",
     ["rate", "Rate", "RATE", "interest_rate", "coupon_rate", "fixed_rate",
      "floating_rate", "spread", "yield"],
     {}),

    # ── Direction / Side ──────────────────────────────────────────────────
    ("Side", "trade_confirm", "side",
     ["side", "Side", "SIDE", "bs_ind", "buy_sell", "BuySell",
      "direction", "Direction", "trade_side", "action", "Action",
      "order_side", "BookingSide", "dr_cr"],
     {"B": "Buy", "S": "Sell", "BUY": "Buy", "SELL": "Sell",
      "1": "Buy", "-1": "Sell", "P": "Buy", "V": "Sell",
      "b": "Buy", "s": "Sell",
      "D": "Debit", "C": "Credit",
      "DEBIT": "Debit", "CREDIT": "Credit",
      "Long": "Buy", "Short": "Sell", "L": "Buy"}),

    ("Dr/Cr", "nostro", "side",
     ["dr_cr", "DrCr", "debit_credit", "dc_flag", "sign"],
     {"D": "Debit", "C": "Credit", "DR": "Debit", "CR": "Credit",
      "d": "Debit", "c": "Credit"}),

    # ── Counterparty ──────────────────────────────────────────────────────
    ("Counterparty", "all", "text",
     ["counterparty", "Counterparty", "cpty", "CPTY", "cp",
      "party", "legal_entity", "broker", "Broker", "dealer", "Dealer",
      "counterpart", "contra_party", "CounterParty", "counter_party",
      "institution", "bank", "Bank"],
     {"GS": "Goldman Sachs", "MS": "Morgan Stanley", "JPM": "JPMorgan",
      "JPMC": "JPMorgan Chase", "C": "Citi", "CITI": "Citi",
      "DB": "Deutsche Bank", "DEUT": "Deutsche Bank",
      "UBS": "UBS", "BARC": "Barclays", "BAR": "Barclays",
      "BNP": "BNP Paribas", "BNPP": "BNP Paribas",
      "CS": "Credit Suisse", "CSFB": "Credit Suisse",
      "HSBC": "HSBC", "SG": "Société Générale",
      "NOM": "Nomura", "MUF": "MUFG",
      "BofA": "Bank of America", "BAML": "Bank of America"}),

    # ── Product / Instrument ──────────────────────────────────────────────
    ("Product", "all", "text",
     ["product", "Product", "instrument", "Instrument", "asset_class",
      "AssetClass", "security_type", "SecurityType", "trade_type",
      "product_type", "ProductType", "instr_type", "asset"],
     {"EQS": "Equity Swap", "CDS": "Credit Default Swap",
      "IRS": "Interest Rate Swap", "FXF": "FX Forward",
      "TRS": "Total Return Swap", "OPT": "Option",
      "FUT": "Future", "FWD": "Forward",
      "BOND": "Bond", "EQ": "Equity"}),

    ("Currency", "all", "currency",
     ["currency", "Currency", "ccy", "CCY", "curr", "Curr",
      "base_ccy", "BaseCcy", "reporting_currency", "settlement_ccy",
      "trade_ccy", "denomination"],
     {}),

    # ── Account / Entity ──────────────────────────────────────────────────
    ("Account", "all", "identifier",
     ["account", "Account", "account_id", "AccountID", "acct", "ACCT",
      "portfolio", "fund", "Fund", "entity", "Entity",
      "book", "Book", "cost_center", "legal_entity"],
     {}),

    ("GL Account", "intercompany", "identifier",
     ["gl_account", "GLAccount", "gl_code", "account_code", "BELNR",
      "BUKRS", "DMBTR", "cost_account", "ledger_account"],
     {}),

    # ── Bank statement specific ───────────────────────────────────────────
    ("Reference", "nostro", "identifier",
     ["reference", "ref", "Ref", "narrative", "bank_reference",
      "payment_ref", "PaymentRef", "transaction_ref", "cust_ref"],
     {}),

    ("Bank Reference", "nostro", "identifier",
     ["bank_reference", "BankRef", "bank_ref", "correspondent_ref",
      "bank_narrative", "bank_txn_id"],
     {}),

    # ── Corporate actions ─────────────────────────────────────────────────
    ("Event Type", "corporate_actions", "text",
     ["event_type", "EventType", "action_type", "ActionType",
      "corporate_action", "ca_type", "announcement_type"],
     {"DIV": "Dividend", "SPLIT": "Stock Split", "MERGE": "Merger",
      "SPIN": "Spin-off", "RIGH": "Rights Issue",
      "TEND": "Tender Offer", "BONU": "Bonus Issue"}),

    ("Ticker", "all", "identifier",
     ["ticker", "Ticker", "TICKER", "symbol", "Symbol",
      "stock_symbol", "equity_ticker", "bbg_ticker"],
     {}),
]


# ── DB init ───────────────────────────────────────────────────────────────────

def init_dict_db():
    conn = _conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS field_dictionary (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name  TEXT NOT NULL,
            domain          TEXT NOT NULL DEFAULT 'all',
            data_type       TEXT NOT NULL DEFAULT 'text',
            aliases         TEXT NOT NULL DEFAULT '[]',
            value_map       TEXT NOT NULL DEFAULT '{}',
            confirmed_count INTEGER NOT NULL DEFAULT 0,
            last_confirmed  TEXT,
            created_at      TEXT NOT NULL
        );

        -- Rule Book: stores confirmed matching rules learned from human review
        -- Each row = one field-pair rule confirmed by a human at least once.
        -- confirmed_count tracks how many runs have validated this rule.
        -- auto_apply triggers when confirmed_count >= auto_apply_threshold.
        CREATE TABLE IF NOT EXISTS rule_book (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_source    TEXT NOT NULL,   -- e.g. "Execution Price"
            canonical_target    TEXT NOT NULL,   -- e.g. "Execution Price"
            match_type          TEXT NOT NULL,   -- exact | numeric_tolerance | date_tolerance | value_lookup | fuzzy
            threshold           TEXT,            -- NULL, "0.01", or JSON dict for value_lookup
            confirmed_count     INTEGER NOT NULL DEFAULT 1,
            auto_apply          INTEGER NOT NULL DEFAULT 0,  -- 1 when confirmed >= threshold
            auto_apply_threshold INTEGER NOT NULL DEFAULT 3, -- confirmations needed for auto-apply
            last_confirmed      TEXT NOT NULL,
            example_source_cols TEXT NOT NULL DEFAULT '[]',  -- raw col names seen for this canonical
            example_target_cols TEXT NOT NULL DEFAULT '[]',
            notes               TEXT DEFAULT '',
            created_at          TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS learning_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type      TEXT NOT NULL,
            canonical_name  TEXT,
            alias_added     TEXT,
            value_added     TEXT,
            rule_detail     TEXT,
            run_id          TEXT,
            confirmed_by    TEXT DEFAULT 'human',
            created_at      TEXT NOT NULL
        );
    """)
    conn.commit()

    # Seed if empty
    count = conn.execute("SELECT COUNT(*) FROM field_dictionary").fetchone()[0]
    if count == 0:
        _seed(conn)
    conn.close()


def _seed(conn: sqlite3.Connection):
    now = _now()
    for canonical, domain, dtype, aliases, value_map in SEED_ENTRIES:
        conn.execute(
            """INSERT INTO field_dictionary
               (canonical_name, domain, data_type, aliases, value_map, confirmed_count, created_at)
               VALUES (?, ?, ?, ?, ?, 0, ?)""",
            (canonical, domain, dtype, json.dumps(aliases), json.dumps(value_map), now),
        )
    conn.commit()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Lookup helpers ────────────────────────────────────────────────────────────

def lookup_field(raw_name: str) -> dict | None:
    """Find the canonical entry for a raw column name (alias or canonical match)."""
    name_lower = raw_name.strip().lower()
    conn = _conn()
    rows = conn.execute("SELECT * FROM field_dictionary").fetchall()
    conn.close()

    best: dict | None = None
    best_score = 0.0

    for row in rows:
        aliases: list[str] = json.loads(row["aliases"])
        canonical_lower = row["canonical_name"].lower()

        # Exact match: canonical name or any alias
        all_names = [canonical_lower] + [a.lower() for a in aliases]
        if name_lower in all_names:
            return dict(row)

        # Fuzzy match fallback
        score = max(SequenceMatcher(None, name_lower, n).ratio() for n in all_names)
        if score > best_score and score > 0.75:
            best_score = score
            best = dict(row)

    return best


def get_context_for_mapping(col_names_a: list[str], col_names_b: list[str]) -> str:
    """Build a compact dictionary context string to inject into the LLM prompt.

    Returns a concise text block showing:
      - Which canonical field each raw column likely represents
      - Known aliases and value mappings for those fields
    """
    relevant: dict[str, dict] = {}

    for name in col_names_a + col_names_b:
        entry = lookup_field(name)
        if entry and entry["canonical_name"] not in relevant:
            relevant[entry["canonical_name"]] = entry

    if not relevant:
        return ""

    lines = ["FIELD DICTIONARY (use this to identify field meanings and value normalisation):"]
    for entry in relevant.values():
        aliases = json.loads(entry["aliases"])
        value_map = json.loads(entry["value_map"])
        conf = entry["confirmed_count"]
        badge = f" [confirmed ×{conf}]" if conf > 0 else " [seeded]"
        alias_str = ", ".join(aliases[:8])
        line = f'  • {entry["canonical_name"]} ({entry["data_type"]}){badge}: aliases=[{alias_str}]'
        if value_map:
            vm_str = ", ".join(f"{k}→{v}" for k, v in list(value_map.items())[:6])
            line += f"  values=[{vm_str}]"
        lines.append(line)

    return "\n".join(lines)


def normalize_column_name(raw: str, actual_columns: list[str]) -> str:
    """Map a label/alias back to the actual column name present in the file.

    The LLM sometimes returns the inferred label ('Trade Date') instead of
    the actual column name ('trd_dt'). This function corrects that.
    """
    # Direct match
    if raw in actual_columns:
        return raw

    raw_lower = raw.strip().lower()

    # Case-insensitive direct match
    for col in actual_columns:
        if col.lower() == raw_lower:
            return col

    # Dictionary alias lookup — find which canonical this maps to, then find
    # which actual column also maps to that canonical
    entry = lookup_field(raw)
    if entry:
        all_aliases = [entry["canonical_name"]] + json.loads(entry["aliases"])
        aliases_lower = {a.lower() for a in all_aliases}
        for col in actual_columns:
            if col.lower() in aliases_lower or lookup_field(col) and lookup_field(col)["canonical_name"] == entry["canonical_name"]:
                return col

    # Fuzzy fallback
    best_col = raw
    best_score = 0.0
    for col in actual_columns:
        score = SequenceMatcher(None, raw_lower, col.lower()).ratio()
        if score > best_score:
            best_score = score
            best_col = col

    return best_col if best_score > 0.5 else raw


# ── Learning loop ─────────────────────────────────────────────────────────────

def record_confirmed_mapping(
    source_col: str,
    target_col: str,
    run_id: str | None = None,
):
    """Called when a human confirms a field mapping.

    Looks up both columns in the dictionary, increments confirmed_count,
    and cross-links aliases if not already present.
    """
    conn = _conn()
    now = _now()

    for col in [source_col, target_col]:
        entry = lookup_field(col)
        if entry:
            # Increment confirmation counter
            conn.execute(
                "UPDATE field_dictionary SET confirmed_count = confirmed_count + 1, last_confirmed = ? WHERE id = ?",
                (now, entry["id"]),
            )
            # Add the raw column name as an alias if not already there
            aliases: list[str] = json.loads(entry["aliases"])
            if col not in aliases and col != entry["canonical_name"]:
                aliases.append(col)
                conn.execute(
                    "UPDATE field_dictionary SET aliases = ? WHERE id = ?",
                    (json.dumps(aliases), entry["id"]),
                )
                conn.execute(
                    "INSERT INTO learning_log (event_type, canonical_name, alias_added, run_id, created_at) VALUES (?,?,?,?,?)",
                    ("alias_discovered", entry["canonical_name"], col, run_id, now),
                )

    conn.commit()
    conn.close()


def learn_value_mapping(field_type: str, abbreviation: str, full_form: str, run_id: str | None = None):
    """Add a new abbreviation→full_form pair discovered/confirmed by a human."""
    conn = _conn()
    now = _now()

    # Find the entry for this field type (by canonical name or data_type)
    row = conn.execute(
        "SELECT * FROM field_dictionary WHERE LOWER(canonical_name) = ? OR LOWER(data_type) = ?",
        (field_type.lower(), field_type.lower()),
    ).fetchone()

    if row:
        vm: dict = json.loads(row["value_map"])
        if abbreviation not in vm:
            vm[abbreviation] = full_form
            conn.execute(
                "UPDATE field_dictionary SET value_map = ? WHERE id = ?",
                (json.dumps(vm), row["id"]),
            )
            conn.execute(
                "INSERT INTO learning_log (event_type, canonical_name, value_added, run_id, created_at) VALUES (?,?,?,?,?)",
                ("value_learned", row["canonical_name"], f"{abbreviation}→{full_form}", run_id, now),
            )
    conn.commit()
    conn.close()


def get_all_entries() -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM field_dictionary ORDER BY confirmed_count DESC, canonical_name"
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["aliases"] = json.loads(d["aliases"])
        d["value_map"] = json.loads(d["value_map"])
        result.append(d)
    return result


def get_learning_log(limit: int = 50) -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM learning_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    conn = _conn()
    total = conn.execute("SELECT COUNT(*) FROM field_dictionary").fetchone()[0]
    confirmed = conn.execute("SELECT COUNT(*) FROM field_dictionary WHERE confirmed_count > 0").fetchone()[0]
    total_aliases = 0
    for row in conn.execute("SELECT aliases FROM field_dictionary").fetchall():
        total_aliases += len(json.loads(row[0]))
    value_maps = conn.execute("SELECT COUNT(*) FROM field_dictionary WHERE value_map != '{}'").fetchone()[0]
    learnings = conn.execute("SELECT COUNT(*) FROM learning_log").fetchone()[0]
    rule_count = conn.execute("SELECT COUNT(*) FROM rule_book").fetchone()[0]
    auto_rules = conn.execute("SELECT COUNT(*) FROM rule_book WHERE auto_apply = 1").fetchone()[0]
    conn.close()
    return {
        "total_entries":       total,
        "confirmed_by_human":  confirmed,
        "total_aliases":       total_aliases,
        "fields_with_value_maps": value_maps,
        "total_learnings":     learnings,
        "rule_book_count":     rule_count,
        "auto_apply_rules":    auto_rules,
    }


# ── Rule Book ──────────────────────────────────────────────────────────────────

def learn_rule(
    source_col: str,
    target_col: str,
    match_type: str,
    threshold: Any,
    run_id: str | None = None,
) -> dict:
    """Store or reinforce a confirmed matching rule in the rule book.

    Called automatically from the pipeline review endpoint every time a human
    approves a rule. The rule is keyed on canonical field names so it generalises
    across different raw column names for the same concept.

    Returns the stored rule dict with current confirmed_count and auto_apply flag.
    """
    conn = _conn()
    now = _now()

    # Resolve canonical names for both columns
    src_entry = lookup_field(source_col)
    tgt_entry = lookup_field(target_col)
    canonical_src = src_entry["canonical_name"] if src_entry else source_col
    canonical_tgt = tgt_entry["canonical_name"] if tgt_entry else target_col

    threshold_str = json.dumps(threshold) if isinstance(threshold, dict) else (str(threshold) if threshold is not None else None)

    existing = conn.execute(
        "SELECT * FROM rule_book WHERE canonical_source = ? AND canonical_target = ?",
        (canonical_src, canonical_tgt),
    ).fetchone()

    if existing:
        new_count = existing["confirmed_count"] + 1
        auto = 1 if new_count >= existing["auto_apply_threshold"] else 0

        # Merge example column names
        ex_src: list = json.loads(existing["example_source_cols"])
        ex_tgt: list = json.loads(existing["example_target_cols"])
        if source_col not in ex_src:
            ex_src.append(source_col)
        if target_col not in ex_tgt:
            ex_tgt.append(target_col)

        conn.execute(
            """UPDATE rule_book SET
               match_type = ?, threshold = ?, confirmed_count = ?,
               auto_apply = ?, last_confirmed = ?,
               example_source_cols = ?, example_target_cols = ?
               WHERE id = ?""",
            (match_type, threshold_str, new_count, auto, now,
             json.dumps(ex_src), json.dumps(ex_tgt), existing["id"]),
        )
        rule_id = existing["id"]
    else:
        cur = conn.execute(
            """INSERT INTO rule_book
               (canonical_source, canonical_target, match_type, threshold,
                confirmed_count, auto_apply, last_confirmed,
                example_source_cols, example_target_cols, created_at)
               VALUES (?,?,?,?,1,0,?,?,?,?)""",
            (canonical_src, canonical_tgt, match_type, threshold_str, now,
             json.dumps([source_col]), json.dumps([target_col]), now),
        )
        rule_id = cur.lastrowid

    conn.execute(
        """INSERT INTO learning_log
           (event_type, canonical_name, rule_detail, run_id, created_at)
           VALUES (?,?,?,?,?)""",
        ("rule_confirmed",
         f"{canonical_src} → {canonical_tgt}",
         f"{match_type}  threshold={threshold_str}",
         run_id, now),
    )
    conn.commit()
    result = dict(conn.execute("SELECT * FROM rule_book WHERE id = ?", (rule_id,)).fetchone())
    conn.close()

    result["example_source_cols"] = json.loads(result["example_source_cols"])
    result["example_target_cols"] = json.loads(result["example_target_cols"])
    result["threshold_parsed"]    = json.loads(result["threshold"]) if result["threshold"] and result["threshold"].startswith("{") else result["threshold"]
    return result


def lookup_rules(col_names_a: list[str], col_names_b: list[str]) -> list[dict]:
    """Find rule book entries applicable to these column sets.

    For each column in A, resolve its canonical name, then find rule book
    entries where canonical_source matches. Same for B side.
    Returns matched rules translated back to the actual raw column names.

    auto_apply=True rules are safe to apply without human review.
    """
    conn = _conn()
    all_rules = conn.execute("SELECT * FROM rule_book ORDER BY confirmed_count DESC").fetchall()
    conn.close()

    # Build canonical → raw-name mappings for both sides
    canonical_a: dict[str, str] = {}  # canonical → raw col name in file A
    for col in col_names_a:
        entry = lookup_field(col)
        if entry:
            canonical_a[entry["canonical_name"]] = col

    canonical_b: dict[str, str] = {}
    for col in col_names_b:
        entry = lookup_field(col)
        if entry:
            canonical_b[entry["canonical_name"]] = col

    matched = []
    for r in all_rules:
        cs, ct = r["canonical_source"], r["canonical_target"]
        if cs in canonical_a and ct in canonical_b:
            threshold_raw = r["threshold"]
            threshold_parsed = (
                json.loads(threshold_raw)
                if threshold_raw and threshold_raw.startswith("{")
                else (float(threshold_raw) if threshold_raw and threshold_raw not in ("None", "null") else None)
            )
            matched.append({
                "source_column":    canonical_a[cs],
                "target_column":    canonical_b[ct],
                "canonical_source": cs,
                "canonical_target": ct,
                "match_type":       r["match_type"],
                "threshold":        threshold_parsed,
                "confirmed_count":  r["confirmed_count"],
                "auto_apply":       bool(r["auto_apply"]),
                "reasoning":        f"Rule book (confirmed ×{r['confirmed_count']}){' — auto-applied' if r['auto_apply'] else ''}",
            })

    return matched


def get_rule_book() -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM rule_book ORDER BY confirmed_count DESC, canonical_source"
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["example_source_cols"] = json.loads(d["example_source_cols"])
        d["example_target_cols"] = json.loads(d["example_target_cols"])
        raw_t = d.get("threshold")
        d["threshold_parsed"] = (
            json.loads(raw_t)
            if raw_t and raw_t.startswith("{")
            else raw_t
        )
        result.append(d)
    return result


def update_rule(rule_id: int, match_type: str, threshold: Any, notes: str = "") -> bool:
    """Human edits a rule directly from the Knowledge Base UI."""
    conn = _conn()
    threshold_str = json.dumps(threshold) if isinstance(threshold, dict) else (str(threshold) if threshold is not None else None)
    conn.execute(
        "UPDATE rule_book SET match_type = ?, threshold = ?, notes = ? WHERE id = ?",
        (match_type, threshold_str, notes, rule_id),
    )
    conn.execute(
        "INSERT INTO learning_log (event_type, rule_detail, created_at) VALUES (?,?,?)",
        ("rule_edited", f"id={rule_id} match_type={match_type} threshold={threshold_str}", _now()),
    )
    conn.commit()
    conn.close()
    return True
