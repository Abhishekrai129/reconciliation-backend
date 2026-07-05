"""
Generates two large synthetic trade CSV files for scale benchmarking.
Usage:  python scripts/generate_benchmark_data.py --rows 50000
Output: benchmark_source.csv  benchmark_target.csv  (in sample_data/)
"""
import argparse
import csv
import random
import os
from datetime import date, timedelta

COUNTERPARTIES = ["Goldman Sachs", "Morgan Stanley", "JPMorgan", "Barclays",
                  "Deutsche Bank", "Citi", "BNP Paribas", "UBS", "HSBC", "Credit Suisse"]
PRODUCTS       = ["Equity Swap", "Credit Default Swap", "FX Forward", "Interest Rate Swap",
                  "Total Return Swap", "Variance Swap", "Commodity Forward"]
CURRENCIES     = ["USD", "EUR", "GBP", "JPY", "CHF"]
SIDES          = ["Buy", "Sell"]

def rand_date(start=date(2024, 1, 1), end=date(2025, 12, 31)):
    return start + timedelta(days=random.randint(0, (end - start).days))

def generate(n_rows: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    src_path = os.path.join(out_dir, "benchmark_source.csv")
    tgt_path = os.path.join(out_dir, "benchmark_target.csv")

    random.seed(42)
    trade_ids = [f"TRD-{i:07d}" for i in range(1, n_rows + 1)]

    src_rows, tgt_rows = [], []
    n_breaks = max(1, n_rows // 20)   # ~5% break rate
    break_ids = set(random.sample(trade_ids, n_breaks))

    for tid in trade_ids:
        notional = round(random.uniform(10_000, 10_000_000), 2)
        cpty     = random.choice(COUNTERPARTIES)
        product  = random.choice(PRODUCTS)
        side     = random.choice(SIDES)
        ccy      = random.choice(CURRENCIES)
        trade_dt = rand_date()
        settle_dt = trade_dt + timedelta(days=random.choice([2, 3, 5]))

        src_rows.append({
            "TradeID": tid, "TradeDate": trade_dt.isoformat(),
            "SettleDate": settle_dt.isoformat(), "Counterparty": cpty,
            "Product": product, "Side": side, "Notional": notional,
            "Currency": ccy, "Rate": round(random.uniform(0.01, 0.15), 6),
        })

        # Target row — introduce intentional breaks for ~5%
        tgt_notional = notional * random.uniform(0.995, 1.005) if tid in break_ids else notional
        tgt_side = ("Sell" if side == "Buy" else "Buy") if tid in break_ids and random.random() < 0.3 else side

        tgt_rows.append({
            "Trade_ID": tid, "Trade_Date": trade_dt.isoformat(),
            "Settlement_Date": settle_dt.isoformat(), "CounterParty": cpty,
            "Instrument": product, "Direction": tgt_side,
            "Notional_Amount": round(tgt_notional, 2),
            "CCY": ccy, "Fixed_Rate": round(random.uniform(0.01, 0.15), 6),
        })

    with open(src_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=src_rows[0].keys())
        w.writeheader(); w.writerows(src_rows)

    with open(tgt_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tgt_rows[0].keys())
        w.writeheader(); w.writerows(tgt_rows)

    print(f"Generated {n_rows:,} rows  →  {src_path}")
    print(f"Generated {n_rows:,} rows  →  {tgt_path}")
    print(f"Intentional breaks: {n_breaks:,}  (~5%)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000)
    ap.add_argument("--out",  type=str, default="sample_data")
    args = ap.parse_args()
    generate(args.rows, args.out)
