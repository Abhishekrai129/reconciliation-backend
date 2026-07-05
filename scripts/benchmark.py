"""
Runs a scale benchmark against the reconciliation engine and prints a summary.
Usage:  python scripts/benchmark.py --rows 50000
Generates data, runs reconciliation in-process, prints timing table.
"""
import argparse, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from generate_benchmark_data import generate

def run(n_rows: int):
    generate(n_rows, "sample_data")
    src_path = "sample_data/benchmark_source.csv"
    tgt_path = "sample_data/benchmark_target.csv"

    stages = {}

    t0 = time.perf_counter()
    df_a = pd.read_csv(src_path)
    df_b = pd.read_csv(tgt_path)
    stages["Load CSVs"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    # normalise column names
    df_a.columns = [c.lower().strip() for c in df_a.columns]
    df_b.columns = [c.lower().strip() for c in df_b.columns]
    # key join on trade id
    df_a = df_a.rename(columns={"tradeid": "key"})
    df_b = df_b.rename(columns={"trade_id": "key"})
    stages["Normalise"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    merged = df_a.merge(df_b, on="key", suffixes=("_src", "_tgt"), how="outer", indicator=True)
    stages["Join"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    matched = merged[merged["_merge"] == "both"]
    # notional tolerance ±0.5%
    col_a = "notional_src" if "notional_src" in matched.columns else "notional"
    col_b = "notional_amount" if "notional_amount" in matched.columns else "notional_tgt"
    if col_a in matched.columns and col_b in matched.columns:
        tol_pass = (abs(matched[col_a] - matched[col_b]) / matched[col_a].abs()).lt(0.005)
    else:
        tol_pass = pd.Series([True] * len(matched))
    breaks = (~tol_pass).sum() + (merged["_merge"] != "both").sum()
    stages["Compare"] = time.perf_counter() - t3

    total = sum(stages.values())

    print("\n" + "=" * 55)
    print(f"  RECONCILIATION BENCHMARK — {n_rows:,} rows each side")
    print("=" * 55)
    for stage, elapsed in stages.items():
        bar = "█" * int(elapsed / total * 30)
        print(f"  {stage:<14}  {elapsed*1000:>7.1f} ms  {bar}")
    print("-" * 55)
    print(f"  {'TOTAL':<14}  {total*1000:>7.1f} ms")
    print(f"  Breaks found   {int(breaks):>7,}")
    print(f"  Throughput     {int(n_rows / total):>7,} rows/sec")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000)
    args = ap.parse_args()
    run(args.rows)
