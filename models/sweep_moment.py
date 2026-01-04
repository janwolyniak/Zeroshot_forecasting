import os
import sys
import argparse
import traceback
from typing import List, Dict, Any

import pandas as pd
import torch

# Ensure project root (parent of `models/` and `utils/`) is importable
SCRIPT_DIR = r'/Users/jan/Documents/working papers/project 1/models'
PROJECT_ROOT = r'/Users/jan/Documents/working papers/project 1'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from moment_runner import run_single_experiment  # local import from models/

# constants
CONTEXT_LEN = 512

# Parameter grid
HORIZONS = [1, 3, 5, 20]
NORMALIZE_INPUTS = True
BATCH_SIZE = 128

# Trading / evaluation parameters
STARTING_CAPITAL = 100_000.0
THRESHOLD = 0.005
FEE_RATE = 0.001


USER_DEFAULTS = {
    "data_csv": "/Users/jan/Documents/working papers/project 1/data/btc_1h_test.csv",
    "results_root": "/Users/jan/Documents/working papers/project 1/moment-results",
    "timestamp_col": "timestamp",
    "target_col": "close",
    "repo_id": "AutonLab/MOMENT-1-small",
}

DRY_RUN_LIMIT = None


def build_arg_parser():
    p = argparse.ArgumentParser(description="Sweep MOMENT horizons / normalization.")
    p.add_argument("--data_csv", type=str)
    p.add_argument("--results_root", type=str)
    p.add_argument("--timestamp_col", type=str)
    p.add_argument("--target_col", type=str)
    p.add_argument("--repo_id", type=str)
    return p


def resolve_params(cli_args: argparse.Namespace) -> argparse.Namespace:
    merged = argparse.Namespace()
    merged.data_csv = cli_args.data_csv or USER_DEFAULTS["data_csv"]
    merged.results_root = cli_args.results_root or USER_DEFAULTS["results_root"]
    merged.timestamp_col = cli_args.timestamp_col or USER_DEFAULTS["timestamp_col"]
    merged.target_col = cli_args.target_col or USER_DEFAULTS["target_col"]
    merged.repo_id = cli_args.repo_id or USER_DEFAULTS["repo_id"]
    return merged


def sweep(params: argparse.Namespace) -> pd.DataFrame:
    print("[sweep] START", flush=True)
    print("[sweep] data_csv     =", params.data_csv)
    print("[sweep] results_root =", params.results_root)
    print("[sweep] context_len  =", CONTEXT_LEN)
    print("[sweep] horizons     =", HORIZONS)
    print("[sweep] normalize    =", NORMALIZE_INPUTS)
    print("[sweep] batch_size   =", BATCH_SIZE)
    print("[sweep] repo_id      =", params.repo_id)
    print("[sweep] eval params  =", dict(starting_capital=STARTING_CAPITAL, threshold=THRESHOLD, fee_rate=FEE_RATE))

    assert os.path.exists(params.data_csv), "data_csv not found"
    os.makedirs(params.results_root, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    combos = list(HORIZONS)
    if DRY_RUN_LIMIT is not None:
        combos = combos[:DRY_RUN_LIMIT]

    for idx, horizon in enumerate(combos, 1):
        run_name = f"ctx{CONTEXT_LEN}_h{horizon}_norm{str(NORMALIZE_INPUTS)}"
        out_dir = os.path.join(params.results_root, run_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"[sweep] ({idx}/{len(combos)}) running {run_name} …", flush=True)
        try:
            summary = run_single_experiment(
                data_csv=params.data_csv,
                output_dir=out_dir,
                context_length=CONTEXT_LEN,
                horizon=horizon,
                normalize_inputs=NORMALIZE_INPUTS,
                batch_size=BATCH_SIZE,
                starting_capital=STARTING_CAPITAL,
                threshold=THRESHOLD,
                fee_rate=FEE_RATE,
                timestamp_col=params.timestamp_col,
                target_col=params.target_col,
                repo_id=params.repo_id,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            row = dict(summary)
            row["run_name"] = run_name
            rows.append(row)

            pd.DataFrame([row]).to_csv(os.path.join(out_dir, "summary_row.csv"), index=False)

            print(f"[sweep]   ✓ finished {run_name}", flush=True)
        except Exception:
            print(f"[sweep]   ✗ FAILED {run_name}", flush=True)
            traceback.print_exc()

    df_summary = pd.DataFrame(rows)
    master_csv_path = os.path.join(params.results_root, f"summary_ctx{CONTEXT_LEN}.csv")
    df_summary.to_csv(master_csv_path, index=False)
    print(f"[sweep] Wrote master CSV: {master_csv_path}", flush=True)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("[sweep] HEAD:")
        print(df_summary.head())

    print("[sweep] FINISHED", flush=True)
    return df_summary


def main():
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    params = resolve_params(cli_args)
    sweep(params)


if __name__ == "__main__":
    main()
