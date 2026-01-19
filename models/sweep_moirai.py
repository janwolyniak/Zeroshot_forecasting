import os
import sys
import argparse
import traceback
from typing import List, Dict, Any

import pandas as pd
import torch

# Ensure project root (parent of `models/` and `utils/`) is importable
SCRIPT_DIR = r"/Users/jan/Documents/working papers/project 1/models"
PROJECT_ROOT = r"/Users/jan/Documents/working papers/project 1"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils.thresholds import DEFAULT_THRESHOLD
from moirai_runner import run_single_experiment

# constants
CONTEXT_LEN = 512
HORIZONS = [1, 3, 5, 20]
PATCH_SIZES = [16]
NUM_SAMPLES = [5, 10, 20]
BATCH_SIZE = 4

# Trading / evaluation parameters
STARTING_CAPITAL = 100_000.0
THRESHOLD = DEFAULT_THRESHOLD
FEE_RATE = 0.001

USER_DEFAULTS = {
    "data_csv": "/Users/jan/Documents/working papers/project 1/data/btc_1h_test.csv",
    "results_root": "/Users/jan/Documents/working papers/project 1/moirai-results",
    "model_family": "moirai",
    "model_id": "Salesforce/moirai-1.1-R-small",
    "timestamp_col": "timestamp",
    "target_col": "close",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

DRY_RUN_LIMIT = None


def _parse_device(device_str: str) -> str:
    return device_str or ("cuda" if torch.cuda.is_available() else "cpu")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Sweep Moirai horizons / patch sizes.")
    p.add_argument("--data_csv", type=str)
    p.add_argument("--results_root", type=str)
    p.add_argument("--model_family", type=str)
    p.add_argument("--model_id", type=str)
    p.add_argument("--timestamp_col", type=str)
    p.add_argument("--target_col", type=str)
    p.add_argument("--device", type=str)
    p.add_argument("--context_length", type=int)
    return p


def resolve_params(cli_args: argparse.Namespace) -> argparse.Namespace:
    merged = argparse.Namespace()
    merged.data_csv = cli_args.data_csv or USER_DEFAULTS["data_csv"]
    merged.results_root = cli_args.results_root or USER_DEFAULTS["results_root"]
    merged.model_family = cli_args.model_family or USER_DEFAULTS["model_family"]
    merged.model_id = cli_args.model_id or USER_DEFAULTS["model_id"]
    merged.timestamp_col = cli_args.timestamp_col or USER_DEFAULTS["timestamp_col"]
    merged.target_col = cli_args.target_col or USER_DEFAULTS["target_col"]
    merged.device = cli_args.device or USER_DEFAULTS["device"]
    merged.context_length = cli_args.context_length or CONTEXT_LEN
    return merged


def sweep(params: argparse.Namespace) -> pd.DataFrame:
    device = _parse_device(params.device)

    print("[sweep] START", flush=True)
    print("[sweep] data_csv     =", params.data_csv)
    print("[sweep] model_family =", params.model_family)
    print("[sweep] model_id     =", params.model_id)
    print("[sweep] results_root =", params.results_root)
    print("[sweep] context_len  =", params.context_length)
    print("[sweep] horizons     =", HORIZONS)
    print("[sweep] patch_sizes  =", PATCH_SIZES)
    print("[sweep] num_samples  =", NUM_SAMPLES)
    print("[sweep] eval params  =", dict(starting_capital=STARTING_CAPITAL, threshold=THRESHOLD, fee_rate=FEE_RATE))

    assert os.path.exists(params.data_csv), "data_csv not found"
    os.makedirs(params.results_root, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    combos = [(h, ps, ns) for h in HORIZONS for ps in PATCH_SIZES for ns in NUM_SAMPLES]
    if DRY_RUN_LIMIT is not None:
        combos = combos[:DRY_RUN_LIMIT]

    for idx, (horizon, patch_size, num_samples) in enumerate(combos, 1):
        run_name = f"ctx{params.context_length}_h{horizon}_ps{patch_size}_ns{num_samples}"
        out_dir = os.path.join(params.results_root, run_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"[sweep] ({idx}/{len(combos)}) running {run_name} ...", flush=True)
        try:
            summary = run_single_experiment(
                data_csv=params.data_csv,
                output_dir=out_dir,
                model_family=params.model_family,
                model_id=params.model_id,
                context_length=params.context_length,
                horizon=horizon,
                patch_size=patch_size,
                num_samples=num_samples,
                batch_size=BATCH_SIZE,
                device=device,
                starting_capital=STARTING_CAPITAL,
                threshold=THRESHOLD,
                fee_rate=FEE_RATE,
                timestamp_col=params.timestamp_col,
                target_col=params.target_col,
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
    master_csv_path = os.path.join(params.results_root, f"summary_ctx{params.context_length}.csv")
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
