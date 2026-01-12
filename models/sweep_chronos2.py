import os
import sys
import argparse
import traceback
from typing import List, Dict, Any, Optional

import pandas as pd
import torch

# Ensure project root (parent of `models/` and `utils/`) is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from chronos2_runner import run_single_experiment  # local import from models/


# constants
CONTEXT_LEN = 512
HORIZONS = [1, 3, 5, 20]

# Trading / evaluation parameters
STARTING_CAPITAL = 100_000.0
THRESHOLD = 0.005
FEE_RATE = 0.001
DEFAULT_BATCH_SIZE = 32

DEFAULT_FEATURE_COLS: Optional[List[str]] = ["open", "high", "low", "volume"]
DEFAULT_USE_FUTURE_COVARIATES = False
DEFAULT_CROSS_LEARNING = False

USER_DEFAULTS = {
    "data_csv": os.path.join(PROJECT_ROOT, "data", "btc_1h_test.csv"),
    "results_root": os.path.join(PROJECT_ROOT, "chronos2-results"),
    "timestamp_col": "timestamp",
    "target_col": "close",
    "model_path": "amazon/chronos-2",
}

DRY_RUN_LIMIT = None


def _parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def build_arg_parser():
    p = argparse.ArgumentParser(description="Sweep Chronos-2 horizons (with optional covariates).")
    p.add_argument("--data_csv", type=str)
    p.add_argument("--results_root", type=str)
    p.add_argument("--timestamp_col", type=str)
    p.add_argument("--target_col", type=str)
    p.add_argument("--model_path", type=str)
    p.add_argument("--context_length", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--feature_cols", type=str, default=None, help="Comma-separated feature columns (covariates).")
    p.add_argument(
        "--use_future_covariates",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_FUTURE_COVARIATES,
        help="If set, uses covariate values over the forecast horizon (can leak future info in backtests).",
    )
    p.add_argument(
        "--cross_learning",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CROSS_LEARNING,
        help="Enable Chronos-2 cross-learning mode.",
    )
    return p


def resolve_params(cli_args: argparse.Namespace) -> argparse.Namespace:
    merged = argparse.Namespace()
    merged.data_csv = cli_args.data_csv or USER_DEFAULTS["data_csv"]
    merged.results_root = cli_args.results_root or USER_DEFAULTS["results_root"]
    merged.timestamp_col = cli_args.timestamp_col or USER_DEFAULTS["timestamp_col"]
    merged.target_col = cli_args.target_col or USER_DEFAULTS["target_col"]
    merged.model_path = cli_args.model_path or USER_DEFAULTS["model_path"]
    merged.context_length = cli_args.context_length or CONTEXT_LEN
    merged.batch_size = cli_args.batch_size or DEFAULT_BATCH_SIZE
    merged.feature_cols = _parse_csv_list(cli_args.feature_cols)
    if merged.feature_cols is None:
        merged.feature_cols = DEFAULT_FEATURE_COLS
    # Future covariates are disallowed: force False regardless of CLI input.
    merged.use_future_covariates = False
    merged.cross_learning = bool(cli_args.cross_learning)
    return merged


def sweep(params: argparse.Namespace) -> pd.DataFrame:
    print("[sweep2] START", flush=True)
    print("[sweep2] data_csv     =", params.data_csv)
    print("[sweep2] results_root =", params.results_root)
    print("[sweep2] model_path   =", params.model_path)
    print("[sweep2] context_len  =", params.context_length)
    print("[sweep2] horizons     =", HORIZONS)
    print("[sweep2] batch_size   =", params.batch_size)
    print("[sweep2] feature_cols =", params.feature_cols)
    print("[sweep2] future_cov   =", params.use_future_covariates)
    print("[sweep2] cross_learn  =", params.cross_learning)
    print("[sweep2] eval params  =", dict(starting_capital=STARTING_CAPITAL, threshold=THRESHOLD, fee_rate=FEE_RATE))

    assert os.path.exists(params.data_csv), "data_csv not found"
    os.makedirs(params.results_root, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    combos = list(HORIZONS)
    if DRY_RUN_LIMIT is not None:
        combos = combos[:DRY_RUN_LIMIT]

    for idx, horizon in enumerate(combos, 1):
        run_name = f"ctx{params.context_length}_h{horizon}_cov{len(params.feature_cols or [])}_fut{int(params.use_future_covariates)}"
        out_dir = os.path.join(params.results_root, run_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"[sweep2] ({idx}/{len(combos)}) running {run_name} …", flush=True)
        try:
            summary = run_single_experiment(
                data_csv=params.data_csv,
                output_dir=out_dir,
                context_length=params.context_length,
                horizon=horizon,
                starting_capital=STARTING_CAPITAL,
                threshold=THRESHOLD,
                fee_rate=FEE_RATE,
                timestamp_col=params.timestamp_col,
                target_col=params.target_col,
                feature_cols=params.feature_cols,
                use_future_covariates=params.use_future_covariates,
                model_path=params.model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=params.batch_size,
                cross_learning=params.cross_learning,
            )
            row = dict(summary)
            row["run_name"] = run_name
            rows.append(row)

            pd.DataFrame([row]).to_csv(os.path.join(out_dir, "summary_row.csv"), index=False)
            print(f"[sweep2]   ✓ finished {run_name}", flush=True)
        except Exception:
            print(f"[sweep2]   ✗ FAILED {run_name}", flush=True)
            traceback.print_exc()

    df_summary = pd.DataFrame(rows)
    master_csv_path = os.path.join(params.results_root, f"summary_ctx{params.context_length}.csv")
    df_summary.to_csv(master_csv_path, index=False)
    print(f"[sweep2] Wrote master CSV: {master_csv_path}", flush=True)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("[sweep2] HEAD:")
        print(df_summary.head())

    print("[sweep2] FINISHED", flush=True)
    return df_summary


def main():
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    params = resolve_params(cli_args)
    sweep(params)


if __name__ == "__main__":
    main()
