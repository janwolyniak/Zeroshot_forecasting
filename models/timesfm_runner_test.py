"""
timesfm_runner.py  — single experiment runner with loud logs for VS Code "Run".

What it does:
  1) Loads 1H data (timestamp, close)
  2) Builds rolling context windows
  3) Loads TimesFM 2.5 from an offline HF snapshot (config.json + model.safetensors)
  4) Times ONLY the forecast call
  5) Builds long/wide/step1 CSVs
  6) Evaluates metrics (utils.metrics.evaluate_forecast)
  7) Saves equity_no_tc.png + metrics.json + summary_row.csv

Run in VS Code:
  - Edit USER_DEFAULTS_RUNNER (paths + params)
  - Right-click → "Run Python File in Terminal"
    (or use a proper Python launch config)

Run in terminal:
  python timesfm_runner.py \
    --context_length 512 \
    --horizon 1 \
    --normalize_inputs True \
    --data_csv ../data/btc_1h_test.csv \
    --ckpt_dir /path/to/hf_snapshot_dir \
    --output_dir ../timesfm-results/test_ctx512_h1_normTrue
"""

import os
import sys
import json
import time
import faulthandler
import traceback
import argparse
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

# Optional: comment out to silence TF32 warning on non-CUDA machines
import torch
# torch.set_float32_matmul_precision("high")  # uncomment if you want it
from safetensors.torch import load_file

# Non-interactive backend for saving plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# --- Ensure project root on sys.path so `utils` is importable ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.metrics import evaluate_forecast  # relies on utils/__init__.py being present
import timesfm


# =======================
# VS CODE RUN DEFAULTS
# =======================
USER_DEFAULTS_RUNNER = {
    # EDIT THESE PATHS:
    "data_csv": "/Users/jan/Documents/working papers/project 1/data/btc_1h_test.csv",
    "ckpt_dir": "/Users/jan/.cache/huggingface/hub/models--google--timesfm-2.5-200m-pytorch/snapshots/1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
    "output_dir": "/Users/jan/Documents/working papers/project 1/timesfm-results/test_ctx512_h1_normTrue",

    # Experiment params:
    "context_length": 512,
    "horizon": 1,
    "normalize_inputs": True,

    # Trading / evaluation params:
    "starting_capital": 100_000.0,
    "threshold": 0.01,
    "fee_rate": 0.001,

    # Data column names:
    "timestamp_col": "timestamp",
    "target_col": "close",
}

# Optional: cap number of rolling windows to forecast for a quick smoke test.
# Set to an integer (e.g., 1 or 4) for a fast run, or None to process ALL windows.
SMOKE_CAP = None  # e.g., 4 for very fast test, then set to None for full run


# ==============
# I/O + data prep
# ==============
def load_data(
    data_csv: str,
    timestamp_col: str,
    target_col: str,
    context_length: int,
    horizon: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    print("[runner] 1/6 loading data…", flush=True)
    df = pd.read_csv(data_csv, usecols=[timestamp_col, target_col]).rename(
        columns={timestamp_col: "timestamp", target_col: "target"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = (
        df.dropna(subset=["timestamp", "target"])
          .sort_values("timestamp")
          .reset_index(drop=True)
    )

    vals = df["target"].to_numpy(np.float32)
    ts = df["timestamp"].to_numpy()

    need = context_length + horizon + 1
    if len(vals) < need:
        raise ValueError(
            f"Not enough rows for CONTEXT+HORIZON: need {need}, have {len(vals)}"
        )

    # base index i -> window vals[i-context_length+1 : i+1]
    idxs = np.arange(context_length - 1, len(vals) - horizon)
    contexts: List[np.ndarray] = [
        vals[i - context_length + 1 : i + 1] for i in idxs
    ]

    if SMOKE_CAP is not None and len(contexts) > SMOKE_CAP:
        contexts = contexts[-SMOKE_CAP:]
        idxs = idxs[-SMOKE_CAP:]

    print(
        f"[runner]   data ready: {len(df)} rows | windows to forecast: {len(contexts)}",
        flush=True,
    )
    return df, ts, vals, idxs, contexts


# ==============
# Model loading
# ==============
def load_timesfm_model(
    ckpt_dir: str,
    context_length: int,
    horizon: int,
    normalize_inputs: bool,
    torch_compile_flag: bool = None,
):
    print("[runner] 2/6 building model…", flush=True)

    cfg_path = os.path.join(ckpt_dir, "config.json")
    weights_path = os.path.join(ckpt_dir, "model.safetensors")

    with open(cfg_path, "r") as f:
        cfg_raw = json.load(f)

    model = timesfm.TimesFM_2p5_200M_torch(**cfg_raw)

    core = model.model
    state_dict = load_file(weights_path, device="cpu")
    missing, unexpected = core.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[runner]   WARNING: missing keys: {len(missing)}")
    if unexpected:
        print(f"[runner]   WARNING: unexpected keys: {len(unexpected)}")

    forecast_cfg = timesfm.ForecastConfig(
        max_context=max(context_length, 32),
        max_horizon=max(horizon, 1),
        normalize_inputs=bool(normalize_inputs),
        use_continuous_quantile_head=False,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )

    if torch_compile_flag is None:
        torch_compile_flag = torch.cuda.is_available()

    model.compile(
        forecast_cfg,
        torch_compile=torch_compile_flag,
    )

    print("[runner]   model compiled.", flush=True)
    return model


# ==============
# Forecast
# ==============
def run_forecast(
    model,
    contexts: List[np.ndarray],
    horizon: int,
) -> Tuple[np.ndarray, Any, float]:
    print("[runner] 3/6 forecasting…", flush=True)
    t0 = time.perf_counter()
    point, quant = model.forecast(horizon=horizon, inputs=contexts)
    runtime_seconds = time.perf_counter() - t0
    print(f"[runner]   forecast done in {runtime_seconds:.3f}s", flush=True)
    return point, quant, runtime_seconds


# ==============
# Outputs
# ==============
def build_output_dfs(
    point: np.ndarray,
    ts: np.ndarray,
    vals: np.ndarray,
    idxs: np.ndarray,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[runner] 4/6 building outputs…", flush=True)
    N, H = point.shape
    if H != horizon:
        raise ValueError(f"horizon mismatch: expected {horizon}, got {H}")

    base_ts = ts[idxs]  # [N]
    future_ts = [ts[i + 1 : i + 1 + H] for i in idxs]
    true_vals = [vals[i + 1 : i + 1 + H] for i in idxs]

    # long format
    rows = []
    for n in range(N):
        for h in range(H):
            rows.append((
                base_ts[n],
                h + 1,
                future_ts[n][h],
                float(point[n, h]),
                float(true_vals[n][h]),
            ))
    df_long = pd.DataFrame(
        rows,
        columns=["base_timestamp", "horizon_step", "forecast_timestamp", "y_pred", "y_true"],
    )

    # wide format
    df_wide = pd.DataFrame({"base_timestamp": base_ts})
    for h in range(H):
        df_wide[f"y_pred_h{h+1}"] = point[:, h]
    for h in range(H):
        df_wide[f"y_true_h{h+1}"] = [float(x[h]) for x in true_vals]

    # step1 slice
    df_step1 = (
        df_long[df_long["horizon_step"] == 1][["forecast_timestamp", "y_true", "y_pred"]]
        .rename(columns={"forecast_timestamp": "timestamp"})
        .reset_index(drop=True)
    )

    return df_long, df_wide, df_step1


def evaluate_and_plot(
    df_step1: pd.DataFrame,
    output_dir: str,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
) -> Dict[str, Any]:
    print("[runner] 5/6 evaluating and plotting…", flush=True)
    results = evaluate_forecast(
        y_true=df_step1["y_true"],
        y_pred=df_step1["y_pred"],
        timestamps=df_step1["timestamp"],
        starting_capital=starting_capital,
        threshold=threshold,
        fee_rate=fee_rate,
    )

    equity_no_tc = results["equity_no_tc"]

    plt.figure(figsize=(10, 5))
    plt.plot(
        equity_no_tc.index,
        equity_no_tc.values,
        label="Equity (No TC)",
        linewidth=1.8,
    )
    plt.title("Equity Curve Without Transaction Costs")
    plt.xlabel("Time")
    plt.ylabel("Equity Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    plt.ticklabel_format(style="plain", axis="y")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "equity_no_tc.png"), dpi=150)
    plt.close()

    return results


def _json_safe(x):
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def save_artifacts(
    output_dir: str,
    df_long: pd.DataFrame,
    df_wide: pd.DataFrame,
    df_step1: pd.DataFrame,
    metrics: Dict[str, Any],
    runtime_seconds: float,
    context_length: int,
    horizon: int,
    normalize_inputs: bool,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
):
    print("[runner] 6/6 saving artifacts…", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    df_long.to_csv(os.path.join(output_dir, "timesfm_long.csv"), index=False)
    df_wide.to_csv(os.path.join(output_dir, "timesfm_wide.csv"), index=False)
    df_step1.to_csv(os.path.join(output_dir, "timesfm_step1.csv"), index=False)

    summary = {
        "context_length": context_length,
        "horizon": horizon,
        "normalize_inputs": bool(normalize_inputs),
        "starting_capital": starting_capital,
        "threshold": threshold,
        "fee_rate": fee_rate,
        "runtime_seconds": runtime_seconds,

        "final_equity_no_tc": metrics.get("final_equity_no_tc"),
        "final_equity_tc": metrics.get("final_equity_tc"),
        "total_fees_paid": metrics.get("total_fees_paid"),

        "arc": metrics.get("arc"),
        "asd": metrics.get("asd"),
        "ir_star": metrics.get("ir_star"),
        "ir_starstar": metrics.get("ir_starstar"),
        "mdd": metrics.get("mdd"),

        "pct_long": metrics.get("pct_long"),
        "pct_short": metrics.get("pct_short"),
        "pct_flat": metrics.get("pct_flat"),

        "trades_long": metrics.get("trades_long"),
        "trades_short": metrics.get("trades_short"),
        "trades_flat": metrics.get("trades_flat"),
        "trades_total": metrics.get("trades_total"),
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_safe)

    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "summary_row.csv"), index=False
    )

    print("[runner] done.", flush=True)
    return summary


# ==================
# Orchestration
# ==================
def build_arg_parser():
    p = argparse.ArgumentParser(description="Run a single TimesFM forecast experiment.")
    p.add_argument("--data_csv", type=str)
    p.add_argument("--ckpt_dir", type=str)
    p.add_argument("--output_dir", type=str)

    p.add_argument("--context_length", type=int)
    p.add_argument("--horizon", type=int)
    p.add_argument("--normalize_inputs", type=lambda s: s.lower() in ["true", "1", "yes"])

    p.add_argument("--starting_capital", type=float)
    p.add_argument("--threshold", type=float)
    p.add_argument("--fee_rate", type=float)

    p.add_argument("--timestamp_col", type=str)
    p.add_argument("--target_col", type=str)
    return p


def resolve_params_runner(cli_args: argparse.Namespace) -> argparse.Namespace:
    merged = argparse.Namespace()

    merged.data_csv = cli_args.data_csv or USER_DEFAULTS_RUNNER["data_csv"]
    merged.ckpt_dir = cli_args.ckpt_dir or USER_DEFAULTS_RUNNER["ckpt_dir"]
    merged.output_dir = cli_args.output_dir or USER_DEFAULTS_RUNNER["output_dir"]

    merged.context_length = cli_args.context_length or USER_DEFAULTS_RUNNER["context_length"]
    merged.horizon = cli_args.horizon or USER_DEFAULTS_RUNNER["horizon"]

    if cli_args.normalize_inputs is None:
        merged.normalize_inputs = USER_DEFAULTS_RUNNER["normalize_inputs"]
    else:
        merged.normalize_inputs = cli_args.normalize_inputs

    merged.starting_capital = (
        cli_args.starting_capital
        if cli_args.starting_capital is not None
        else USER_DEFAULTS_RUNNER["starting_capital"]
    )
    merged.threshold = (
        cli_args.threshold
        if cli_args.threshold is not None
        else USER_DEFAULTS_RUNNER["threshold"]
    )
    merged.fee_rate = (
        cli_args.fee_rate
        if cli_args.fee_rate is not None
        else USER_DEFAULTS_RUNNER["fee_rate"]
    )

    merged.timestamp_col = cli_args.timestamp_col or USER_DEFAULTS_RUNNER["timestamp_col"]
    merged.target_col = cli_args.target_col or USER_DEFAULTS_RUNNER["target_col"]
    return merged


def main():
    # show Python crashes instead of silent exits
    faulthandler.enable()

    # parse (may be empty when using VS Code Run)
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    params = resolve_params_runner(cli_args)

    # loud start + early path checks
    print("[runner] START", flush=True)
    print("[runner] data_csv   =", params.data_csv)
    print("[runner] ckpt_dir   =", params.ckpt_dir)
    print("[runner] output_dir =", params.output_dir)
    print("[runner] ctx,hzn    =", params.context_length, params.horizon)
    print("[runner] normalize  =", params.normalize_inputs)

    assert os.path.exists(params.data_csv), "data_csv not found"
    assert os.path.exists(os.path.join(params.ckpt_dir, "config.json")), "config.json missing in ckpt_dir"
    assert os.path.exists(os.path.join(params.ckpt_dir, "model.safetensors")), "model.safetensors missing in ckpt_dir"

    os.makedirs(params.output_dir, exist_ok=True)

    try:
        df, ts, vals, idxs, contexts = load_data(
            data_csv=params.data_csv,
            timestamp_col=params.timestamp_col,
            target_col=params.target_col,
            context_length=params.context_length,
            horizon=params.horizon,
        )

        model = load_timesfm_model(
            ckpt_dir=params.ckpt_dir,
            context_length=params.context_length,
            horizon=params.horizon,
            normalize_inputs=params.normalize_inputs,
            torch_compile_flag=None,
        )

        point, quant, runtime_seconds = run_forecast(
            model=model,
            contexts=contexts,
            horizon=params.horizon,
        )

        df_long, df_wide, df_step1 = build_output_dfs(
            point=point,
            ts=ts,
            vals=vals,
            idxs=idxs,
            horizon=params.horizon,
        )

        metrics = evaluate_and_plot(
            df_step1=df_step1,
            output_dir=params.output_dir,
            starting_capital=params.starting_capital,
            threshold=params.threshold,
            fee_rate=params.fee_rate,
        )

        save_artifacts(
            output_dir=params.output_dir,
            df_long=df_long,
            df_wide=df_wide,
            df_step1=df_step1,
            metrics=metrics,
            runtime_seconds=runtime_seconds,
            context_length=params.context_length,
            horizon=params.horizon,
            normalize_inputs=params.normalize_inputs,
            starting_capital=params.starting_capital,
            threshold=params.threshold,
            fee_rate=params.fee_rate,
        )

    except Exception:
        print("[runner] ERROR occurred:", flush=True)
        traceback.print_exc()
        raise

    print("[runner] FINISHED", flush=True)


if __name__ == "__main__":
    main()