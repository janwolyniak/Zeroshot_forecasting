"""
timesfm_runner.py — single experiment runner for TimesFM forecasts.

Pipeline:
  1) Load data
  2) Build rolling contexts
  3) Load TimesFM 2.5 from local HF snapshot (config.json + model.safetensors)
  4) Forecast (timed)
  5) Build long / wide / step1 CSVs
  6) Evaluate metrics + save equity_no_tc plot
  7) Save JSON summary (scalarized) + one-row CSV

This file can be:
  • run directly in VS Code (Run Python File in Terminal),
  • imported by sweep_timesfm.py via run_single_experiment(...).
"""

import os
import sys
import json
import time
import faulthandler
import traceback
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ----------------------------
# Make sure project imports work (absolute raw paths for your setup)
# ----------------------------
SCRIPT_DIR = r'/Users/jan/Documents/working papers/project 1/models'
PROJECT_ROOT = r'/Users/jan/Documents/working papers/project 1'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils.thresholds import DEFAULT_TIMESFM_THRESHOLD
from utils.metrics import evaluate_forecast  
import timesfm


# =======================
# VS CODE RUN DEFAULTS
# =======================
USER_DEFAULTS_RUNNER = {
    "data_csv": "/Users/jan/Documents/working papers/project 1/data/btc_1h_test.csv",
    "ckpt_dir": "/Users/jan/.cache/huggingface/hub/models--google--timesfm-2.5-200m-pytorch/snapshots/1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
    "results_root": "/Users/jan/Documents/working papers/project 1/timesfm-results",

    "context_length": 512,
    "horizon": 1,
    "normalize_inputs": True,

    "starting_capital": 100_000.0,
    "threshold": DEFAULT_TIMESFM_THRESHOLD,
    "fee_rate": 0.0,

    "timestamp_col": "timestamp",
    "target_col": "close",
}


# =======================
# Helpers
# =======================
def _json_safe(x):
    """Convert metrics values to JSON-safe scalars where possible."""
    import numpy as _np
    import pandas as _pd
    if isinstance(x, (_np.floating, _np.integer)):
        return float(x)
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    # Time series / arrays → store final value (scalar) to keep JSON small and serializable
    if isinstance(x, (_pd.Series, _pd.DataFrame, _np.ndarray, list, tuple)):
        try:
            if hasattr(x, "iloc"):
                return float(x.iloc[-1])
            return float(x[-1])
        except Exception:
            return None
    # Fallback: string representation (last resort)
    try:
        return float(x)  # might be a numpy scalar subtype
    except Exception:
        return str(type(x).__name__)


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
    df = df.dropna(subset=["timestamp", "target"]).sort_values("timestamp").reset_index(drop=True)

    vals = df["target"].to_numpy(np.float32)
    ts = df["timestamp"].to_numpy()

    need = context_length + horizon + 1
    if len(vals) < need:
        raise ValueError(f"Not enough rows for CONTEXT+HORIZON: need {need}, have {len(vals)}")

    # base index i -> window vals[i-context_length+1 : i+1]
    idxs = np.arange(context_length - 1, len(vals) - horizon)
    contexts: List[np.ndarray] = [vals[i - context_length + 1 : i + 1] for i in idxs]

    print(f"[runner]   data ready: {len(df)} rows | windows to forecast: {len(contexts)}", flush=True)
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

    model.compile(forecast_cfg, torch_compile=torch_compile_flag)
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

    # wide format (preds & trues by horizon columns)
    df_wide = pd.DataFrame({"base_timestamp": base_ts})
    for h in range(H):
        df_wide[f"y_pred_h{h+1}"] = point[:, h]
    # true values into columns
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

    # Plot equity_no_tc
    equity_no_tc = results.get("equity_no_tc")
    if equity_no_tc is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(equity_no_tc.index, equity_no_tc.values, label="Equity (No TC)", linewidth=1.8)
        plt.title("Equity Curve Without Transaction Costs")
        plt.xlabel("Time")
        plt.ylabel("Equity")
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
) -> Dict[str, Any]:
    print("[runner] 6/6 saving artifacts…", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    # CSVs
    df_long.to_csv(os.path.join(output_dir, "timesfm_long.csv"), index=False)
    df_wide.to_csv(os.path.join(output_dir, "timesfm_wide.csv"), index=False)
    df_step1.to_csv(os.path.join(output_dir, "timesfm_step1.csv"), index=False)

    # JSON-able summary (scalarized)
    summary = {
        "context_length": context_length,
        "horizon": horizon,
        "normalize_inputs": bool(normalize_inputs),
        "starting_capital": starting_capital,
        "threshold": threshold,
        "fee_rate": fee_rate,
        "runtime_seconds": float(runtime_seconds),
    }
    # Attach metrics safely
    for k, v in (metrics or {}).items():
        summary[k] = _json_safe(v)

    # Write JSON (never crash saving)
    try:
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        print("[runner] WARNING: metrics.json write failed; continuing.", flush=True)
        traceback.print_exc()

    # Always also write a one-row CSV summary
    try:
        pd.DataFrame([summary]).to_csv(os.path.join(output_dir, "summary_row.csv"), index=False)
    except Exception:
        print("[runner] WARNING: summary_row.csv write failed; continuing.", flush=True)
        traceback.print_exc()

    print("[runner] done.", flush=True)
    return summary


# =====================================================
# Public API for sweeper and programmatic use
# =====================================================
def run_single_experiment(
    *,
    data_csv: str,
    ckpt_dir: str,
    output_dir: str,
    context_length: int,
    horizon: int,
    normalize_inputs: bool,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
    timestamp_col: str,
    target_col: str,
) -> Dict[str, Any]:
    print(f"[runner] API run_single_experiment → {output_dir}", flush=True)

    df, ts, vals, idxs, contexts = load_data(
        data_csv=data_csv,
        timestamp_col=timestamp_col,
        target_col=target_col,
        context_length=context_length,
        horizon=horizon,
    )

    model = load_timesfm_model(
        ckpt_dir=ckpt_dir,
        context_length=context_length,
        horizon=horizon,
        normalize_inputs=normalize_inputs,
        torch_compile_flag=None,
    )

    point, _, runtime_seconds = run_forecast(model, contexts, horizon)

    df_long, df_wide, df_step1 = build_output_dfs(point, ts, vals, idxs, horizon)

    metrics = evaluate_and_plot(
        df_step1=df_step1,
        output_dir=output_dir,
        starting_capital=starting_capital,
        threshold=threshold,
        fee_rate=fee_rate,
    )

    summary = save_artifacts(
        output_dir=output_dir,
        df_long=df_long,
        df_wide=df_wide,
        df_step1=df_step1,
        metrics=metrics,
        runtime_seconds=runtime_seconds,
        context_length=context_length,
        horizon=horizon,
        normalize_inputs=normalize_inputs,
        starting_capital=starting_capital,
        threshold=threshold,
        fee_rate=fee_rate,
    )
    return summary


# =====================================================
# Standalone VS Code entry point
# =====================================================
def main():
    faulthandler.enable()

    # Fill from defaults (you can still add argparse if you want)
    params = dict(**USER_DEFAULTS_RUNNER)

    # Per-run subfolder
    run_name = f"ctx{params['context_length']}_h{params['horizon']}_norm{params['normalize_inputs']}"
    output_dir = os.path.join(params["results_root"], run_name)
    if os.path.exists(output_dir):
        from datetime import datetime
        output_dir += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    print("[runner] START")
    print("[runner] data_csv     =", params["data_csv"])
    print("[runner] ckpt_dir     =", params["ckpt_dir"])
    print("[runner] results_root =", params["results_root"])
    print("[runner] output_dir   =", output_dir)
    print("[runner] ctx,hzn      =", params["context_length"], params["horizon"])
    print("[runner] normalize    =", params["normalize_inputs"])

    # Early existence checks
    assert os.path.exists(params["data_csv"]), "data_csv not found"
    assert os.path.exists(os.path.join(params["ckpt_dir"], "config.json")), "config.json missing in ckpt_dir"
    assert os.path.exists(os.path.join(params["ckpt_dir"], "model.safetensors")), "model.safetensors missing in ckpt_dir"

    try:
        summary = run_single_experiment(
            data_csv=params["data_csv"],
            ckpt_dir=params["ckpt_dir"],
            output_dir=output_dir,
            context_length=params["context_length"],
            horizon=params["horizon"],
            normalize_inputs=params["normalize_inputs"],
            starting_capital=params["starting_capital"],
            threshold=params["threshold"],
            fee_rate=params["fee_rate"],
            timestamp_col=params["timestamp_col"],
            target_col=params["target_col"],
        )
        print("[runner] FINISHED")
        print(json.dumps(summary, indent=2))
    except Exception:
        print("[runner] ERROR occurred:", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
