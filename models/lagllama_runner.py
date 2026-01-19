"""
lagllama_runner.py — single experiment runner for Lag-Llama zero-shot forecasts.

Pipeline:
  1) Load data
  2) Build rolling contexts (ctx=512)
  3) Load Lag-Llama core once (H in {1,3,5,20} via autoregressive rollout)
  4) Forecast chosen horizon (mean point forecasts)
  5) Save lagllama_long / lagllama_wide / lagllama_step1 CSVs
  6) Evaluate metrics + save equity_no_tc plot
  7) Save JSON summary + one-row CSV
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

SCRIPT_DIR = r'/Users/jan/Documents/working papers/project 1/models'
PROJECT_ROOT = r'/Users/jan/Documents/working papers/project 1'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils.thresholds import DEFAULT_THRESHOLD
from utils.metrics import evaluate_forecast
from utils.metrics_sweep import write_threshold_sweeps
import lagllama_no_gluonts as ll


USER_DEFAULTS_RUNNER = {
    "data_csv": "/Users/jan/Documents/working papers/project 1/data/btc_1h_test.csv",
    "results_root": "/Users/jan/Documents/working papers/project 1/lagllama-results",

    "context_length": 512,
    "horizon": 1,
    "freq": "1H",
    "num_samples": 400,
    "rope_scale": 1.5,

    "starting_capital": 100_000.0,
    "threshold": DEFAULT_THRESHOLD,
    "fee_rate": 0.001,

    "timestamp_col": "timestamp",
    "target_col": "close",
}


def _json_safe(x):
    import numpy as _np
    import pandas as _pd
    if isinstance(x, (_np.floating, _np.integer)):
        return float(x)
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    if isinstance(x, (_pd.Series, _pd.DataFrame, _np.ndarray, list, tuple)):
        try:
            if hasattr(x, "iloc"):
                return float(x.iloc[-1])
            return float(x[-1])
        except Exception:
            return None
    try:
        return float(x)
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
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    print("[runner] 1/6 loading data…", flush=True)
    df = pd.read_csv(data_csv, usecols=[timestamp_col, target_col]).rename(
        columns={timestamp_col: "timestamp", target_col: "target"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "target"]).sort_values("timestamp").reset_index(drop=True)

    vals = df["target"].to_numpy(np.float32)
    ts = df["timestamp"].to_numpy()

    need = context_length + horizon + 1
    if len(vals) < need:
        raise ValueError(f"Not enough rows for CONTEXT+HORIZON: need {need}, have {len(vals)}")

    idxs = np.arange(context_length - 1, len(vals) - horizon)

    print(f"[runner]   data ready: {len(df)} rows | windows to forecast: {len(idxs)}", flush=True)
    return df, ts, vals, idxs


# ==============
# Model loading
# ==============
def load_lagllama_core(rope_scale: float):
    print("[runner] 2/6 loading Lag-Llama core…", flush=True)
    core, model_kwargs = ll.load_core(rope_scale=rope_scale)
    print("[runner]   core ready on", ll.DEVICE, flush=True)
    return core, model_kwargs


# ==============
# Forecast loop
# ==============
def forecast_windows(
    core,
    df: pd.DataFrame,
    ts: np.ndarray,
    vals: np.ndarray,
    idxs: np.ndarray,
    context_length: int,
    horizon: int,
    freq: str,
    num_samples: int,
) -> Tuple[np.ndarray, float]:
    print("[runner] 3/6 forecasting windows…", flush=True)
    N = len(idxs)
    preds = np.zeros((N, horizon), dtype=np.float32)

    t0 = time.perf_counter()
    for j, i in enumerate(idxs):
        # context series with timestamp index
        ctx_slice = slice(i - context_length + 1, i + 1)
        ctx_series = pd.Series(vals[ctx_slice], index=pd.to_datetime(ts[ctx_slice]), dtype="float32")

        fc_df = ll.forecast_with_core(
            core,
            y=ctx_series,
            prediction_length=horizon,
            context_length=context_length,
            freq=freq,
            num_samples=num_samples,
        )
        preds[j, :] = fc_df["mean"].to_numpy(np.float32)

        if j % 200 == 0:
            print(f"[runner]   {j}/{N} windows done", flush=True)

    runtime_seconds = time.perf_counter() - t0
    print(f"[runner]   forecast loop done in {runtime_seconds:.3f}s", flush=True)
    return preds, runtime_seconds


# ==============
# Outputs
# ==============
def build_output_dfs(
    preds: np.ndarray,
    ts: np.ndarray,
    vals: np.ndarray,
    idxs: np.ndarray,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[runner] 4/6 building outputs…", flush=True)
    N, H = preds.shape
    base_ts = ts[idxs]  # [N]
    future_ts = [ts[i + 1 : i + 1 + H] for i in idxs]
    true_vals = [vals[i + 1 : i + 1 + H] for i in idxs]

    rows = []
    for n in range(N):
        for h in range(H):
            rows.append((
                base_ts[n],
                h + 1,
                future_ts[n][h],
                float(preds[n, h]),
                float(true_vals[n][h]),
            ))
    df_long = pd.DataFrame(
        rows,
        columns=["base_timestamp", "horizon_step", "forecast_timestamp", "y_pred", "y_true"],
    )

    df_wide = pd.DataFrame({"base_timestamp": base_ts})
    for h in range(H):
        df_wide[f"y_pred_h{h+1}"] = preds[:, h]
    for h in range(H):
        df_wide[f"y_true_h{h+1}"] = [float(x[h]) for x in true_vals]

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
        plt.savefig(os.path.join(output_dir, "lagllama_equity_no_tc.png"), dpi=150)
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
    freq: str,
    num_samples: int,
    rope_scale: float,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
) -> Dict[str, Any]:
    print("[runner] 6/6 saving artifacts…", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    df_long.to_csv(os.path.join(output_dir, "lagllama_long.csv"), index=False)
    df_wide.to_csv(os.path.join(output_dir, "lagllama_wide.csv"), index=False)
    df_step1.to_csv(os.path.join(output_dir, "lagllama_step1.csv"), index=False)

    summary = {
        "context_length": context_length,
        "horizon": horizon,
        "freq": freq,
        "num_samples": num_samples,
        "rope_scale": rope_scale,
        "starting_capital": starting_capital,
        "threshold": threshold,
        "fee_rate": fee_rate,
        "runtime_seconds": float(runtime_seconds),
    }
    for k, v in (metrics or {}).items():
        summary[k] = _json_safe(v)

    try:
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        print("[runner] WARNING: metrics.json write failed; continuing.", flush=True)
        traceback.print_exc()

    try:
        pd.DataFrame([summary]).to_csv(os.path.join(output_dir, "summary_row.csv"), index=False)
    except Exception:
        print("[runner] WARNING: summary_row.csv write failed; continuing.", flush=True)
        traceback.print_exc()

    try:
        write_threshold_sweeps(output_dir, model="lagllama")
    except Exception:
        print("[runner] WARNING: metrics_sweep generation failed; continuing.", flush=True)
        traceback.print_exc()

    print("[runner] done.", flush=True)
    return summary


# =====================================================
# Public API
# =====================================================
def run_single_experiment(
    *,
    data_csv: str,
    output_dir: str,
    context_length: int,
    horizon: int,
    freq: str,
    num_samples: int,
    rope_scale: float,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
    timestamp_col: str,
    target_col: str,
) -> Dict[str, Any]:
    if horizon not in (1, 3, 5, 20):
        raise ValueError(f"Horizon must be one of (1, 3, 5, 20); got {horizon}")

    print(f"[runner] API run_single_experiment → {output_dir}", flush=True)

    df, ts, vals, idxs = load_data(
        data_csv=data_csv,
        timestamp_col=timestamp_col,
        target_col=target_col,
        context_length=context_length,
        horizon=horizon,
    )

    core, _ = load_lagllama_core(rope_scale=rope_scale)

    preds, runtime_seconds = forecast_windows(
        core=core,
        df=df,
        ts=ts,
        vals=vals,
        idxs=idxs,
        context_length=context_length,
        horizon=horizon,
        freq=freq,
        num_samples=num_samples,
    )

    df_long, df_wide, df_step1 = build_output_dfs(
        preds=preds,
        ts=ts,
        vals=vals,
        idxs=idxs,
        horizon=horizon,
    )

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
        freq=freq,
        num_samples=num_samples,
        rope_scale=rope_scale,
        starting_capital=starting_capital,
        threshold=threshold,
        fee_rate=fee_rate,
    )
    return summary


# =====================================================
# Standalone entry point (VS Code Run)
# =====================================================
def main():
    faulthandler.enable()

    params = dict(**USER_DEFAULTS_RUNNER)

    run_name = f"ctx{params['context_length']}_h{params['horizon']}_samples{params['num_samples']}_rope{params['rope_scale']}"
    output_dir = os.path.join(params["results_root"], run_name)
    if os.path.exists(output_dir):
        from datetime import datetime
        output_dir += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    print("[runner] START")
    print("[runner] data_csv     =", params["data_csv"])
    print("[runner] results_root =", params["results_root"])
    print("[runner] output_dir   =", output_dir)
    print("[runner] ctx,hzn      =", params["context_length"], params["horizon"])
    print("[runner] freq         =", params["freq"])
    print("[runner] num_samples  =", params["num_samples"])
    print("[runner] rope_scale   =", params["rope_scale"])

    assert os.path.exists(params["data_csv"]), "data_csv not found"

    try:
        summary = run_single_experiment(
            data_csv=params["data_csv"],
            output_dir=output_dir,
            context_length=params["context_length"],
            horizon=params["horizon"],
            freq=params["freq"],
            num_samples=params["num_samples"],
            rope_scale=params["rope_scale"],
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
