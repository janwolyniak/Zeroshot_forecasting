"""
ttm_runner.py — single experiment runner for Granite TTM forecasts.

Pipeline (mirrors timesfm_runner style):
  1) Load data
  2) Build rolling contexts (ctx=512)
  3) Load Granite TTM model (ttm-r2) via tsfm_public
  4) Forecast chosen horizon (H in {1, 3, 5, 20})
  5) Save ttm_long / ttm_wide / ttm_step1 CSVs
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

# Absolute paths so VS Code "Run" works without env tweaks
SCRIPT_DIR = r'/Users/jan/Documents/working papers/project 1/models'
PROJECT_ROOT = r'/Users/jan/Documents/working papers/project 1'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils.metrics import evaluate_forecast
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.time_series_forecasting_pipeline import TimeSeriesForecastingPipeline
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor


USER_DEFAULTS_RUNNER = {
    "data_csv": "/Users/jan/Documents/working papers/project 1/data/btc_1h_test.csv",
    "results_root": "/Users/jan/Documents/working papers/project 1/ttm-results",

    "context_length": 512,
    "horizon": 1,
    "prediction_length": 96,  # TTM default; must cover max horizon
    "scale_inputs": False,    # toggle scaling in the preprocessor

    "starting_capital": 100_000.0,
    "threshold": 0.005,
    "fee_rate": 0.001,

    "timestamp_col": "timestamp",
    "target_col": "close",
    "id_col": "series",
    "series_id": "BTCUSD",
    "model_path": "ibm-granite/granite-timeseries-ttm-r2",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def _json_safe(x):
    """Convert values to JSON-safe scalars where possible."""
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
    id_col: str,
    series_id: str,
    context_length: int,
    horizon: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    print("[runner] 1/6 loading data…", flush=True)
    df = pd.read_csv(data_csv, usecols=[timestamp_col, target_col])
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    df = df.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col).reset_index(drop=True)

    df[id_col] = series_id
    df = df[[timestamp_col, id_col, target_col]].copy()

    vals = df[target_col].to_numpy(np.float32)
    ts = df[timestamp_col].to_numpy()

    need = context_length + horizon
    if len(vals) < need:
        raise ValueError(f"Not enough rows for CONTEXT+HORIZON: need {need}, have {len(vals)}")

    print(f"[runner]   data ready: {len(df)} rows | rolling windows: {len(vals) - need + 1}", flush=True)
    return df, ts, vals


# ==============
# Model + pipe
# ==============
def build_pipeline(
    model_path: str,
    context_length: int,
    prediction_length: int,
    scale_inputs: bool,
    timestamp_col: str,
    id_col: str,
    target_col: str,
    device: str,
) -> TimeSeriesForecastingPipeline:
    print("[runner] 2/6 building model + pipeline…", flush=True)
    prediction_length = max(prediction_length, 1)

    column_specifiers = {
        "timestamp_column": timestamp_col,
        "id_columns": [id_col],
        "target_columns": [target_col],
        "control_columns": [],
    }

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=prediction_length,
        scaling=bool(scale_inputs),
        encode_categorical=False,
        freq=None,
    )

    model = get_model(
        model_path=model_path,
        context_length=context_length,
        prediction_length=prediction_length,
        freq=None,
    ).to(device)

    pipe = TimeSeriesForecastingPipeline(
        model=model,
        feature_extractor=tsp,
        device=device,
        batch_size=32,
    )
    return pipe


# ==============
# Forecast helpers
# ==============
def _extract_h_step(forecast_df: pd.DataFrame, h_target: int) -> float:
    df = forecast_df.copy()

    for hcol in ("horizon", "step", "k"):
        if hcol in df.columns:
            df = df[df[hcol].astype(int) == h_target]
            break

    if len(df) > 1 and any(c in df.columns for c in ("horizon", "step", "k")):
        df = df.sort_values(by=[c for c in ("horizon", "step", "k") if c in df.columns]).head(1)

    for pcol in ("point", "mean", "yhat", "prediction", "pred"):
        if pcol in df.columns and not df.empty:
            val = df.iloc[0][pcol]
            if isinstance(val, (list, tuple, np.ndarray)):
                return float(val[h_target - 1])
            return float(val)

    for qcol in ("p50", "median", "q50", "q_0.5"):
        if qcol in df.columns and not df.empty:
            val = df.iloc[0][qcol]
            if isinstance(val, (list, tuple, np.ndarray)):
                return float(val[h_target - 1])
            return float(val)

    for col in df.columns:
        val = df.iloc[0][col]
        if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= h_target:
            return float(val[h_target - 1])

    raise ValueError(f"Could not extract horizon {h_target}; seen columns: {list(forecast_df.columns)}")


def run_forecast_loop(
    pipe: TimeSeriesForecastingPipeline,
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    context_length: int,
    horizon: int,
    roll_step: int = 1,
) -> Tuple[pd.DataFrame, float]:
    print("[runner] 3/6 rolling forecast…", flush=True)
    values = df[target_col].to_numpy(float)
    timestamps = df[timestamp_col].to_numpy()

    y_true: List[float] = []
    y_pred: List[float] = []
    base_ts: List[np.datetime64] = []
    forecast_ts: List[np.datetime64] = []

    start = context_length
    stop = len(values) - horizon + 1

    t0 = time.perf_counter()
    for i in range(start, stop, roll_step):
        ctx_df = df.iloc[i - context_length : i].copy()
        forecast_df = pipe(ctx_df)

        pred_i = _extract_h_step(forecast_df, h_target=horizon)

        y_true.append(values[i + horizon - 1])
        y_pred.append(pred_i)
        base_ts.append(timestamps[i - 1])
        forecast_ts.append(timestamps[i + horizon - 1])

        if (i - start) % 200 == 0:
            print(f"[runner]   processed i={i}/{stop-1}", flush=True)

    runtime_seconds = time.perf_counter() - t0
    print(f"[runner]   forecast loop done in {runtime_seconds:.3f}s", flush=True)

    df_pred = pd.DataFrame({
        "base_timestamp": pd.to_datetime(base_ts),
        "timestamp": pd.to_datetime(forecast_ts),
        "y_true": np.asarray(y_true, dtype=np.float32),
        "y_pred": np.asarray(y_pred, dtype=np.float32),
    })
    return df_pred, runtime_seconds


# ==============
# Outputs
# ==============
def build_output_dfs(
    df_pred: pd.DataFrame,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[runner] 4/6 building outputs…", flush=True)
    df_step1 = df_pred[["timestamp", "y_true", "y_pred"]].copy()

    df_long = df_pred.copy()
    df_long.insert(1, "horizon_step", horizon)
    df_long = df_long.rename(columns={"timestamp": "forecast_timestamp"})

    df_wide = pd.DataFrame({
        "base_timestamp": df_pred["base_timestamp"],
        f"y_pred_h{horizon}": df_pred["y_pred"],
        f"y_true_h{horizon}": df_pred["y_true"],
    })

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
        plt.savefig(os.path.join(output_dir, "ttm_equity_no_tc.png"), dpi=150)
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
    prediction_length: int,
    scale_inputs: bool,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
    model_path: str,
) -> Dict[str, Any]:
    print("[runner] 6/6 saving artifacts…", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    df_long.to_csv(os.path.join(output_dir, "ttm_long.csv"), index=False)
    df_wide.to_csv(os.path.join(output_dir, "ttm_wide.csv"), index=False)
    df_step1.to_csv(os.path.join(output_dir, "ttm_step1.csv"), index=False)

    summary = {
        "context_length": context_length,
        "horizon": horizon,
        "prediction_length": prediction_length,
        "scale_inputs": bool(scale_inputs),
        "starting_capital": starting_capital,
        "threshold": threshold,
        "fee_rate": fee_rate,
        "runtime_seconds": float(runtime_seconds),
        "model_path": model_path,
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
    prediction_length: int,
    scale_inputs: bool,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
    timestamp_col: str,
    target_col: str,
    id_col: str,
    series_id: str,
    model_path: str,
    device: str,
) -> Dict[str, Any]:
    if horizon not in (1, 3, 5, 20):
        raise ValueError(f"Horizon must be one of (1, 3, 5, 20); got {horizon}")

    print(f"[runner] API run_single_experiment → {output_dir}", flush=True)

    df, ts, vals = load_data(
        data_csv=data_csv,
        timestamp_col=timestamp_col,
        target_col=target_col,
        id_col=id_col,
        series_id=series_id,
        context_length=context_length,
        horizon=horizon,
    )

    pipe = build_pipeline(
        model_path=model_path,
        context_length=context_length,
        prediction_length=max(prediction_length, horizon),
        scale_inputs=scale_inputs,
        timestamp_col=timestamp_col,
        id_col=id_col,
        target_col=target_col,
        device=device,
    )

    df_pred, runtime_seconds = run_forecast_loop(
        pipe=pipe,
        df=df,
        timestamp_col=timestamp_col,
        target_col=target_col,
        context_length=context_length,
        horizon=horizon,
        roll_step=1,
    )

    df_long, df_wide, df_step1 = build_output_dfs(df_pred=df_pred, horizon=horizon)

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
        prediction_length=prediction_length,
        scale_inputs=scale_inputs,
        starting_capital=starting_capital,
        threshold=threshold,
        fee_rate=fee_rate,
        model_path=model_path,
    )
    return summary


# =====================================================
# Standalone entry point (VS Code Run)
# =====================================================
def main():
    faulthandler.enable()

    params = dict(**USER_DEFAULTS_RUNNER)
    params["prediction_length"] = max(params["prediction_length"], params["horizon"])

    run_name = f"ctx{params['context_length']}_h{params['horizon']}_scale{params['scale_inputs']}"
    output_dir = os.path.join(params["results_root"], run_name)
    if os.path.exists(output_dir):
        from datetime import datetime
        output_dir += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    print("[runner] START")
    print("[runner] data_csv     =", params["data_csv"])
    print("[runner] results_root =", params["results_root"])
    print("[runner] output_dir   =", output_dir)
    print("[runner] ctx,hzn      =", params["context_length"], params["horizon"])
    print("[runner] scale_inputs =", params["scale_inputs"])
    print("[runner] model        =", params["model_path"])
    print("[runner] device       =", params["device"])

    assert os.path.exists(params["data_csv"]), "data_csv not found"

    try:
        summary = run_single_experiment(
            data_csv=params["data_csv"],
            output_dir=output_dir,
            context_length=params["context_length"],
            horizon=params["horizon"],
            prediction_length=params["prediction_length"],
            scale_inputs=params["scale_inputs"],
            starting_capital=params["starting_capital"],
            threshold=params["threshold"],
            fee_rate=params["fee_rate"],
            timestamp_col=params["timestamp_col"],
            target_col=params["target_col"],
            id_col=params["id_col"],
            series_id=params["series_id"],
            model_path=params["model_path"],
            device=params["device"],
        )
        print("[runner] FINISHED")
        print(json.dumps(summary, indent=2))
    except Exception:
        print("[runner] ERROR occurred:", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
