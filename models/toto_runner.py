"""
toto_runner.py — single experiment runner for Toto forecasts.

Pipeline (mirrors other runners):
  1) Load data
  2) Build rolling contexts (ctx=512)
  3) Load Toto model + forecaster
  4) Forecast chosen horizon (H in {1, 3, 5, 20})
  5) Save toto_long / toto_wide / toto_step1 CSVs
  6) Evaluate metrics + save equity_no_tc plot
  7) Save JSON summary + one-row CSV
"""

import os
import sys
import json
import time
import faulthandler
import traceback
from typing import Dict, Any, Tuple, List, Optional

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
TOTO_ROOT = r'/Users/jan/Documents/working papers/project 1/TOTO'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if TOTO_ROOT not in sys.path:
    sys.path.insert(0, TOTO_ROOT)

from utils.thresholds import DEFAULT_THRESHOLD
from utils.metrics import evaluate_forecast
from utils.metrics_sweep import write_threshold_sweeps
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto


USER_DEFAULTS_RUNNER = {
    "data_csv": "/Users/jan/Documents/working papers/project 1/data/btc_1h_test.csv",
    "results_root": "/Users/jan/Documents/working papers/project 1/toto-results",

    "context_length": 512,
    "horizon": 1,
    "batch_size": 8,

    "starting_capital": 100_000.0,
    "threshold": DEFAULT_THRESHOLD,
    "fee_rate": 0.001,

    "timestamp_col": "timestamp",
    "target_col": "close",
    "feature_cols": ["open", "high", "low", "volume"],
    "model_path": "Datadog/Toto-Open-Base-1.0",
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


def _parse_feature_cols(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    feature_cols: Optional[List[str]],
) -> List[str]:
    if feature_cols is None:
        excluded = {timestamp_col, target_col}
        feature_cols = [c for c in df.columns if c not in excluded]
    feature_cols = [c for c in feature_cols if c in df.columns and c not in (timestamp_col, target_col)]
    return feature_cols


# ==============
# I/O + data prep
# ==============
def load_data(
    data_csv: str,
    timestamp_col: str,
    target_col: str,
    feature_cols: Optional[List[str]],
    context_length: int,
    horizon: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, List[str]]:
    print("[runner] 1/6 loading data...", flush=True)
    df = pd.read_csv(data_csv)
    if timestamp_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Expected {timestamp_col=} and {target_col=} in CSV columns, found: {list(df.columns)}")

    df = df.rename(columns={timestamp_col: "timestamp", target_col: "target"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")

    feature_cols = _parse_feature_cols(df, timestamp_col="timestamp", target_col="target", feature_cols=feature_cols)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    drop_subset = ["timestamp", "target"] + feature_cols
    df = df.dropna(subset=drop_subset).sort_values("timestamp").reset_index(drop=True)

    vals = df["target"].to_numpy(np.float32)
    ts = df["timestamp"].to_numpy()
    feature_vals = [df[col].to_numpy(np.float32) for col in feature_cols]
    series_vals = np.vstack([vals] + feature_vals) if feature_vals else vals[None, :]

    need = context_length + horizon
    if len(vals) < need:
        raise ValueError(f"Not enough rows for CONTEXT+HORIZON: need {need}, have {len(vals)}")

    ts_ns = pd.to_datetime(ts).view("i8")
    ts_seconds = (ts_ns // 1_000_000_000).astype(np.int64)

    if len(ts_seconds) >= 2:
        diffs = np.diff(ts_seconds)
        median_step = int(np.nanmedian(diffs)) if np.all(np.isfinite(diffs)) else 0
    else:
        median_step = 0

    interval_seconds = int(median_step) if median_step > 0 else 3600

    print(
        f"[runner]   data ready: {len(df)} rows | rolling windows: {len(vals) - need + 1} | variates: {series_vals.shape[0]}",
        flush=True,
    )
    return df, ts, vals, series_vals, ts_seconds, interval_seconds, feature_cols


# ==============
# Model + forecaster
# ==============
def build_forecaster(
    model_path: str,
    device: str,
) -> TotoForecaster:
    print("[runner] 2/6 loading Toto model...", flush=True)
    model = Toto.from_pretrained(model_path)
    model = model.to(device)
    return TotoForecaster(model.model)


# ==============
# Forecast helpers
# ==============
def run_forecast_loop(
    forecaster: TotoForecaster,
    target_vals: np.ndarray,
    series_vals: np.ndarray,
    timestamps: np.ndarray,
    ts_seconds: np.ndarray,
    interval_seconds: int,
    context_length: int,
    horizon: int,
    batch_size: int,
    device: str,
) -> Tuple[pd.DataFrame, float]:
    print("[runner] 3/6 forecasting...", flush=True)
    y_true: List[float] = []
    y_pred: List[float] = []
    base_ts: List[np.datetime64] = []
    forecast_ts: List[np.datetime64] = []

    start = context_length
    stop = len(target_vals) - horizon + 1
    idxs = np.arange(start, stop, dtype=int)
    n_variates = series_vals.shape[0]
    target_idx = 0

    t0 = time.perf_counter()
    for batch_start in range(0, len(idxs), batch_size):
        batch_idxs = idxs[batch_start: batch_start + batch_size]

        ctx_vals = np.stack([series_vals[:, i - context_length: i] for i in batch_idxs], axis=0)
        ctx_ts_base = np.stack([ts_seconds[i - context_length: i] for i in batch_idxs], axis=0)
        ctx_ts = np.repeat(ctx_ts_base[:, None, :], n_variates, axis=1)

        series = torch.from_numpy(ctx_vals).to(dtype=torch.float32)
        padding_mask = torch.ones_like(series, dtype=torch.bool)
        id_mask = torch.zeros_like(series, dtype=torch.int)
        ts_tensor = torch.from_numpy(ctx_ts).to(dtype=torch.int64)
        interval_tensor = torch.full((series.shape[0], n_variates), interval_seconds, dtype=torch.int64)

        inputs = MaskedTimeseries(
            series=series,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=ts_tensor,
            time_interval_seconds=interval_tensor,
        ).to(torch.device(device))

        with torch.no_grad():
            outputs = forecaster.forecast(
                inputs=inputs,
                prediction_length=horizon,
                num_samples=None,
                samples_per_batch=10,
                use_kv_cache=True,
            )
            pred = outputs.mean[:, target_idx, horizon - 1].detach().cpu().numpy()

        y_true.extend(target_vals[batch_idxs + horizon - 1].tolist())
        y_pred.extend(pred.tolist())
        base_ts.extend(timestamps[batch_idxs - 1].tolist())
        forecast_ts.extend(timestamps[batch_idxs + horizon - 1].tolist())

        if batch_start % max(batch_size * 20, 1) == 0:
            print(f"[runner]   processed {batch_start}/{len(idxs)} contexts", flush=True)

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
    print("[runner] 4/6 building outputs...", flush=True)
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
    print("[runner] 5/6 evaluating and plotting...", flush=True)
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
        plt.savefig(os.path.join(output_dir, "toto_equity_no_tc.png"), dpi=150)
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
    starting_capital: float,
    threshold: float,
    fee_rate: float,
    model_path: str,
    batch_size: int,
    feature_cols: List[str],
) -> Dict[str, Any]:
    print("[runner] 6/6 saving artifacts...", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    df_long.to_csv(os.path.join(output_dir, "toto_long.csv"), index=False)
    df_wide.to_csv(os.path.join(output_dir, "toto_wide.csv"), index=False)
    df_step1.to_csv(os.path.join(output_dir, "toto_step1.csv"), index=False)

    summary = {
        "context_length": context_length,
        "horizon": horizon,
        "batch_size": batch_size,
        "starting_capital": starting_capital,
        "threshold": threshold,
        "fee_rate": fee_rate,
        "runtime_seconds": float(runtime_seconds),
        "model_path": model_path,
        "feature_cols": feature_cols,
        "n_variates": int(1 + len(feature_cols)),
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
        write_threshold_sweeps(output_dir, model="toto")
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
    batch_size: int,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
    timestamp_col: str,
    target_col: str,
    feature_cols: Optional[List[str]],
    model_path: str,
    device: str,
) -> Dict[str, Any]:
    if horizon not in (1, 3, 5, 20):
        raise ValueError(f"Horizon must be one of (1, 3, 5, 20); got {horizon}")

    print(f"[runner] API run_single_experiment -> {output_dir}", flush=True)

    _, ts, target_vals, series_vals, ts_seconds, interval_seconds, feature_cols = load_data(
        data_csv=data_csv,
        timestamp_col=timestamp_col,
        target_col=target_col,
        feature_cols=feature_cols,
        context_length=context_length,
        horizon=horizon,
    )

    forecaster = build_forecaster(
        model_path=model_path,
        device=device,
    )

    df_pred, runtime_seconds = run_forecast_loop(
        forecaster=forecaster,
        target_vals=target_vals,
        series_vals=series_vals,
        timestamps=ts,
        ts_seconds=ts_seconds,
        interval_seconds=interval_seconds,
        context_length=context_length,
        horizon=horizon,
        batch_size=batch_size,
        device=device,
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
        starting_capital=starting_capital,
        threshold=threshold,
        fee_rate=fee_rate,
        model_path=model_path,
        batch_size=batch_size,
        feature_cols=feature_cols,
    )
    return summary


# =====================================================
# Standalone entry point (VS Code Run)
# =====================================================
def main():
    faulthandler.enable()

    params = dict(**USER_DEFAULTS_RUNNER)

    run_name = f"ctx{params['context_length']}_h{params['horizon']}"
    output_dir = os.path.join(params["results_root"], run_name)
    if os.path.exists(output_dir):
        from datetime import datetime
        output_dir += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    print("[runner] START")
    print("[runner] data_csv     =", params["data_csv"])
    print("[runner] results_root =", params["results_root"])
    print("[runner] output_dir   =", output_dir)
    print("[runner] ctx,hzn      =", params["context_length"], params["horizon"])
    print("[runner] model        =", params["model_path"])
    print("[runner] batch_size   =", params["batch_size"])
    print("[runner] device       =", params["device"])

    assert os.path.exists(params["data_csv"]), "data_csv not found"

    try:
        summary = run_single_experiment(
            data_csv=params["data_csv"],
            output_dir=output_dir,
            context_length=params["context_length"],
            horizon=params["horizon"],
            batch_size=params["batch_size"],
            starting_capital=params["starting_capital"],
            threshold=params["threshold"],
            fee_rate=params["fee_rate"],
            timestamp_col=params["timestamp_col"],
            target_col=params["target_col"],
            feature_cols=params["feature_cols"],
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
