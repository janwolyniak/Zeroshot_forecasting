
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
import chronos

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

from utils.thresholds import DEFAULT_THRESHOLD
from utils.metrics import evaluate_forecast
from utils.metrics_sweep import write_threshold_sweeps

faulthandler.enable()


# =======================
# VS CODE RUN DEFAULTS
# =======================
USER_DEFAULTS_RUNNER = {
    "data_csv": "/Users/jan/Documents/working papers/project 1/data/btc_1h_test.csv",
    "results_root": "/Users/jan/Documents/working papers/project 1/chronos-results",

    "context_length": 512,
    "horizon": 1,
    "num_samples": 5,

    "starting_capital": 100_000.0,
    "threshold": DEFAULT_THRESHOLD,
    "fee_rate": 0.001,

    "timestamp_col": "timestamp",
    "target_col": "close",
    "model_path": "amazon/chronos-t5-mini",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 4,
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
    context_length: int,
    horizon: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
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
    contexts: List[np.ndarray] = [vals[i - context_length + 1 : i + 1] for i in idxs]

    print(f"[runner]   data ready: {len(df)} rows | windows to forecast: {len(contexts)}", flush=True)
    return df, ts, vals, idxs, contexts


# ==============
# Model loading
# ==============
def load_chronos_pipeline(
    model_path: str,
    device: str,
) -> chronos.ChronosPipeline:
    print("[runner] 2/6 loading ChronosPipeline…", flush=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = chronos.ChronosPipeline.from_pretrained(
        model_path,
        device_map="auto" if device == "cuda" else {"": device},
        dtype=dtype,
    )
    return pipe


# ==============
# Forecast
# ==============
def _reduce_samples(arr: np.ndarray, horizon: int, num_samples: int) -> np.ndarray:
    """Collapse Chronos samples → median forecast."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if num_samples in arr.shape:
            axis = list(arr.shape).index(num_samples)
            return np.median(arr, axis=axis)
        return np.median(arr, axis=1)
    if arr.ndim == 2:
        if arr.shape[0] == num_samples and arr.shape[1] == horizon:
            return np.median(arr, axis=0, keepdims=True)
        return arr
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    raise ValueError(f"Unexpected forecast shape {arr.shape}")


def run_forecast(
    pipe: chronos.ChronosPipeline,
    contexts: List[np.ndarray],
    horizon: int,
    num_samples: int,
    batch_size: int,
) -> Tuple[np.ndarray, float]:
    print("[runner] 3/6 forecasting…", flush=True)
    preds: List[np.ndarray] = []
    t0 = time.perf_counter()

    for start in range(0, len(contexts), batch_size):
        end = start + batch_size
        batch = np.stack(contexts[start:end]).astype(np.float32)
        device = getattr(pipe, "device", None)
        if device is None and hasattr(pipe, "model") and hasattr(pipe.model, "device"):
            device = pipe.model.device
        batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
        samples = pipe.predict(
            batch_tensor,
            prediction_length=horizon,
            num_samples=num_samples,
            limit_prediction_length=False,
        )
        if hasattr(samples, "detach"):
            samples_np = samples.detach().cpu().numpy()
        else:
            samples_np = np.asarray(samples)
        preds.append(_reduce_samples(samples_np, horizon=horizon, num_samples=num_samples))

    runtime_seconds = time.perf_counter() - t0
    point = np.concatenate(preds, axis=0)
    print(f"[runner]   forecast done in {runtime_seconds:.3f}s", flush=True)
    return point, runtime_seconds


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

    base_ts = ts[idxs]
    future_ts = [ts[i + 1 : i + 1 + H] for i in idxs]
    true_vals = [vals[i + 1 : i + 1 + H] for i in idxs]

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

    df_wide = pd.DataFrame({"base_timestamp": base_ts})
    for h in range(H):
        df_wide[f"y_pred_h{h+1}"] = point[:, h]
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
    num_samples: int,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
    model_path: str,
) -> Dict[str, Any]:
    print("[runner] 6/6 saving artifacts…", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    df_long.to_csv(os.path.join(output_dir, "chronos_long.csv"), index=False)
    df_wide.to_csv(os.path.join(output_dir, "chronos_wide.csv"), index=False)
    df_step1.to_csv(os.path.join(output_dir, "chronos_step1.csv"), index=False)

    summary = {
        "context_length": context_length,
        "horizon": horizon,
        "num_samples": num_samples,
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

    try:
        write_threshold_sweeps(output_dir, model="chronos")
    except Exception:
        print("[runner] WARNING: metrics_sweep generation failed; continuing.", flush=True)
        traceback.print_exc()

    print("[runner] done.", flush=True)
    return summary


# =====================================================
# Public API for sweeper and programmatic use
# =====================================================
def run_single_experiment(
    *,
    data_csv: str,
    output_dir: str,
    context_length: int,
    horizon: int,
    num_samples: int,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
    timestamp_col: str,
    target_col: str,
    model_path: str,
    device: str = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    print(f"[runner] API run_single_experiment → {output_dir}", flush=True)

    df, ts, vals, idxs, contexts = load_data(
        data_csv=data_csv,
        timestamp_col=timestamp_col,
        target_col=target_col,
        context_length=context_length,
        horizon=horizon,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = load_chronos_pipeline(
        model_path=model_path,
        device=device,
    )

    point, runtime_seconds = run_forecast(
        pipe=pipe,
        contexts=contexts,
        horizon=horizon,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    df_long, df_wide, df_step1 = build_output_dfs(
        point=point,
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
        num_samples=num_samples,
        starting_capital=starting_capital,
        threshold=threshold,
        fee_rate=fee_rate,
        model_path=model_path,
    )

    summary["data_rows"] = len(df)
    summary["n_windows"] = len(contexts)
    summary["device"] = device
    return summary


# =====================================================
# CLI entrypoint for VS Code "Run"
# =====================================================
def main():
    params = USER_DEFAULTS_RUNNER
    out_dir = os.path.join(params["results_root"], f"ctx{params['context_length']}_h{params['horizon']}")
    run_single_experiment(
        data_csv=params["data_csv"],
        output_dir=out_dir,
        context_length=params["context_length"],
        horizon=params["horizon"],
        num_samples=params["num_samples"],
        starting_capital=params["starting_capital"],
        threshold=params["threshold"],
        fee_rate=params["fee_rate"],
        timestamp_col=params["timestamp_col"],
        target_col=params["target_col"],
        model_path=params["model_path"],
        device=params["device"],
        batch_size=params["batch_size"],
    )


if __name__ == "__main__":
    main()
