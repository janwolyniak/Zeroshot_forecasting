"""
moment_runner.py — single experiment runner for MOMENT zero-shot forecasts.

Pipeline (mirrors other runners):
  1) Load data
  2) Z-score normalize for model input (optional)
  3) Load MOMENT model (momentfm)
  4) Rolling forecast for chosen horizon
  5) Save moment_long / moment_wide / moment_step1 CSVs
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

from utils.thresholds import DEFAULT_THRESHOLD
from utils.metrics import evaluate_forecast

faulthandler.enable()


# =======================
# VS CODE RUN DEFAULTS
# =======================
USER_DEFAULTS_RUNNER = {
    "data_csv": "/Users/jan/Documents/working papers/project 1/data/btc_1h_test.csv",
    "results_root": "/Users/jan/Documents/working papers/project 1/moment-results",

    "context_length": 512,
    "horizon": 1,
    "normalize_inputs": True,
    "batch_size": 128,

    "starting_capital": 100_000.0,
    "threshold": DEFAULT_THRESHOLD,
    "fee_rate": 0.001,

    "timestamp_col": "timestamp",
    "target_col": "close",
    "repo_id": "AutonLab/MOMENT-1-small",
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
    context_length: int,
    horizon: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    print("[runner] 1/6 loading data…", flush=True)
    df = pd.read_csv(data_csv, usecols=[timestamp_col, target_col])
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    df = df.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col).reset_index(drop=True)

    vals = df[target_col].astype(float).to_numpy(np.float32)
    ts = df[timestamp_col].to_numpy()

    need = context_length + horizon
    if len(vals) < need:
        raise ValueError(f"Not enough rows for CONTEXT+HORIZON: need {need}, have {len(vals)}")
    if context_length < 32:
        raise ValueError("Context length must be >= 32 for foundation models.")

    idxs = np.arange(context_length - 1, len(vals) - horizon)
    print(f"[runner]   data ready: {len(df)} rows | windows to forecast: {len(idxs)}", flush=True)
    return df, ts, vals, idxs


def normalize_series(vals: np.ndarray, normalize_inputs: bool) -> Tuple[np.ndarray, float, float]:
    if not normalize_inputs:
        return vals.astype(np.float32), 0.0, 1.0
    mu = float(np.mean(vals))
    sigma = float(np.std(vals, ddof=0))
    scale = sigma if sigma > 0 else 1.0
    x = (vals - mu) / scale
    return x.astype(np.float32), mu, sigma


# ==============
# Model loading
# ==============
def load_moment_pipeline(repo_id: str, device: str):
    print("[runner] 2/6 loading MOMENTPipeline…", flush=True)
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(repo_id)
    if hasattr(model, "init"):
        model.init()
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "to"):
        model.to(device)
    elif hasattr(model, "model") and hasattr(model.model, "to"):
        model.model.to(device)
    return model


# ==============
# Forecast helpers
# ==============
def _extract_forecast_array(out: Any) -> np.ndarray:
    if isinstance(out, dict):
        y_hat = None
        for key in ("forecast", "prediction", "y_hat", "pred"):
            if key in out:
                y_hat = out[key]
                break
        if y_hat is None and len(out) == 1:
            y_hat = next(iter(out.values()))
    else:
        y_hat = None
        for key in ("forecast", "prediction", "y_hat", "pred"):
            if hasattr(out, key):
                y_hat = getattr(out, key)
                break
        if y_hat is None:
            y_hat = out

    if hasattr(y_hat, "detach"):
        return y_hat.detach().cpu().numpy()
    return np.asarray(y_hat)


def _coerce_forecast_shape(y_hat: np.ndarray, batch_size: int, horizon: int) -> np.ndarray:
    if y_hat.ndim == 3 and y_hat.shape[1] == 1:
        y_hat = y_hat[:, 0, :]
    elif y_hat.ndim == 3 and y_hat.shape[0] == 1 and y_hat.shape[1] == horizon:
        y_hat = y_hat[0]

    if y_hat.ndim == 2:
        pass
    elif y_hat.ndim == 1:
        if y_hat.size == horizon and batch_size == 1:
            y_hat = y_hat[None, :]
        elif y_hat.size == batch_size and horizon == 1:
            y_hat = y_hat[:, None]
        elif (y_hat.size % horizon) == 0:
            y_hat = y_hat.reshape(-1, horizon)
        else:
            raise ValueError(f"Unexpected 1D output shape {y_hat.shape} for B={batch_size}, H={horizon}")
    elif y_hat.ndim == 0:
        if batch_size == 1 and horizon == 1:
            y_hat = np.array([[float(y_hat)]])
        else:
            raise ValueError(f"Scalar forecast for B={batch_size}, H={horizon} is unsupported")
    else:
        y_hat = np.squeeze(y_hat)
        if y_hat.ndim == 1:
            if y_hat.size == horizon and batch_size == 1:
                y_hat = y_hat[None, :]
            elif y_hat.size == batch_size and horizon == 1:
                y_hat = y_hat[:, None]
            else:
                raise ValueError(f"Unexpected squeezed shape {y_hat.shape}")

    if y_hat.ndim != 2:
        raise ValueError(f"Could not coerce forecast to [B,H]; got {y_hat.shape}")
    if y_hat.shape[0] != batch_size:
        if y_hat.size == batch_size * horizon:
            y_hat = y_hat.reshape(batch_size, horizon)
        else:
            raise ValueError(f"Batch mismatch: forecast {y_hat.shape} vs batch {batch_size}")
    if y_hat.shape[1] < horizon:
        raise ValueError(f"Returned horizon {y_hat.shape[1]} < requested {horizon}")
    return y_hat[:, :horizon]


def run_forecast_loop(
    model: Any,
    x: np.ndarray,
    context_length: int,
    horizon: int,
    batch_size: int,
    device: str,
    normalize_inputs: bool,
    mu: float,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    print("[runner] 3/6 rolling forecast…", flush=True)
    starts = list(range(context_length, len(x) - horizon + 1))

    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []

    t0 = time.perf_counter()
    for i in range(0, len(starts), batch_size):
        batch_idx = starts[i : i + batch_size]
        batch_ctx = np.stack([x[s - context_length : s] for s in batch_idx], axis=0).astype(np.float32)
        x_t = torch.from_numpy(batch_ctx).unsqueeze(1).to(dtype=torch.float32, device=device)
        input_mask = torch.ones(x_t.shape[0], x_t.shape[2], dtype=torch.bool, device=device)

        with torch.no_grad():
            out = model.forecast(x_enc=x_t, input_mask=input_mask, horizon=int(horizon))

        y_hat = _extract_forecast_array(out)
        y_hat = _coerce_forecast_shape(y_hat, batch_size=len(batch_idx), horizon=horizon)
        preds.append(y_hat)

        for s in batch_idx:
            true_slice = x[s : s + horizon]
            if len(true_slice) == horizon:
                trues.append(true_slice)

        if (i // batch_size) % 20 == 0:
            print(f"[runner]   processed {min(i + batch_size, len(starts))}/{len(starts)}", flush=True)

    runtime_seconds = time.perf_counter() - t0
    y_pred = np.vstack(preds)
    y_true = np.vstack(trues)

    if normalize_inputs:
        scale = sigma if sigma > 0 else 1.0
        y_pred = (y_pred * scale) + mu
        y_true = (y_true * scale) + mu

    print(f"[runner]   forecast loop done in {runtime_seconds:.3f}s", flush=True)
    return y_pred, y_true, runtime_seconds


# ==============
# Outputs
# ==============
def build_output_dfs(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    ts: np.ndarray,
    idxs: np.ndarray,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[runner] 4/6 building outputs…", flush=True)
    N, H = y_pred.shape
    if H != horizon:
        raise ValueError(f"horizon mismatch: expected {horizon}, got {H}")

    base_ts = ts[idxs]
    future_ts = [ts[i + 1 : i + 1 + H] for i in idxs]

    rows = []
    for n in range(N):
        for h in range(H):
            rows.append((
                base_ts[n],
                h + 1,
                future_ts[n][h],
                float(y_pred[n, h]),
                float(y_true[n, h]),
            ))
    df_long = pd.DataFrame(
        rows,
        columns=["base_timestamp", "horizon_step", "forecast_timestamp", "y_pred", "y_true"],
    )

    df_wide = pd.DataFrame({"base_timestamp": base_ts})
    for h in range(H):
        df_wide[f"y_pred_h{h+1}"] = y_pred[:, h]
    for h in range(H):
        df_wide[f"y_true_h{h+1}"] = y_true[:, h]

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
        plt.savefig(os.path.join(output_dir, "moment_equity_no_tc.png"), dpi=150)
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
    batch_size: int,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
    repo_id: str,
    mu: float,
    sigma: float,
) -> Dict[str, Any]:
    print("[runner] 6/6 saving artifacts…", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    df_long.to_csv(os.path.join(output_dir, "moment_long.csv"), index=False)
    df_wide.to_csv(os.path.join(output_dir, "moment_wide.csv"), index=False)
    df_step1.to_csv(os.path.join(output_dir, "moment_step1.csv"), index=False)

    summary = {
        "context_length": context_length,
        "horizon": horizon,
        "normalize_inputs": bool(normalize_inputs),
        "batch_size": int(batch_size),
        "starting_capital": starting_capital,
        "threshold": threshold,
        "fee_rate": fee_rate,
        "runtime_seconds": float(runtime_seconds),
        "repo_id": repo_id,
        "mu": float(mu),
        "sigma": float(sigma),
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
    normalize_inputs: bool,
    batch_size: int,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
    timestamp_col: str,
    target_col: str,
    repo_id: str,
    device: str,
) -> Dict[str, Any]:
    if horizon not in (1, 3, 5, 20):
        raise ValueError(f"Horizon must be one of (1, 3, 5, 20); got {horizon}")

    print(f"[runner] API run_single_experiment → {output_dir}", flush=True)

    _, ts, vals, idxs = load_data(
        data_csv=data_csv,
        timestamp_col=timestamp_col,
        target_col=target_col,
        context_length=context_length,
        horizon=horizon,
    )

    x, mu, sigma = normalize_series(vals=vals, normalize_inputs=normalize_inputs)

    model = load_moment_pipeline(repo_id=repo_id, device=device)

    y_pred, y_true, runtime_seconds = run_forecast_loop(
        model=model,
        x=x,
        context_length=context_length,
        horizon=horizon,
        batch_size=batch_size,
        device=device,
        normalize_inputs=normalize_inputs,
        mu=mu,
        sigma=sigma,
    )

    df_long, df_wide, df_step1 = build_output_dfs(
        y_pred=y_pred,
        y_true=y_true,
        ts=ts,
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
        normalize_inputs=normalize_inputs,
        batch_size=batch_size,
        starting_capital=starting_capital,
        threshold=threshold,
        fee_rate=fee_rate,
        repo_id=repo_id,
        mu=mu,
        sigma=sigma,
    )
    return summary


# =====================================================
# Standalone entry point (VS Code Run)
# =====================================================
def main():
    params = dict(**USER_DEFAULTS_RUNNER)

    run_name = f"ctx{params['context_length']}_h{params['horizon']}_norm{params['normalize_inputs']}"
    output_dir = os.path.join(params["results_root"], run_name)
    if os.path.exists(output_dir):
        from datetime import datetime
        output_dir += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    print("[runner] START")
    print("[runner] data_csv     =", params["data_csv"])
    print("[runner] results_root =", params["results_root"])
    print("[runner] output_dir   =", output_dir)
    print("[runner] ctx,hzn      =", params["context_length"], params["horizon"])
    print("[runner] normalize    =", params["normalize_inputs"])
    print("[runner] repo_id      =", params["repo_id"])
    print("[runner] device       =", params["device"])

    assert os.path.exists(params["data_csv"]), "data_csv not found"

    try:
        summary = run_single_experiment(
            data_csv=params["data_csv"],
            output_dir=output_dir,
            context_length=params["context_length"],
            horizon=params["horizon"],
            normalize_inputs=params["normalize_inputs"],
            batch_size=params["batch_size"],
            starting_capital=params["starting_capital"],
            threshold=params["threshold"],
            fee_rate=params["fee_rate"],
            timestamp_col=params["timestamp_col"],
            target_col=params["target_col"],
            repo_id=params["repo_id"],
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
