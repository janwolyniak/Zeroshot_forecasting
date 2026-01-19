"""
moirai_runner.py - single experiment runner for Moirai (Uni2TS) forecasts.
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

# ----------------------------
# Make sure project + Uni2TS imports work
# ----------------------------
SCRIPT_DIR = r"/Users/jan/Documents/working papers/project 1/models"
PROJECT_ROOT = r"/Users/jan/Documents/working papers/project 1"
MOIRAI_SRC = r"/Users/jan/Documents/working papers/project 1/moirai/src"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if MOIRAI_SRC not in sys.path:
    sys.path.insert(0, MOIRAI_SRC)

from gluonts.dataset.pandas import PandasDataset
from utils.thresholds import DEFAULT_THRESHOLD
from utils.metrics import evaluate_forecast
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule


USER_DEFAULTS_RUNNER = {
    "data_csv": "/Users/jan/Documents/working papers/project 1/data/btc_1h_test.csv",
    "results_root": "/Users/jan/Documents/working papers/project 1/moirai-results",
    "model_family": "moirai",
    "model_id": "Salesforce/moirai-1.1-R-small",
    "context_length": 512,
    "horizon": 1,
    "patch_size": 16,
    "num_samples": 20,
    "batch_size": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
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


def _forecast_to_point(forecast) -> np.ndarray:
    if hasattr(forecast, "samples"):
        return np.median(forecast.samples, axis=0)
    if hasattr(forecast, "mean"):
        return np.asarray(forecast.mean)
    return np.asarray(forecast)


def load_data(
    data_csv: str,
    timestamp_col: str,
    target_col: str,
    context_length: int,
    horizon: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    print("[runner] 1/6 loading data...", flush=True)
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

    idxs = np.arange(context_length - 1, len(vals) - horizon)
    print(f"[runner]   data ready: {len(df)} rows | windows to forecast: {len(idxs)}", flush=True)
    return df, ts, vals, idxs


def load_model(
    model_family: str,
    model_id: str,
    context_length: int,
    horizon: int,
    patch_size: int,
    num_samples: int,
) -> object:
    print("[runner] 2/6 loading model...", flush=True)
    target_dim = 1
    feat_dynamic_real_dim = 0
    past_feat_dynamic_real_dim = 0

    if model_family == "moirai":
        module = MoiraiModule.from_pretrained(model_id)
        model = MoiraiForecast(
            module=module,
            prediction_length=horizon,
            context_length=context_length,
            patch_size=patch_size,
            num_samples=num_samples,
            target_dim=target_dim,
            feat_dynamic_real_dim=feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        )
    elif model_family == "moirai_moe":
        module = MoiraiMoEModule.from_pretrained(model_id)
        model = MoiraiMoEForecast(
            module=module,
            prediction_length=horizon,
            context_length=context_length,
            patch_size=patch_size,
            num_samples=num_samples,
            target_dim=target_dim,
            feat_dynamic_real_dim=feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        )
    elif model_family == "moirai2":
        module = Moirai2Module.from_pretrained(model_id)
        model = Moirai2Forecast(
            module=module,
            prediction_length=horizon,
            context_length=context_length,
            target_dim=target_dim,
            feat_dynamic_real_dim=feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        )
    else:
        raise ValueError("model_family must be one of: moirai, moirai_moe, moirai2")

    model.eval()
    return model


def build_context_dataset(
    df: pd.DataFrame,
    idxs: np.ndarray,
    context_length: int,
) -> PandasDataset:
    series_map = {}
    ts = df["timestamp"].values
    vals = df["target"].values
    for offset, idx in enumerate(idxs):
        win_start = idx - context_length + 1
        win_end = idx + 1
        series = pd.Series(vals[win_start:win_end], index=ts[win_start:win_end])
        series_map[f"w{offset:06d}"] = series
    return PandasDataset(series_map)


def run_forecast(
    model: object,
    df: pd.DataFrame,
    idxs: np.ndarray,
    context_length: int,
    horizon: int,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, float]:
    print("[runner] 3/6 forecasting...", flush=True)
    preds: List[np.ndarray] = []
    t0 = time.perf_counter()

    predictor = model.create_predictor(batch_size=batch_size, device=device)
    dataset = build_context_dataset(df, idxs, context_length)
    for forecast in predictor.predict(dataset):
        preds.append(_forecast_to_point(forecast))

    runtime_seconds = time.perf_counter() - t0
    point = np.stack(preds, axis=0)
    print(f"[runner]   forecast done in {runtime_seconds:.3f}s", flush=True)
    return point, runtime_seconds


def build_output_dfs(
    point: np.ndarray,
    ts: np.ndarray,
    vals: np.ndarray,
    idxs: np.ndarray,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[runner] 4/6 building outputs...", flush=True)
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
    model_family: str,
    model_id: str,
    context_length: int,
    horizon: int,
    patch_size: int,
    num_samples: int,
    batch_size: int,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
) -> Dict[str, Any]:
    print("[runner] 6/6 saving artifacts...", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    df_long.to_csv(os.path.join(output_dir, "moirai_long.csv"), index=False)
    df_wide.to_csv(os.path.join(output_dir, "moirai_wide.csv"), index=False)
    df_step1.to_csv(os.path.join(output_dir, "moirai_step1.csv"), index=False)

    summary = {
        "model_family": model_family,
        "model_id": model_id,
        "context_length": context_length,
        "horizon": horizon,
        "patch_size": int(patch_size),
        "num_samples": int(num_samples),
        "batch_size": int(batch_size),
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

    print("[runner] done.", flush=True)
    return summary


def run_single_experiment(
    *,
    data_csv: str,
    output_dir: str,
    model_family: str,
    model_id: str,
    context_length: int,
    horizon: int,
    patch_size: int,
    num_samples: int,
    batch_size: int,
    device: str,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
    timestamp_col: str,
    target_col: str,
) -> Dict[str, Any]:
    print(f"[runner] API run_single_experiment -> {output_dir}", flush=True)

    df, ts, vals, idxs = load_data(
        data_csv=data_csv,
        timestamp_col=timestamp_col,
        target_col=target_col,
        context_length=context_length,
        horizon=horizon,
    )

    model = load_model(
        model_family=model_family,
        model_id=model_id,
        context_length=context_length,
        horizon=horizon,
        patch_size=patch_size,
        num_samples=num_samples,
    )

    point, runtime_seconds = run_forecast(
        model=model,
        df=df,
        idxs=idxs,
        context_length=context_length,
        horizon=horizon,
        batch_size=batch_size,
        device=device,
    )

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
        model_family=model_family,
        model_id=model_id,
        context_length=context_length,
        horizon=horizon,
        patch_size=patch_size,
        num_samples=num_samples,
        batch_size=batch_size,
        starting_capital=starting_capital,
        threshold=threshold,
        fee_rate=fee_rate,
    )
    return summary


def main():
    faulthandler.enable()

    params = dict(**USER_DEFAULTS_RUNNER)

    run_name = f"ctx{params['context_length']}_h{params['horizon']}_ps{params['patch_size']}"
    output_dir = os.path.join(params["results_root"], run_name)
    if os.path.exists(output_dir):
        from datetime import datetime
        output_dir += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    print("[runner] START")
    print("[runner] data_csv     =", params["data_csv"])
    print("[runner] model_family =", params["model_family"])
    print("[runner] model_id     =", params["model_id"])
    print("[runner] results_root =", params["results_root"])
    print("[runner] output_dir   =", output_dir)
    print("[runner] ctx,hzn      =", params["context_length"], params["horizon"])

    assert os.path.exists(params["data_csv"]), "data_csv not found"

    try:
        summary = run_single_experiment(
            data_csv=params["data_csv"],
            output_dir=output_dir,
            model_family=params["model_family"],
            model_id=params["model_id"],
            context_length=params["context_length"],
            horizon=params["horizon"],
            patch_size=params["patch_size"],
            num_samples=params["num_samples"],
            batch_size=params["batch_size"],
            device=params["device"],
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
