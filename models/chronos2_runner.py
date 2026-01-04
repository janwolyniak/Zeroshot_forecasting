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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils.metrics import evaluate_forecast

faulthandler.enable()


# =======================
# VS CODE RUN DEFAULTS
# =======================
USER_DEFAULTS_RUNNER = {
    "data_csv": os.path.join(PROJECT_ROOT, "data", "btc_1h_test.csv"),
    "results_root": os.path.join(PROJECT_ROOT, "chronos2-results"),
    "context_length": 512,
    "horizon": 1,
    "starting_capital": 100_000.0,
    "threshold": 0.005,
    "fee_rate": 0.001,
    "timestamp_col": "timestamp",
    "target_col": "close",
    "feature_cols": ["open", "high", "low", "volume"],
    "use_future_covariates": True,
    "model_path": "amazon/chronos-2",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 32,
    "cross_learning": False,
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


def _parse_feature_cols(df: pd.DataFrame, timestamp_col: str, target_col: str, feature_cols: Optional[List[str]]):
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
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, np.ndarray], List[int]]:
    print("[runner2] 1/6 loading data…", flush=True)

    df = pd.read_csv(data_csv)
    if timestamp_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Expected {timestamp_col=} and {target_col=} in CSV columns, found: {list(df.columns)}")

    df = df.rename(columns={timestamp_col: "timestamp", target_col: "target"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "target"]).sort_values("timestamp").reset_index(drop=True)

    feature_cols = _parse_feature_cols(df, timestamp_col="timestamp", target_col="target", feature_cols=feature_cols)
    feature_cols = [c for c in feature_cols if c not in ("timestamp", "target")]

    vals = df["target"].to_numpy(np.float32)
    ts = df["timestamp"].to_numpy()

    covariates: Dict[str, np.ndarray] = {}
    for col in feature_cols:
        # best-effort numeric cast (Chronos-2 can handle categorical too, but keep this runner simple)
        covariates[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float32)

    need = context_length + horizon + 1
    if len(vals) < need:
        raise ValueError(f"Not enough rows for CONTEXT+HORIZON: need {need}, have {len(vals)}")

    idxs = np.arange(context_length - 1, len(vals) - horizon).tolist()

    print(
        f"[runner2]   data ready: {len(df)} rows | windows to forecast: {len(idxs)} | covariates: {list(covariates.keys())}",
        flush=True,
    )
    return df, ts, vals, covariates, idxs


def _build_tasks_for_indices(
    *,
    idxs: List[int],
    target_vals: np.ndarray,
    covariates: Dict[str, np.ndarray],
    context_length: int,
    horizon: int,
    use_future_covariates: bool,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    tasks: List[Dict[str, Any]] = []
    kept_indices: List[int] = []

    for i in idxs:
        start = i - context_length + 1
        end = i + 1
        target_context = target_vals[start:end]
        if np.isnan(target_context).any():
            continue

        task: Dict[str, Any] = {"target": target_context}

        if covariates:
            past_cov = {}
            future_cov = {}
            ok = True
            for name, arr in covariates.items():
                past = arr[start:end]
                if past.shape[0] != context_length or np.isnan(past).any():
                    ok = False
                    break
                past_cov[name] = past

                if use_future_covariates:
                    fut = arr[i + 1 : i + 1 + horizon]
                    if fut.shape[0] != horizon or np.isnan(fut).any():
                        ok = False
                        break
                    future_cov[name] = fut

            if not ok:
                continue

            task["past_covariates"] = past_cov
            if use_future_covariates:
                task["future_covariates"] = future_cov

        tasks.append(task)
        kept_indices.append(i)

    return tasks, kept_indices


# ==============
# Model loading
# ==============
def load_chronos2_pipeline(model_path: str, device: str):
    print("[runner2] 2/6 loading Chronos2Pipeline…", flush=True)
    from chronos import Chronos2Pipeline

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = Chronos2Pipeline.from_pretrained(
        model_path,
        device_map="auto" if device == "cuda" else {"": device},
        torch_dtype=dtype,
    )
    return pipe


# ==============
# Forecast
# ==============
def run_forecast(
    pipe,
    *,
    target_vals: np.ndarray,
    covariates: Dict[str, np.ndarray],
    idxs: List[int],
    context_length: int,
    horizon: int,
    batch_size: int,
    cross_learning: bool,
    use_future_covariates: bool,
) -> Tuple[np.ndarray, List[int], float]:
    print("[runner2] 3/6 forecasting…", flush=True)
    preds: List[np.ndarray] = []
    kept_all: List[int] = []
    t0 = time.perf_counter()

    for start in range(0, len(idxs), batch_size):
        end = start + batch_size
        chunk_idxs = idxs[start:end]
        tasks, kept = _build_tasks_for_indices(
            idxs=chunk_idxs,
            target_vals=target_vals,
            covariates=covariates,
            context_length=context_length,
            horizon=horizon,
            use_future_covariates=use_future_covariates,
        )
        if not tasks:
            continue

        quantiles, mean = pipe.predict_quantiles(
            inputs=tasks,
            prediction_length=horizon,
            quantile_levels=[0.5],
            batch_size=batch_size,
            context_length=context_length,
            cross_learning=cross_learning,
            limit_prediction_length=False,
        )
        # mean is a list of tensors, one per task; each tensor shape (n_variates, horizon)
        chunk_point = np.stack([m.squeeze(0).detach().cpu().numpy() for m in mean], axis=0).astype(np.float32)

        preds.append(chunk_point)
        kept_all.extend(kept)

    runtime_seconds = time.perf_counter() - t0
    if not preds:
        raise RuntimeError("No valid forecast windows produced (check NaNs / covariates / horizon).")

    point = np.concatenate(preds, axis=0)
    print(f"[runner2]   forecast done in {runtime_seconds:.3f}s | windows kept: {len(kept_all)}", flush=True)
    return point, kept_all, runtime_seconds


# ==============
# Outputs
# ==============
def build_output_dfs(
    point: np.ndarray,
    ts: np.ndarray,
    vals: np.ndarray,
    kept_indices: List[int],
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[runner2] 4/6 building outputs…", flush=True)
    N, H = point.shape
    if H != horizon:
        raise ValueError(f"horizon mismatch: expected {horizon}, got {H}")
    if N != len(kept_indices):
        raise ValueError(f"kept_indices mismatch: {N=} but {len(kept_indices)=}")

    base_ts = ts[np.array(kept_indices, dtype=int)]
    future_ts = [ts[i + 1 : i + 1 + H] for i in kept_indices]
    true_vals = [vals[i + 1 : i + 1 + H] for i in kept_indices]

    rows = []
    for n in range(N):
        for h in range(H):
            rows.append(
                (
                    base_ts[n],
                    h + 1,
                    future_ts[n][h],
                    float(point[n, h]),
                    float(true_vals[n][h]),
                )
            )
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
    print("[runner2] 5/6 evaluating and plotting…", flush=True)
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
    feature_cols: List[str],
    use_future_covariates: bool,
    cross_learning: bool,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
    model_path: str,
    n_windows_total: int,
    n_windows_kept: int,
    device: str,
) -> Dict[str, Any]:
    print("[runner2] 6/6 saving artifacts…", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    df_long.to_csv(os.path.join(output_dir, "chronos2_long.csv"), index=False)
    df_wide.to_csv(os.path.join(output_dir, "chronos2_wide.csv"), index=False)
    df_step1.to_csv(os.path.join(output_dir, "chronos2_step1.csv"), index=False)

    summary = {
        "context_length": context_length,
        "horizon": horizon,
        "feature_cols": list(feature_cols),
        "use_future_covariates": bool(use_future_covariates),
        "cross_learning": bool(cross_learning),
        "starting_capital": starting_capital,
        "threshold": threshold,
        "fee_rate": fee_rate,
        "runtime_seconds": float(runtime_seconds),
        "model_path": model_path,
        "n_windows_total": int(n_windows_total),
        "n_windows_kept": int(n_windows_kept),
        "device": device,
    }
    for k, v in (metrics or {}).items():
        summary[k] = _json_safe(v)

    try:
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        print("[runner2] WARNING: metrics.json write failed; continuing.", flush=True)
        traceback.print_exc()

    try:
        pd.DataFrame([summary]).to_csv(os.path.join(output_dir, "summary_row.csv"), index=False)
    except Exception:
        print("[runner2] WARNING: summary_row.csv write failed; continuing.", flush=True)
        traceback.print_exc()

    print("[runner2] done.", flush=True)
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
    starting_capital: float,
    threshold: float,
    fee_rate: float,
    timestamp_col: str,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    use_future_covariates: bool = True,
    model_path: str,
    device: Optional[str] = None,
    batch_size: int = 32,
    cross_learning: bool = False,
) -> Dict[str, Any]:
    print(f"[runner2] API run_single_experiment → {output_dir}", flush=True)

    df, ts, vals, covariates, idxs_list = load_data(
        data_csv=data_csv,
        timestamp_col=timestamp_col,
        target_col=target_col,
        feature_cols=feature_cols,
        context_length=context_length,
        horizon=horizon,
    )

    # Re-map inferred features to covariate dict in load_data, but keep feature_cols for metadata.
    inferred_feature_cols = list(covariates.keys())

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = load_chronos2_pipeline(model_path=model_path, device=device)

    point, kept_indices, runtime_seconds = run_forecast(
        pipe=pipe,
        target_vals=vals,
        covariates=covariates,
        idxs=idxs_list,
        context_length=context_length,
        horizon=horizon,
        batch_size=batch_size,
        cross_learning=cross_learning,
        use_future_covariates=use_future_covariates,
    )

    df_long, df_wide, df_step1 = build_output_dfs(
        point=point,
        ts=ts,
        vals=vals,
        kept_indices=kept_indices,
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
        feature_cols=inferred_feature_cols,
        use_future_covariates=use_future_covariates,
        cross_learning=cross_learning,
        starting_capital=starting_capital,
        threshold=threshold,
        fee_rate=fee_rate,
        model_path=model_path,
        n_windows_total=len(idxs_list),
        n_windows_kept=len(kept_indices),
        device=device,
    )

    summary["data_rows"] = len(df)
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
        starting_capital=params["starting_capital"],
        threshold=params["threshold"],
        fee_rate=params["fee_rate"],
        timestamp_col=params["timestamp_col"],
        target_col=params["target_col"],
        feature_cols=params["feature_cols"],
        use_future_covariates=params["use_future_covariates"],
        model_path=params["model_path"],
        device=params["device"],
        batch_size=params["batch_size"],
        cross_learning=params["cross_learning"],
    )


if __name__ == "__main__":
    main()
