from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NOTE: this script does not depend on "norm" at all; runs with different
# normalization flags are treated the same. Folder names may still include
# "_normTrue", but that is ignored everywhere.

from metrics import evaluate_forecast  # your existing backtest function

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

CUT_TIMESTAMP = pd.Timestamp("2018-03-27 07:00:00")
THRESHOLDS: Sequence[float] = (0.01, 0.015, 0.02, 0.025, 0.03)

EQUITY_SWEEP_CSV = "equity_sweep.csv"
METRICS_SWEEP_CSV = "metrics_sweep.csv"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_step1_and_wide(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load timesfm_step1.csv and timesfm_wide.csv for a run."""
    run_dir = Path(run_dir)
    step_path = run_dir / "timesfm_step1.csv"
    wide_path = run_dir / "timesfm_wide.csv"
    if not step_path.exists():
        raise FileNotFoundError(f"Missing {step_path}")
    if not wide_path.exists():
        raise FileNotFoundError(f"Missing {wide_path}")
    df_step1 = pd.read_csv(step_path)
    df_step1["timestamp"] = pd.to_datetime(df_step1["timestamp"], errors="coerce")
    df_wide = pd.read_csv(wide_path)
    if "base_timestamp" in df_wide.columns:
        df_wide["base_timestamp"] = pd.to_datetime(df_wide["base_timestamp"], errors="coerce")
    if len(df_step1) != len(df_wide):
        raise ValueError(
            f"Row mismatch in {run_dir}: step1={len(df_step1)}, wide={len(df_wide)}"
        )
    return df_step1, df_wide


def _hard_cut(df_step1: pd.DataFrame, df_wide: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only rows with timestamp >= CUT_TIMESTAMP."""
    mask = df_step1["timestamp"] >= CUT_TIMESTAMP
    return df_step1.loc[mask].reset_index(drop=True), df_wide.loc[mask].reset_index(drop=True)


def _get_pred_cols(df_wide: pd.DataFrame) -> List[str]:
    cols = [c for c in df_wide.columns if c.startswith("y_pred_h")]
    def _h(col: str) -> int:
        try: return int(col.split("h", 1)[1])
        except Exception: return 10**9
    return sorted(cols, key=_h)


def _mean_horizon_pred(df_wide: pd.DataFrame, pred_cols: Sequence[str]) -> np.ndarray:
    preds = df_wide[list(pred_cols)].astype(float).to_numpy()
    return preds.mean(axis=1)


def _maybe_read_baseline_settings(run_dir: Path) -> Tuple[float, float]:
    start_cap = 100_000.0
    fee_rate = 0.001
    mpath = Path(run_dir) / "metrics.json"
    if not mpath.exists():
        return start_cap, fee_rate
    try:
        with open(mpath, "r") as f:
            obj = json.load(f)
        if "starting_capital" in obj: start_cap = float(obj["starting_capital"])
        if "fee_rate" in obj: fee_rate = float(obj["fee_rate"])
    except Exception:
        pass
    return start_cap, fee_rate


def _evaluate_variant(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.Series,
    thresholds: Sequence[float],
    starting_capital: float,
    fee_rate: float,
) -> Dict[float, Dict[str, Any]]:
    out: Dict[float, Dict[str, Any]] = {}
    for thr in thresholds:
        res = evaluate_forecast(
            y_true=y_true,
            y_pred=y_pred,
            timestamps=timestamps,
            starting_capital=starting_capital,
            threshold=float(thr),
            fee_rate=fee_rate,
        )
        out[float(thr)] = res
    return out


def _build_equity_sweep_df(
    timestamps: pd.Series,
    y_true: np.ndarray,
    step1_metrics: Mapping[float, Mapping[str, Any]],
    mean_metrics: Mapping[float, Mapping[str, Any]] | None,
) -> pd.DataFrame:
    df = pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "y_true": y_true.astype(float)})
    for thr, m in step1_metrics.items():
        df[f"eq_step1_thr{thr:.3f}_no_tc"] = np.asarray(m["equity_no_tc"], dtype=float)
        df[f"eq_step1_thr{thr:.3f}_tc"] = np.asarray(m["equity_tc"], dtype=float)
    if mean_metrics:
        for thr, m in mean_metrics.items():
            df[f"eq_mean_h_thr{thr:.3f}_no_tc"] = np.asarray(m["equity_no_tc"], dtype=float)
            df[f"eq_mean_h_thr{thr:.3f}_tc"] = np.asarray(m["equity_tc"], dtype=float)
    return df


def _scalarize_metrics(m: Mapping[str, Any]) -> Dict[str, float]:
    keys = ["final_equity_no_tc","final_equity_tc","total_fees_paid","arc","asd",
            "mdd","ir_star","ir_starstar","pct_long","pct_short","pct_flat",
            "trades_long","trades_short","trades_flat","trades_total"]
    out: Dict[str, float] = {}
    for k in keys:
        v = m.get(k)
        try: out[k] = float(v)
        except Exception: out[k] = float("nan")
    return out


def _build_metrics_sweep_df(
    step1_metrics: Mapping[float, Mapping[str, Any]],
    mean_metrics: Mapping[float, Mapping[str, Any]] | None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for thr, m in step1_metrics.items():
        row = {"variant": "step1", "threshold": float(thr)}; row.update(_scalarize_metrics(m)); rows.append(row)
    if mean_metrics:
        for thr, m in mean_metrics.items():
            row = {"variant": "mean_horizon", "threshold": float(thr)}; row.update(_scalarize_metrics(m)); rows.append(row)
    return pd.DataFrame(rows)


def _plot_equity_sweep(df_eq: pd.DataFrame, out_dir: Path, use_tc: bool) -> None:
    suffix = "tc" if use_tc else "no_tc"
    ts = pd.to_datetime(df_eq["timestamp"])
    fig, ax = plt.subplots(figsize=(12, 6))
    colours = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or [f"C{i}" for i in range(10)]
    has_mean = any(col.startswith("eq_mean_h_thr") and col.endswith(f"_{suffix}") for col in df_eq.columns)
    for idx, thr in enumerate(THRESHOLDS):
        colour = colours[idx % len(colours)]
        step_col = f"eq_step1_thr{thr:.3f}_{suffix}"
        mean_col = f"eq_mean_h_thr{thr:.3f}_{suffix}"
        if step_col in df_eq.columns:
            ax.plot(ts, df_eq[step_col].values, label=f"step1, thr={thr:.3f}", color=colour, linestyle="-")
        if has_mean and mean_col in df_eq.columns:
            ax.plot(ts, df_eq[mean_col].values, label=f"mean_h, thr={thr:.3f}", color=colour, linestyle="--")
    ax.set_title(f"Equity by threshold ({'with TC' if use_tc else 'no TC'})")
    ax.set_xlabel("Time"); ax.set_ylabel("Equity"); ax.grid(True, linestyle="--", alpha=0.3); ax.legend(loc="best", ncol=2)
    fig.tight_layout(); fig.savefig(out_dir / f"equity_sweep_{suffix}.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------

def process_run_dir(run_dir: Path, overwrite: bool = False) -> None:
    run_dir = Path(run_dir)
    eq_path, metrics_path = run_dir / EQUITY_SWEEP_CSV, run_dir / METRICS_SWEEP_CSV
    if not overwrite and eq_path.exists() and metrics_path.exists():
        print(f"[metrics] Skipping {run_dir.name} (already has sweep CSVs)"); return
    print(f"[metrics] Processing {run_dir}", flush=True)

    df_step1, df_wide = _load_step1_and_wide(run_dir)
    df_step1, df_wide = _hard_cut(df_step1, df_wide)

    timestamps = df_step1["timestamp"]
    y_true = df_step1["y_true"].astype(float).to_numpy()
    y_pred_step1 = df_step1["y_pred"].astype(float).to_numpy()
    pred_cols = _get_pred_cols(df_wide)
    start_cap, fee_rate = _maybe_read_baseline_settings(run_dir)

    step1_metrics = _evaluate_variant(y_true, y_pred_step1, timestamps, THRESHOLDS, start_cap, fee_rate)
    mean_metrics: Dict[float, Dict[str, Any]] | None = None
    if len(pred_cols) > 1:
        y_pred_mean = _mean_horizon_pred(df_wide, pred_cols)
        mean_metrics = _evaluate_variant(y_true, y_pred_mean, timestamps, THRESHOLDS, start_cap, fee_rate)
        print(f"[metrics]   horizons: {len(pred_cols)} -> computed mean-horizon")
    else:
        print("[metrics]   horizon=1 -> only step1")

    df_eq = _build_equity_sweep_df(timestamps, y_true, step1_metrics, mean_metrics)
    df_met = _build_metrics_sweep_df(step1_metrics, mean_metrics)
    df_eq.to_csv(eq_path, index=False); df_met.to_csv(metrics_path, index=False)
    print(f"[metrics]   wrote {eq_path.name}, {metrics_path.name}")

    _plot_equity_sweep(df_eq, run_dir, use_tc=False)
    _plot_equity_sweep(df_eq, run_dir, use_tc=True)
    print("[metrics]   wrote equity_sweep_no_tc.png / equity_sweep_tc.png")

# ---------------------------------------------------------------------
# Scan root + public entrypoints
# ---------------------------------------------------------------------

def scan_results_root(results_root: Path, overwrite: bool = False) -> None:
    results_root = Path(results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"results_root not found: {results_root}")
    run_dirs = [
        d for d in sorted(results_root.iterdir()) if d.is_dir()
        and (d / "timesfm_step1.csv").exists() and (d / "timesfm_wide.csv").exists()
    ]
    if not run_dirs:
        print(f"[metrics] No valid run dirs under {results_root}"); return
    print(f"[metrics] Found {len(run_dirs)} run dirs under {results_root}")
    print(f"[metrics] Hard cut at timestamp >= {CUT_TIMESTAMP}")
    for rd in run_dirs:
        try: process_run_dir(rd, overwrite=overwrite)
        except Exception as e: print(f"[metrics] ERROR in {rd.name}: {e}")

def run_all(results_root: str | Path = "timesfm-results", overwrite: bool = False) -> None:
    scan_results_root(Path(results_root).expanduser().resolve(), overwrite=overwrite)

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Post-hoc metrics + plots for TimesFM forecasts.")
    p.add_argument("--results_root", type=str, default="timesfm-results",
                   help="Root directory with run subfolders (default: 'timesfm-results').")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if sweep CSVs exist.")
    return p

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv); run_all(args.results_root, overwrite=args.overwrite)

if __name__ == "__main__":
    main(["--overwrite"])