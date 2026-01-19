from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from utils.metrics import evaluate_forecast  # type: ignore
except ImportError:
    from metrics import evaluate_forecast  # type: ignore

try:
    from utils.thresholds import THRESHOLDS  # type: ignore
except ImportError:
    from thresholds import THRESHOLDS  # type: ignore

EQUITY_SWEEP_CSV = "equity_sweep.csv"
METRICS_SWEEP_CSV = "metrics_sweep.csv"


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    model: str
    step1_name: str
    wide_name: str
    results_root: Path
    thresholds: Tuple[float, ...]
    cut_timestamp: pd.Timestamp | None


_MODEL_PRESETS: Dict[str, ModelConfig] = {
    "timesfm": ModelConfig(
        model="timesfm",
        step1_name="timesfm_step1.csv",
        wide_name="timesfm_wide.csv",
        results_root=Path("timesfm-results"),
        thresholds=THRESHOLDS,
        cut_timestamp=None,
    ),
    "chronos": ModelConfig(
        model="chronos",
        step1_name="chronos_step1.csv",
        wide_name="chronos_wide.csv",
        results_root=Path("chronos-results"),
        thresholds=THRESHOLDS,
        cut_timestamp=None,
    ),
    "chronos2": ModelConfig(
        model="chronos2",
        step1_name="chronos2_step1.csv",
        wide_name="chronos2_wide.csv",
        results_root=Path("chronos2-results"),
        thresholds=THRESHOLDS,
        cut_timestamp=None,
    ),
    "ttm": ModelConfig(
        model="ttm",
        step1_name="ttm_step1.csv",
        wide_name="ttm_wide.csv",
        results_root=Path("ttm-results"),
        thresholds=THRESHOLDS,
        cut_timestamp=None,
    ),
    "lagllama": ModelConfig(
        model="lagllama",
        step1_name="lagllama_step1.csv",
        wide_name="lagllama_wide.csv",
        results_root=Path("lagllama-results"),
        thresholds=THRESHOLDS,
        cut_timestamp=None,
    ),
    "moirai": ModelConfig(
        model="moirai",
        step1_name="moirai_step1.csv",
        wide_name="moirai_wide.csv",
        results_root=Path("moirai-results"),
        thresholds=THRESHOLDS,
        cut_timestamp=None,
    ),
    "moment": ModelConfig(
        model="moment",
        step1_name="moment_step1.csv",
        wide_name="moment_wide.csv",
        results_root=Path("moment-results"),
        thresholds=THRESHOLDS,
        cut_timestamp=None,
    ),
    "toto": ModelConfig(
        model="toto",
        step1_name="toto_step1.csv",
        wide_name="toto_wide.csv",
        results_root=Path("toto-results"),
        thresholds=THRESHOLDS,
        cut_timestamp=None,
    ),
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _parse_thresholds(raw: Optional[str], defaults: Sequence[float]) -> Tuple[float, ...]:
    if not raw:
        return tuple(float(x) for x in defaults)
    vals = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        vals.append(float(piece))
    if not vals:
        return tuple(float(x) for x in defaults)
    return tuple(vals)


def _aligned_cut(ts: pd.Series, cut: pd.Timestamp | None) -> pd.Timestamp | None:
    """Align cut tz to the timestamp series to avoid tz comparison errors."""
    if cut is None:
        return None
    if getattr(ts.dt, "tz", None) is not None and cut.tzinfo is None:
        try:
            return cut.tz_localize(ts.dt.tz)
        except Exception:
            return cut
    if getattr(ts.dt, "tz", None) is None and cut.tzinfo is not None:
        try:
            return cut.tz_localize(None)
        except Exception:
            return cut
    return cut


def _load_step1_and_wide(run_dir: Path, cfg: ModelConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = Path(run_dir)
    step_path = run_dir / cfg.step1_name
    wide_path = run_dir / cfg.wide_name
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


def _hard_cut(
    df_step1: pd.DataFrame,
    df_wide: pd.DataFrame,
    cut_timestamp: pd.Timestamp | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cut_timestamp is None:
        return df_step1, df_wide
    ts = df_step1["timestamp"]
    cut_ts = _aligned_cut(ts, cut_timestamp)
    if cut_ts is None:
        return df_step1, df_wide
    mask = ts >= cut_ts
    return df_step1.loc[mask].reset_index(drop=True), df_wide.loc[mask].reset_index(drop=True)


def _get_pred_cols(df_wide: pd.DataFrame) -> List[str]:
    cols = [c for c in df_wide.columns if c.startswith("y_pred_h")]

    def _h(col: str) -> int:
        try:
            return int(col.split("h", 1)[1])
        except Exception:
            return 10**9

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
        if "starting_capital" in obj:
            start_cap = float(obj["starting_capital"])
        if "fee_rate" in obj:
            fee_rate = float(obj["fee_rate"])
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
    keys = [
        "final_equity_no_tc",
        "final_equity_tc",
        "total_fees_paid",
        "arc",
        "asd",
        "mdd",
        "ir_star",
        "ir_starstar",
        "pct_long",
        "pct_short",
        "pct_flat",
        "trades_long",
        "trades_short",
        "trades_flat",
        "trades_total",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        v = m.get(k)
        try:
            out[k] = float(v)
        except Exception:
            out[k] = float("nan")
    return out


def _build_metrics_sweep_df(
    step1_metrics: Mapping[float, Mapping[str, Any]],
    mean_metrics: Mapping[float, Mapping[str, Any]] | None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for thr, m in step1_metrics.items():
        row = {"variant": "step1", "threshold": float(thr)}
        row.update(_scalarize_metrics(m))
        rows.append(row)
    if mean_metrics:
        for thr, m in mean_metrics.items():
            row = {"variant": "mean_horizon", "threshold": float(thr)}
            row.update(_scalarize_metrics(m))
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_equity_sweep(df_eq: pd.DataFrame, out_dir: Path, use_tc: bool, cfg: ModelConfig) -> None:
    suffix = "tc" if use_tc else "no_tc"
    ts = pd.to_datetime(df_eq["timestamp"])
    fig, ax = plt.subplots(figsize=(12, 6))
    colours = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or [f"C{i}" for i in range(10)]
    has_mean = any(col.startswith("eq_mean_h_thr") and col.endswith(f"_{suffix}") for col in df_eq.columns)
    for idx, thr in enumerate(cfg.thresholds):
        colour = colours[idx % len(colours)]
        step_col = f"eq_step1_thr{thr:.3f}_{suffix}"
        mean_col = f"eq_mean_h_thr{thr:.3f}_{suffix}"
        if step_col in df_eq.columns:
            ax.plot(ts, df_eq[step_col].values, label=f"step1, thr={thr:.3f}", color=colour, linestyle="-")
        if has_mean and mean_col in df_eq.columns:
            ax.plot(ts, df_eq[mean_col].values, label=f"mean_h, thr={thr:.3f}", color=colour, linestyle="--")
    ax.set_title(f"{cfg.model} · Equity by threshold ({'with TC' if use_tc else 'no TC'})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"equity_sweep_{suffix}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------

def process_run_dir(run_dir: Path, cfg: ModelConfig, overwrite: bool = False) -> None:
    run_dir = Path(run_dir)
    eq_path, metrics_path = run_dir / EQUITY_SWEEP_CSV, run_dir / METRICS_SWEEP_CSV
    if not overwrite and eq_path.exists() and metrics_path.exists():
        print(f"[metrics] Skipping {run_dir.name} (already has sweep CSVs)")
        return
    print(f"[metrics] Processing {run_dir} [{cfg.model}]", flush=True)

    df_step1, df_wide = _load_step1_and_wide(run_dir, cfg)
    df_step1, df_wide = _hard_cut(df_step1, df_wide, cfg.cut_timestamp)

    timestamps = df_step1["timestamp"]
    y_true = df_step1["y_true"].astype(float).to_numpy()
    y_pred_step1 = df_step1["y_pred"].astype(float).to_numpy()
    pred_cols = _get_pred_cols(df_wide)
    start_cap, fee_rate = _maybe_read_baseline_settings(run_dir)

    step1_metrics = _evaluate_variant(y_true, y_pred_step1, timestamps, cfg.thresholds, start_cap, fee_rate)
    mean_metrics: Dict[float, Dict[str, Any]] | None = None
    if len(pred_cols) > 1:
        y_pred_mean = _mean_horizon_pred(df_wide, pred_cols)
        mean_metrics = _evaluate_variant(y_true, y_pred_mean, timestamps, cfg.thresholds, start_cap, fee_rate)
        print(f"[metrics]   horizons: {len(pred_cols)} -> computed mean-horizon")
    else:
        print("[metrics]   horizon=1 -> only step1")

    df_eq = _build_equity_sweep_df(timestamps, y_true, step1_metrics, mean_metrics)
    df_met = _build_metrics_sweep_df(step1_metrics, mean_metrics)
    df_eq.to_csv(eq_path, index=False)
    df_met.to_csv(metrics_path, index=False)
    print(f"[metrics]   wrote {eq_path.name}, {metrics_path.name}")

    _plot_equity_sweep(df_eq, run_dir, use_tc=False, cfg=cfg)
    _plot_equity_sweep(df_eq, run_dir, use_tc=True, cfg=cfg)
    print("[metrics]   wrote equity_sweep_no_tc.png / equity_sweep_tc.png")


# ---------------------------------------------------------------------
# Scan root + public entrypoints
# ---------------------------------------------------------------------

def scan_results_root(cfg: ModelConfig, overwrite: bool = False) -> None:
    results_root = Path(cfg.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"results_root not found: {results_root}")
    run_dirs = [
        d
        for d in sorted(results_root.iterdir())
        if d.is_dir()
        and not d.name.startswith("_")
        and (d / cfg.step1_name).exists()
        and (d / cfg.wide_name).exists()
    ]
    if not run_dirs:
        print(f"[metrics] No valid run dirs under {results_root}")
        return
    print(f"[metrics] Found {len(run_dirs)} run dirs under {results_root}")
    if cfg.cut_timestamp is not None:
        print(f"[metrics] Hard cut at timestamp >= {cfg.cut_timestamp}")
    for rd in run_dirs:
        try:
            process_run_dir(rd, cfg, overwrite=overwrite)
        except Exception as e:
            print(f"[metrics] ERROR in {rd.name}: {e}")


def run_all(cfg: ModelConfig, overwrite: bool = False) -> None:
    scan_results_root(cfg, overwrite=overwrite)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Post-hoc metrics + plots for forecast results.")
    p.add_argument(
        "--model",
        choices=["all", *sorted(_MODEL_PRESETS.keys())],
        default="all",
        help="Which preset/model to process (default: all).",
    )
    p.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="Override results root directory. Defaults to the preset root.",
    )
    p.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Optional comma-separated thresholds to sweep (e.g., '0.005,0.01,0.02').",
    )
    p.add_argument("--overwrite", action="store_true", help="Recompute even if sweep CSVs exist.")
    return p


def _build_cfg(model: str, results_root: Optional[str], thresholds: Optional[str]) -> ModelConfig:
    if model not in _MODEL_PRESETS:
        raise ValueError(f"Unknown model preset: {model}")
    base = _MODEL_PRESETS[model]
    thr = _parse_thresholds(thresholds, base.thresholds)
    cfg = replace(base, thresholds=thr)
    if results_root:
        cfg = replace(cfg, results_root=Path(results_root))
    # keep paths relative unless user asked for resolve; caller handles resolve/expanduser
    cfg = replace(cfg, results_root=cfg.results_root.expanduser().resolve())
    return cfg


def get_model_config(
    model: str,
    results_root: Optional[str] = None,
    thresholds: Optional[str] = None,
) -> ModelConfig:
    return _build_cfg(model, results_root, thresholds)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    overwrite = True if args.model == "all" else args.overwrite

    if args.model == "all":
        for model in sorted(_MODEL_PRESETS.keys()):
            cfg = _build_cfg(model, args.results_root, args.thresholds)
            if not cfg.results_root.exists():
                print(f"[metrics] Skipping {model} (missing {cfg.results_root})")
                continue
            run_all(cfg, overwrite=overwrite)
        return

    cfg = _build_cfg(args.model, args.results_root, args.thresholds)
    run_all(cfg, overwrite=overwrite)


if __name__ == "__main__":
    main()
