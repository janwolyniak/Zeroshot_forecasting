#!/usr/bin/env python3
"""
Aggregate and compare trading run summaries from chronos, chronos2, timesfm, ttm, lagllama, moirai, moment, and toto.

Outputs consolidated CSVs into models_comparison/ by default:
- combined_results.csv: all normalized rows with derived metrics
- leaderboard_<metric>.csv: sorted overall leaderboard
- best_by_horizon_ctx_<metric>.csv: top run per (horizon, context_length)
- best_by_horizon_<metric>.csv: top-k runs per horizon

Usage example:
    python models_comparison/compare_results.py --metric final_equity_tc --top-k 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

MPL_CONFIG_DIR = Path("models_comparison/.matplotlib_cache")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
plt.switch_backend("Agg")  # Use non-interactive backend for headless runs.
import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive logging
        raise RuntimeError(f"Failed to read {path}") from exc


def _ensure_column(df: pd.DataFrame, col: str, value) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = value
    return df


def _normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "context_length",
        "horizon",
        "num_samples",
        "prediction_length",
        "runtime_seconds",
        "starting_capital",
        "threshold",
        "fee_rate",
        "equity_no_tc",
        "equity_tc",
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
        "data_rows",
        "n_windows",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    boolish_cols = ["scale_inputs", "normalize_inputs"]
    for col in boolish_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})

    return df


def _derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_column(df, "starting_capital", pd.NA)
    df.replace([np.inf, -np.inf], pd.NA, inplace=True)

    df["roi_no_tc"] = df["final_equity_no_tc"] / df["starting_capital"] - 1
    df["roi_tc"] = df["final_equity_tc"] / df["starting_capital"] - 1
    df["fee_impact"] = df["final_equity_no_tc"] - df["final_equity_tc"]
    df["fee_ratio_vs_start"] = df["total_fees_paid"] / df["starting_capital"]
    df["runtime_minutes"] = df["runtime_seconds"] / 60.0
    return df


def _default_variant(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "variant" not in df.columns:
        df["variant"] = name
    return df


def _load_metrics_sweeps(
    base: Path, model_family: str, meta: pd.DataFrame, default_threshold: float | None = None
) -> pd.DataFrame:
    """
    Load metrics_sweep.csv files under a base directory and attach metadata from the
    corresponding summary rows (matched by run_name).
    """
    sweeps: List[pd.DataFrame] = []
    meta_lookup = (
        meta.groupby("run_name").tail(1).set_index("run_name").to_dict(orient="index")
        if not meta.empty and "run_name" in meta.columns
        else {}
    )

    for path in base.rglob("metrics_sweep.csv"):
        if any(part.startswith("_") for part in path.parts):
            continue
        run_dir = path.parent
        run_name = run_dir.name
        df = _read_csv(path)
        if default_threshold is not None:
            _ensure_column(df, "threshold", default_threshold)
        df["run_name"] = run_name
        df["model_family"] = model_family
        df["source_path"] = str(path)

        # Attach metadata if available (context_length, horizon, starting_capital, fee_rate, etc.).
        meta_row = meta_lookup.get(run_name)
        if meta_row:
            for k, v in meta_row.items():
                if k in ("threshold", "variant", "source_path"):
                    continue
                if k not in df.columns:
                    df[k] = v

        sweeps.append(df)

    if not sweeps:
        return pd.DataFrame()
    combined = pd.concat(sweeps, ignore_index=True)
    combined = _normalize_types(combined)
    combined = _derive_metrics(_default_variant(combined, "sweep"))
    return combined


def load_chronos(base: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for summary in base.glob("summary_*.csv"):
        df = _read_csv(summary)
        _ensure_column(df, "threshold", 0.005)
        _ensure_column(df, "run_name", summary.stem.replace("summary_", ""))
        df["model_family"] = "chronos"
        df["source_path"] = str(summary)
        dfs.append(df)

    for summary in base.rglob("summary_row.csv"):
        df = _read_csv(summary)
        _ensure_column(df, "threshold", 0.005)
        _ensure_column(df, "run_name", summary.parent.name)
        df["model_family"] = "chronos"
        df["source_path"] = str(summary)
        dfs.append(df)

    meta = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    meta = _derive_metrics(_default_variant(_normalize_types(meta), "summary")) if not meta.empty else meta

    sweeps = _load_metrics_sweeps(base, "chronos", meta, default_threshold=0.005)
    if meta.empty and sweeps.empty:
        return pd.DataFrame()
    frames = [f for f in (meta, sweeps) if not f.empty]
    return pd.concat(frames, ignore_index=True)


def load_chronos2(base: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for summary in base.glob("summary_*.csv"):
        df = _read_csv(summary)
        _ensure_column(df, "threshold", 0.005)
        _ensure_column(df, "run_name", summary.stem.replace("summary_", ""))
        df["model_family"] = "chronos2"
        df["source_path"] = str(summary)
        dfs.append(df)

    for summary in base.rglob("summary_row.csv"):
        df = _read_csv(summary)
        _ensure_column(df, "threshold", 0.005)
        _ensure_column(df, "run_name", summary.parent.name)
        df["model_family"] = "chronos2"
        df["source_path"] = str(summary)
        dfs.append(df)

    meta = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    meta = _derive_metrics(_default_variant(_normalize_types(meta), "summary")) if not meta.empty else meta

    sweeps = _load_metrics_sweeps(base, "chronos2", meta, default_threshold=0.005)
    if meta.empty and sweeps.empty:
        return pd.DataFrame()
    frames = [f for f in (meta, sweeps) if not f.empty]
    return pd.concat(frames, ignore_index=True)


def load_ttm(base: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for summary in base.glob("summary_*.csv"):
        df = _read_csv(summary)
        _ensure_column(df, "run_name", summary.stem.replace("summary_", ""))
        df["model_family"] = "ttm"
        df["source_path"] = str(summary)
        dfs.append(df)

    for summary in base.rglob("summary_row.csv"):
        df = _read_csv(summary)
        _ensure_column(df, "run_name", summary.parent.name)
        df["model_family"] = "ttm"
        df["source_path"] = str(summary)
        dfs.append(df)

    meta = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    meta = _derive_metrics(_default_variant(_normalize_types(meta), "summary")) if not meta.empty else meta

    sweeps = _load_metrics_sweeps(base, "ttm", meta)
    if meta.empty and sweeps.empty:
        return pd.DataFrame()
    frames = [f for f in (meta, sweeps) if not f.empty]
    return pd.concat(frames, ignore_index=True)


def load_timesfm(base: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for summary in base.rglob("summary_row.csv"):
        # TimesFM stores per-run summaries only in run folders.
        df = _read_csv(summary)
        _ensure_column(df, "run_name", summary.parent.name)
        df["model_family"] = "timesfm"
        df["source_path"] = str(summary)
        dfs.append(df)

    meta = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    meta = _derive_metrics(_default_variant(_normalize_types(meta), "summary")) if not meta.empty else meta

    sweeps = _load_metrics_sweeps(base, "timesfm", meta)
    if meta.empty and sweeps.empty:
        return pd.DataFrame()
    frames = [f for f in (meta, sweeps) if not f.empty]
    return pd.concat(frames, ignore_index=True)


def load_lagllama(base: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for summary in base.glob("summary_*.csv"):
        df = _read_csv(summary)
        _ensure_column(df, "run_name", summary.stem.replace("summary_", ""))
        df["model_family"] = "lagllama"
        df["source_path"] = str(summary)
        dfs.append(df)

    for summary in base.rglob("summary_row.csv"):
        df = _read_csv(summary)
        _ensure_column(df, "run_name", summary.parent.name)
        df["model_family"] = "lagllama"
        df["source_path"] = str(summary)
        dfs.append(df)

    meta = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    meta = _derive_metrics(_default_variant(_normalize_types(meta), "summary")) if not meta.empty else meta

    sweeps = _load_metrics_sweeps(base, "lagllama", meta)
    if meta.empty and sweeps.empty:
        return pd.DataFrame()
    frames = [f for f in (meta, sweeps) if not f.empty]
    return pd.concat(frames, ignore_index=True)


def load_moirai(base: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for summary in base.glob("summary_*.csv"):
        df = _read_csv(summary)
        _ensure_column(df, "run_name", summary.stem.replace("summary_", ""))
        df["model_family"] = "moirai"
        df["source_path"] = str(summary)
        dfs.append(df)

    for summary in base.rglob("summary_row.csv"):
        df = _read_csv(summary)
        _ensure_column(df, "run_name", summary.parent.name)
        df["model_family"] = "moirai"
        df["source_path"] = str(summary)
        dfs.append(df)

    meta = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    meta = _derive_metrics(_default_variant(_normalize_types(meta), "summary")) if not meta.empty else meta

    sweeps = _load_metrics_sweeps(base, "moirai", meta)
    if meta.empty and sweeps.empty:
        return pd.DataFrame()
    frames = [f for f in (meta, sweeps) if not f.empty]
    return pd.concat(frames, ignore_index=True)


def load_moment(base: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for summary in base.glob("summary_*.csv"):
        df = _read_csv(summary)
        _ensure_column(df, "run_name", summary.stem.replace("summary_", ""))
        df["model_family"] = "moment"
        df["source_path"] = str(summary)
        dfs.append(df)

    for summary in base.rglob("summary_row.csv"):
        df = _read_csv(summary)
        _ensure_column(df, "run_name", summary.parent.name)
        df["model_family"] = "moment"
        df["source_path"] = str(summary)
        dfs.append(df)

    meta = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    meta = _derive_metrics(_default_variant(_normalize_types(meta), "summary")) if not meta.empty else meta

    sweeps = _load_metrics_sweeps(base, "moment", meta)
    if meta.empty and sweeps.empty:
        return pd.DataFrame()
    frames = [f for f in (meta, sweeps) if not f.empty]
    return pd.concat(frames, ignore_index=True)


def load_toto(base: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for summary in base.glob("summary_*.csv"):
        df = _read_csv(summary)
        _ensure_column(df, "run_name", summary.stem.replace("summary_", ""))
        df["model_family"] = "toto"
        df["source_path"] = str(summary)
        dfs.append(df)

    for summary in base.rglob("summary_row.csv"):
        df = _read_csv(summary)
        _ensure_column(df, "run_name", summary.parent.name)
        df["model_family"] = "toto"
        df["source_path"] = str(summary)
        dfs.append(df)

    meta = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    meta = _derive_metrics(_default_variant(_normalize_types(meta), "summary")) if not meta.empty else meta

    sweeps = _load_metrics_sweeps(base, "toto", meta)
    if meta.empty and sweeps.empty:
        return pd.DataFrame()
    frames = [f for f in (meta, sweeps) if not f.empty]
    return pd.concat(frames, ignore_index=True)


def combine_runs(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Drop duplicates if the same run/variant/threshold appears multiple times.
    keys: Tuple[str, ...] = ("model_family", "run_name", "variant", "threshold")
    use_keys = [k for k in keys if k in df.columns]
    if use_keys:
        df = df.drop_duplicates(subset=use_keys, keep="first")
    return df


def write_outputs_for_metric(df: pd.DataFrame, metric: str, top_k: int, output_dir: Path) -> None:
    metric_df = df.dropna(subset=[metric]) if metric in df.columns else df
    if metric_df.empty:
        print(f"No data available for metric '{metric}'.")
        return

    leaderboard = metric_df.sort_values(metric, ascending=False)
    leaderboard_path = output_dir / f"leaderboard_{metric}.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    best_by_horizon_ctx = (
        leaderboard.groupby(["horizon", "context_length"], as_index=False).head(1)
        if {"horizon", "context_length"}.issubset(leaderboard.columns)
        else pd.DataFrame()
    )
    if not best_by_horizon_ctx.empty:
        best_by_horizon_ctx.to_csv(output_dir / f"best_by_horizon_ctx_{metric}.csv", index=False)

    best_by_horizon = (
        leaderboard.groupby("horizon", as_index=False).head(top_k)
        if "horizon" in leaderboard.columns
        else pd.DataFrame()
    )
    if not best_by_horizon.empty:
        best_by_horizon.to_csv(output_dir / f"best_by_horizon_{metric}.csv", index=False)

    print(f"[{metric}] Wrote leaderboard to {leaderboard_path}")
    if not best_by_horizon_ctx.empty:
        print(f"[{metric}] Wrote best-by-(horizon,context) to {output_dir / f'best_by_horizon_ctx_{metric}.csv'}")
    if not best_by_horizon.empty:
        print(f"[{metric}] Wrote top-{top_k} per horizon to {output_dir / f'best_by_horizon_{metric}.csv'}")


def _family_colors() -> dict:
    return {
        "chronos": "#1f77b4",
        "chronos2": "#17becf",
        "timesfm": "#2ca02c",
        "ttm": "#ff7f0e",
        "lagllama": "#d62728",
        "moirai": "#9467bd",
        "moment": "#8c564b",
        "toto": "#bcbd22",
    }


def _clean_run_label(run_name: str) -> str:
    # Drop normalization markers that are not informative.
    for token in ("_normTrue", "_normFalse"):
        run_name = run_name.replace(token, "")
    return run_name


def _build_label_column(df: pd.DataFrame) -> pd.Series:
    run = df["run_name"].astype(str).apply(_clean_run_label)
    fam = df.get("model_family", pd.Series(["unknown"] * len(df)))
    base = fam.astype(str) + " | " + run
    if "threshold" not in df.columns:
        return base

    def _fmt_threshold(val) -> str:
        if pd.isna(val):
            return "thr=?"
        try:
            return f"thr={float(val):g}"
        except Exception:
            return f"thr={val}"

    thresholds = df["threshold"].apply(_fmt_threshold)
    return base + " | " + thresholds


def _equity_column_from_sweep(df: pd.DataFrame, threshold: float | None, use_tc: bool) -> str | None:
    if use_tc:
        candidates = [col for col in df.columns if col.endswith("_tc") and not col.endswith("_no_tc")]
    else:
        candidates = [col for col in df.columns if col.endswith("_no_tc")]
    if not candidates:
        return None
    if threshold is None or pd.isna(threshold):
        return candidates[0]

    def _parse_threshold(col: str) -> float | None:
        if "thr" not in col:
            return None
        try:
            return float(col.split("thr", 1)[1].split("_", 1)[0])
        except ValueError:
            return None

    for col in candidates:
        value = _parse_threshold(col)
        if value is not None and abs(value - float(threshold)) < 1e-6:
            return col
    return candidates[0]


def _equity_sweep_path(source_path: str) -> Path | None:
    source = Path(source_path)
    if source.is_dir():
        candidate = source / "equity_sweep.csv"
        if candidate.exists():
            return candidate
    if source.is_file() or source.parent.exists():
        candidate = source.parent / "equity_sweep.csv"
        if candidate.exists():
            return candidate
    return None


def _top_runs_with_equity(
    df: pd.DataFrame, metric: str, top_n: int | None
) -> List[pd.Series]:
    metric_df = df.dropna(subset=[metric]).copy()
    if metric_df.empty:
        return []
    if "source_path" not in metric_df.columns:
        return []
    metric_df = metric_df[metric_df["source_path"].astype(str).str.endswith("summary_row.csv")]
    if metric_df.empty:
        return []
    metric_df = metric_df.sort_values(metric, ascending=False)

    selected: List[pd.Series] = []
    for _, row in metric_df.iterrows():
        if top_n is not None and len(selected) >= top_n:
            break
        sweep_path = _equity_sweep_path(str(row["source_path"]))
        if sweep_path is None:
            continue
        selected.append(row)
    return selected


def _plot_equity_evolution(
    runs: List[pd.Series],
    metric: str,
    output_path: Path,
    title: str,
    use_tc: bool,
) -> None:
    if not runs:
        print(f"No equity sweep data found for {metric}; skipping equity evolution plot.")
        return

    color_cycle = plt.get_cmap("tab20").colors
    fig, ax = plt.subplots(figsize=(12, 6))
    buy_hold_plotted = False
    legend_handles = []
    legend_labels = []

    for idx, row in enumerate(runs):
        sweep_path = _equity_sweep_path(str(row["source_path"]))
        if sweep_path is None:
            continue
        sweep_df = _read_csv(sweep_path)
        eq_col = _equity_column_from_sweep(sweep_df, row.get("threshold"), use_tc=use_tc)
        if eq_col is None:
            continue

        x = np.arange(len(sweep_df))
        y = sweep_df[eq_col]
        label = _build_label_column(pd.DataFrame([row])).iloc[0]
        color = color_cycle[idx % len(color_cycle)]
        line = ax.plot(x, y, label=label, color=color, alpha=0.9, linewidth=1.6)[0]
        legend_handles.append(line)
        legend_labels.append(label)

        if not buy_hold_plotted and "y_true" in sweep_df.columns:
            starting_capital = row.get("starting_capital")
            if pd.isna(starting_capital):
                starting_capital = 1.0
            base = sweep_df["y_true"].iloc[0]
            if pd.notna(base) and base != 0:
                buy_hold = starting_capital * sweep_df["y_true"] / base
                line = ax.plot(x, buy_hold, label="buy & hold", color="black", linewidth=2.0, alpha=0.9)[0]
                legend_handles.append(line)
                legend_labels.append("buy & hold")
                buy_hold_plotted = True

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("equity")
    ax.legend(legend_handles, legend_labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _select_leaderboard_rows(df: pd.DataFrame, metric: str, top_n: int) -> pd.DataFrame:
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return plot_df
    if "variant" in plot_df.columns:
        group_keys = [k for k in ("model_family", "run_name", "threshold") if k in plot_df.columns]
        if group_keys:
            def _pick_variant(group: pd.DataFrame) -> pd.Series:
                summary = group[group["variant"] == "summary"]
                if not summary.empty:
                    return summary.sort_values(metric, ascending=False).iloc[0]
                return group.sort_values(metric, ascending=False).iloc[0]

            plot_df = plot_df.groupby(group_keys, as_index=False).apply(_pick_variant).reset_index(drop=True)
    return plot_df.sort_values(metric, ascending=False).head(top_n)


def _match_equity_rows(
    df: pd.DataFrame, leaderboard: pd.DataFrame, metric: str
) -> List[pd.Series]:
    if leaderboard.empty:
        return []
    if "source_path" not in df.columns:
        return []

    equity_df = df[df["source_path"].astype(str).str.endswith("summary_row.csv")].copy()
    selected: List[pd.Series] = []
    for _, row in leaderboard.iterrows():
        matches = equity_df
        for key in ("model_family", "run_name"):
            if key in leaderboard.columns and key in equity_df.columns:
                matches = matches[matches[key] == row[key]]
        if "threshold" in leaderboard.columns and "threshold" in equity_df.columns:
            threshold = row.get("threshold")
            if pd.isna(threshold):
                matches = matches[matches["threshold"].isna()]
            else:
                matches = matches[np.isclose(matches["threshold"].astype(float), float(threshold), rtol=0, atol=1e-6)]
        if matches.empty:
            base_path = Path(str(row.get("source_path", ""))).parent
            run_name = str(row.get("run_name", ""))
            if run_name and base_path.name != run_name:
                run_dir = base_path / run_name
            else:
                run_dir = base_path
            if _equity_sweep_path(str(run_dir)) is not None:
                fallback = row.copy()
                fallback["source_path"] = str(run_dir)
                selected.append(fallback)
            continue
        matches = matches.sort_values(metric, ascending=False)
        selected.append(matches.iloc[0])
    return selected


def generate_equity_evolution_plots(df: pd.DataFrame, top_n: int | None, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if top_n is None:
        top_n = 10
    tc_leaderboard = _select_leaderboard_rows(df, "final_equity_tc", top_n)
    tc_runs = _match_equity_rows(df, tc_leaderboard, "final_equity_tc")
    _plot_equity_evolution(
        tc_runs,
        metric="final_equity_tc",
        output_path=plots_dir / "equity_evolution_tc.png",
        title=f"Equity evolution ({len(tc_runs)} runs by final_equity_tc)",
        use_tc=True,
    )

    no_tc_leaderboard = _select_leaderboard_rows(df, "final_equity_no_tc", top_n)
    no_tc_runs = _match_equity_rows(df, no_tc_leaderboard, "final_equity_no_tc")
    _plot_equity_evolution(
        no_tc_runs,
        metric="final_equity_no_tc",
        output_path=plots_dir / "equity_evolution_no_tc.png",
        title=f"Equity evolution ({len(no_tc_runs)} runs by final_equity_no_tc)",
        use_tc=False,
    )


def generate_plots(df: pd.DataFrame, metric: str, top_k: int, output_dir: Path) -> None:
    if metric not in df.columns:
        print(f"Metric '{metric}' not found; skipping plots.")
        return

    metric_df = df.dropna(subset=[metric]).copy()
    if metric_df.empty:
        print(f"No data available for metric '{metric}'; skipping plots.")
        return

    colors = _family_colors()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Leaderboard bar plot (top max(top_k, 10) entries).
    top_n = max(top_k, 10)
    leaderboard = _select_leaderboard_rows(metric_df, metric, top_n)
    leaderboard_labels = _build_label_column(leaderboard)
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(leaderboard) + 2))
    ax.barh(
        leaderboard_labels,
        leaderboard[metric],
        color=[colors.get(fam, "#7f7f7f") for fam in leaderboard["model_family"]],
    )
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(f"Top {top_n} runs by {metric}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(plots_dir / f"leaderboard_{metric}.png", dpi=150)
    plt.close(fig)

    # Scatter: metric vs horizon, sized by context_length, colored by family.
    if {"horizon", "context_length"}.issubset(metric_df.columns):
        fig, ax = plt.subplots(figsize=(10, 6))
        for family, group in metric_df.groupby("model_family"):
            ax.scatter(
                group["horizon"],
                group[metric],
                s=group["context_length"].fillna(0) * 0.2,
                alpha=0.7,
                label=family,
                color=colors.get(family, None),
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_xlabel("Horizon")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs horizon (marker size ~ context_length)")
        ax.set_xticks([1, 3, 5, 20])
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"scatter_horizon_{metric}.png", dpi=150)
        plt.close(fig)

    # Box plot by model family for metric distribution.
    fig, ax = plt.subplots(figsize=(8, 6))
    metric_df.boxplot(column=metric, by="model_family", ax=ax, grid=False)
    ax.set_title(f"{metric} distribution by model family")
    ax.set_xlabel("Model family")
    ax.set_ylabel(metric)
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(plots_dir / f"box_by_family_{metric}.png", dpi=150)
    plt.close(fig)

    print(f"Wrote plots to {plots_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Chronos, Chronos2, TimesFM, TTM, Lag-Llama, Moirai, MOMENT, and TOTO results."
    )
    parser.add_argument("--chronos-dir", type=Path, default=Path("chronos-results"), help="Path to chronos results.")
    parser.add_argument("--chronos2-dir", type=Path, default=Path("chronos2-results"), help="Path to chronos2 results.")
    parser.add_argument("--timesfm-dir", type=Path, default=Path("timesfm-results"), help="Path to timesfm results.")
    parser.add_argument("--ttm-dir", type=Path, default=Path("ttm-results"), help="Path to ttm results.")
    parser.add_argument(
        "--lagllama-dir", type=Path, default=Path("lagllama-results"), help="Path to lagllama results."
    )
    parser.add_argument("--moirai-dir", type=Path, default=Path("moirai-results"), help="Path to moirai results.")
    parser.add_argument("--moment-dir", type=Path, default=Path("moment-results"), help="Path to moment results.")
    parser.add_argument("--toto-dir", type=Path, default=Path("toto-results"), help="Path to toto results.")
    parser.add_argument("--metric", type=str, default="final_equity_tc", help="Metric used for leaderboards.")
    parser.add_argument(
        "--extra-metrics",
        nargs="+",
        default=["final_equity_no_tc"],
        help="Additional metrics to render leaderboards/plots for (e.g., roi_no_tc roi_tc).",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k rows to keep per horizon.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models_comparison"),
        help="Directory where comparison CSVs will be written.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=["chronos", "chronos2", "timesfm", "ttm", "lagllama", "moirai", "moment", "toto"],
        help="Which families to include (any of: chronos chronos2 timesfm ttm lagllama moirai moment toto).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="If set, skip generating plots into <output-dir>/plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames: List[pd.DataFrame] = []

    include = {name.lower() for name in args.include}
    if "chronos" in include and args.chronos_dir.exists():
        frames.append(load_chronos(args.chronos_dir))
    if "chronos2" in include and args.chronos2_dir.exists():
        frames.append(load_chronos2(args.chronos2_dir))
    if "ttm" in include and args.ttm_dir.exists():
        frames.append(load_ttm(args.ttm_dir))
    if "timesfm" in include and args.timesfm_dir.exists():
        frames.append(load_timesfm(args.timesfm_dir))
    if "lagllama" in include and args.lagllama_dir.exists():
        frames.append(load_lagllama(args.lagllama_dir))
    if "moirai" in include and args.moirai_dir.exists():
        frames.append(load_moirai(args.moirai_dir))
    if "moment" in include and args.moment_dir.exists():
        frames.append(load_moment(args.moment_dir))
    if "toto" in include and args.toto_dir.exists():
        frames.append(load_toto(args.toto_dir))

    combined = combine_runs(frames)
    if combined.empty:
        raise SystemExit("No data loaded. Check paths and included families.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.output_dir / "combined_results.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Wrote combined results to {combined_path}")

    metrics = [args.metric, *args.extra_metrics]
    seen = set()
    for metric in metrics:
        if metric in seen:
            continue
        seen.add(metric)
        write_outputs_for_metric(combined, metric, args.top_k, args.output_dir)
        if not args.no_plots:
            generate_plots(combined, metric, args.top_k, args.output_dir)
    if not args.no_plots:
        top_n = max(args.top_k, 10)
        generate_equity_evolution_plots(combined, top_n=top_n, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
