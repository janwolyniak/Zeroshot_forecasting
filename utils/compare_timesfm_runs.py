from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================== config ======================

RESULTS_ROOT = Path("timesfm-results")

BASE_OUT = RESULTS_ROOT / "_comparisons"
HEATMAPS_DIR = BASE_OUT / "heatmaps"
OVERLAYS_DIR = BASE_OUT / "overlays"
SMALLS_DIR   = BASE_OUT / "small_multiples"
SENS_DIR     = BASE_OUT / "sensitivity"

for d in (HEATMAPS_DIR, OVERLAYS_DIR, SMALLS_DIR, SENS_DIR):
    d.mkdir(parents=True, exist_ok=True)

THRESHOLDS = (0.01, 0.015, 0.02, 0.025, 0.03)

TOP_K = 6
GROUP_BY = "h"
GROUP_VALUE = None   # e.g. "5" if you ever want to filter mean_h overlays by horizon


# ====================== loading & metadata ======================

# "_normTrue/False" is optional; we ignore it semantically.
RUN_RE = re.compile(r"ctx(?P<context>\d+)_h(?P<h>\d+)(?:_norm(?P<norm>True|False))?")

def parse_run_name(name: str) -> Dict[str, str]:
    m = RUN_RE.search(name)
    if not m:
        return {"context": None, "h": None, "norm": None}
    d = m.groupdict()
    return {"context": d["context"], "h": d["h"], "norm": d["norm"]}

def discover_runs(results_root: Path) -> List[Path]:
    runs = []
    for p in sorted(results_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "metrics_sweep.csv").exists() and (p / "equity_sweep.csv").exists():
            runs.append(p)
    return runs

def load_metrics_table(runs: List[Path]) -> pd.DataFrame:
    rows = []
    for rd in runs:
        meta = parse_run_name(rd.name)
        df = pd.read_csv(rd / "metrics_sweep.csv")
        df["run"] = rd.name
        df["context"] = meta["context"]
        df["h"] = meta["h"]
        df["norm"] = meta["norm"]
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_equity_tables(runs: List[Path]) -> Dict[str, pd.DataFrame]:
    return {rd.name: pd.read_csv(rd / "equity_sweep.csv", parse_dates=["timestamp"]) for rd in runs}

def build_context_colors(metrics_df: pd.DataFrame) -> Dict[int, str]:
    """
    Assign a consistent colour to each context.
    """
    m = metrics_df.copy()
    m["context"] = pd.to_numeric(m["context"], errors="coerce")
    ctxs = sorted(m["context"].dropna().unique().astype(int).tolist())
    base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not base_colors:
        base_colors = [f"C{i}" for i in range(10)]
    return {ctx: base_colors[i % len(base_colors)] for i, ctx in enumerate(ctxs)}


# ====================== heatmaps ======================

def _heatmap_ax(ax, data: pd.DataFrame, title: str):
    im = ax.imshow(data.values, aspect="auto")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_title(title, fontsize=10)
    return im

def heatmaps_all(metrics_df: pd.DataFrame, variant: str = "step1", use_tc: bool = False) -> None:
    """
    Heatmaps for a given variant.

    step1:
        - horizon does NOT matter.
        - We collapse over h and produce ONE heatmap with:
              rows   = thresholds
              cols   = context
              values = final_equity_* (non-TC or TC)
    mean_horizon:
        - keep the old style: for each threshold a (h × ctx) heatmap,
          plus a montage of all thresholds.
    """
    if metrics_df.empty:
        print("[compare] no metrics to plot (heatmaps)")
        return

    value_col = "final_equity_tc" if use_tc else "final_equity_no_tc"

    m = metrics_df.copy()
    m = m[m["variant"] == variant]
    m["context"] = pd.to_numeric(m["context"], errors="coerce")
    m["h"] = pd.to_numeric(m["h"], errors="coerce")

    if m.empty:
        print(f"[compare] no rows for variant={variant}")
        return

    if variant == "step1":
        # collapse over h: step1 is identical across horizons for given ctx
        g = (m.groupby(["threshold", "context"], as_index=False)[value_col]
               .mean())
        g = g[g["context"].notna()]
        g["context"] = g["context"].astype(int)
        g = g.sort_values(["threshold", "context"])
        pivot = g.pivot(index="threshold", columns="context", values=value_col)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = _heatmap_ax(
            ax,
            pivot,
            f"step1 · {'TC' if use_tc else 'no TC'} · metric={value_col}",
        )
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_ylabel("threshold")
        ax.set_xlabel("context")
        fig.tight_layout()
        fig.savefig(
            HEATMAPS_DIR / f"heatmap_step1_{'tc' if use_tc else 'notc'}_all_thresholds.png",
            dpi=150,
        )
        plt.close(fig)
        return

    # ----- mean_horizon: one per threshold + montage (old behaviour) -----
    pivots = []
    for thr in THRESHOLDS:
        df_thr = m[m["threshold"] == float(thr)]
        if df_thr.empty:
            continue
        pivot = (
            df_thr.pivot_table(
                index="h", columns="context", values=value_col, aggfunc="mean"
            )
            .sort_index()
            .sort_index(axis=1)
        )
        pivots.append((thr, pivot))

        fig, ax = plt.subplots(figsize=(10, 6))
        im = _heatmap_ax(
            ax,
            pivot,
            f"mean_horizon · {'TC' if use_tc else 'no TC'} · thr={thr:.3f}",
        )
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_ylabel("horizon")
        ax.set_xlabel("context")
        fig.tight_layout()
        fig.savefig(
            HEATMAPS_DIR / f"heatmap_mean_h_{'tc' if use_tc else 'notc'}_thr{thr:.3f}.png",
            dpi=150,
        )
        plt.close(fig)

    if not pivots:
        return

    cols = min(3, len(pivots))
    rows = int(np.ceil(len(pivots) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    vmin = min(p[1].values.min() for p in pivots)
    vmax = max(p[1].values.max() for p in pivots)

    for ax, (thr, pivot) in zip(axes, pivots):
        im = ax.imshow(pivot.values, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(int))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.astype(int), rotation=45, ha="right")
        ax.set_title(f"thr={thr:.3f}", fontsize=10)
        ax.set_ylabel("horizon")
        ax.set_xlabel("context")

    for ax in axes[len(pivots) :]:
        ax.axis("off")

    fig.suptitle(f"HEATMAPS · mean_horizon · {'TC' if use_tc else 'no TC'}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    cbar.set_label(value_col)
    fig.savefig(
        HEATMAPS_DIR / f"heatmaps_all_mean_h_{'tc' if use_tc else 'notc'}.png",
        dpi=150,
    )
    plt.close(fig)


# ====================== overlays ======================

def _legend_label(row, variant: str) -> str:
    ctx = row.get("context", "")
    h = row.get("h", "")
    if variant == "step1":
        return f"ctx{ctx}"
    return f"ctx{ctx}_h{h}"

def overlays_all(
    equities: Dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
    ctx_colors: Dict[int, str],
    variant: str = "step1",
    use_tc: bool = False,
    top_k: int = TOP_K,
    group_by: str = GROUP_BY,
    group_value: str | None = GROUP_VALUE,
) -> None:
    """
    For each threshold, overlay Top-K runs and make an all-threshold montage.

    - Colours: per-context (consistent across plots).
    - Line style: step1 solid, mean_horizon dashed.
    - step1 de-dupes by context (one run per ctx).
    """
    value_col = "final_equity_tc" if use_tc else "final_equity_no_tc"
    suffix = "tc" if use_tc else "notc"
    eq_variant = "mean_h" if variant == "mean_horizon" else "step1"

    def eq_col(thr: float) -> str:
        return f"eq_{eq_variant}_thr{thr:.3f}_{'tc' if use_tc else 'no_tc'}"

    m = metrics_df.copy()
    m = m[m["variant"] == variant]
    m["context"] = pd.to_numeric(m["context"], errors="coerce")
    m["h"] = pd.to_numeric(m["h"], errors="coerce")

    if variant != "step1" and group_value is not None:
        m = m[m[group_by] == group_value]

    if m.empty:
        print("[compare] no metrics to plot (overlays)")
        return

    def reps_for_threshold(thr: float) -> pd.DataFrame:
        df = m[m["threshold"] == float(thr)]
        if df.empty:
            return df
        if variant == "step1":
            df = (
                df.sort_values(value_col, ascending=False)
                .drop_duplicates(subset=["context"], keep="first")
            )
        else:
            df = df.sort_values(value_col, ascending=False)
        return df.head(top_k)

    # individual overlays
    for thr in THRESHOLDS:
        df = reps_for_threshold(thr)
        if df.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        for _, row in df.iterrows():
            run = row["run"]
            eq = equities.get(run)
            if eq is None or eq_col(thr) not in eq.columns:
                continue
            ctx = int(row["context"]) if pd.notna(row["context"]) else None
            color = ctx_colors.get(ctx, None)
            linestyle = "-" if variant == "step1" else "--"
            ax.plot(
                eq["timestamp"],
                eq[eq_col(thr)].values,
                label=_legend_label(row, variant),
                color=color,
                linestyle=linestyle,
            )
        ttl = f"Top-{len(df)} · {variant} · {'TC' if use_tc else 'no TC'} · thr={thr:.3f}"
        if variant != "step1" and group_value is not None:
            ttl += f" · {group_by}={group_value}"
        ax.set_title(ttl)
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(
            OVERLAYS_DIR
            / f"overlay_top{top_k}_{variant}_{suffix}_thr{thr:.3f}"
            f"{f'_{group_by}{group_value}' if (variant!='step1' and group_value) else ''}.png",
            dpi=150,
        )
        plt.close(fig)

    # montage
    cols = min(3, len(THRESHOLDS))
    rows = int(np.ceil(len(THRESHOLDS) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.2 * rows), sharey=False, sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, thr in zip(axes, THRESHOLDS):
        df = reps_for_threshold(thr)
        for _, row in df.iterrows():
            run = row["run"]
            eq = equities.get(run)
            if eq is None or eq_col(thr) not in eq.columns:
                continue
            ctx = int(row["context"]) if pd.notna(row["context"]) else None
            color = ctx_colors.get(ctx, None)
            linestyle = "-" if variant == "step1" else "--"
            ax.plot(
                eq["timestamp"],
                eq[eq_col(thr)].values,
                label=_legend_label(row, variant),
                color=color,
                linestyle=linestyle,
            )
        ax.set_title(f"thr={thr:.3f}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=7, loc="best", frameon=False)
        if rows > 1:
            ax.tick_params(labelbottom=True)

    for ax in axes[len(THRESHOLDS) :]:
        ax.axis("off")

    big_title = f"OVERLAYS (Top-{top_k}) · {variant} · {'TC' if use_tc else 'no TC'}"
    if variant != "step1" and group_value is not None:
        big_title += f" · {group_by}={group_value}"
    fig.suptitle(big_title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(
        OVERLAYS_DIR
        / f"overlays_all_top{top_k}_{variant}_{suffix}"
        f"{f'_{group_by}{group_value}' if (variant!='step1' and group_value) else ''}.png",
        dpi=150,
    )
    plt.close(fig)


# ====================== small multiples ======================

def _title_label(run_name: str, variant: str) -> str:
    meta = parse_run_name(run_name)
    ctx = meta.get("context", "")
    h = meta.get("h", "")
    if variant == "step1":
        return f"ctx{ctx}"
    return f"ctx{ctx}_h{h}"

def small_multiples(
    equities: Dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
    ctx_colors: Dict[int, str],
    threshold: float,
    variant: str = "step1",
    use_tc: bool = False,
) -> None:
    """
    One big montage (single PNG) per (variant, threshold).

    step1:
        - de-duplicate by context (one run per ctx, best final_equity_no_tc).
    mean_horizon:
        - include all runs with that equity column.
    """
    eq_col = f"eq_{'mean_h' if variant=='mean_horizon' else 'step1'}_thr{threshold:.3f}_{'tc' if use_tc else 'no_tc'}"

    m = metrics_df.copy()
    m = m[(m["variant"] == variant) & (m["threshold"] == float(threshold))]
    m["context"] = pd.to_numeric(m["context"], errors="coerce")
    m["h"] = pd.to_numeric(m["h"], errors="coerce")

    if variant == "step1":
        # best per context by final_equity_no_tc
        if "final_equity_no_tc" in m.columns:
            idx = m.groupby("context")["final_equity_no_tc"].idxmax()
        else:
            idx = m.groupby("context")["final_equity_tc"].idxmax()
        m = m.loc[idx]

    # sorted runs by chosen ranking metric (desc)
    rank_metric = "ir_star" if "ir_star" in m.columns else "final_equity_no_tc"
    m = m.sort_values(rank_metric, ascending=False)

    runs = [
        r
        for r in m["run"].tolist()
        if eq_col in equities.get(r, pd.DataFrame()).columns
    ]

    if not runs:
        print(f"[compare] no runs contain column {eq_col}")
        return

    n = len(runs)
    max_cols = 8
    cols = min(max_cols, max(1, int(np.ceil(np.sqrt(n)))))
    rows = int(np.ceil(n / cols))

    fig_w = max(10, 2.6 * cols)
    fig_h = max(6, 1.8 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), sharex=True, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, run in zip(axes, runs):
        df = equities[run]
        meta = parse_run_name(run)
        ctx = int(meta.get("context")) if meta.get("context") is not None else None
        color = ctx_colors.get(ctx, None)
        ax.plot(df["timestamp"], df[eq_col].values, color=color)
        ax.set_title(_title_label(run, variant), fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.25)

    for ax in axes[len(runs) :]:
        ax.axis("off")

    fig.suptitle(
        f"{variant} · thr={threshold:.3f} · {'TC' if use_tc else 'no TC'}  (all runs)",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = SMALLS_DIR / f"smallmultiples_all_{variant}_{'tc' if use_tc else 'notc'}_thr{threshold:.3f}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ====================== threshold sensitivity ======================

def threshold_sensitivity(
    metrics_df: pd.DataFrame,
    ctx_colors: Dict[int, str],
    metric: str = "final_equity_no_tc",
) -> None:
    """
    Strip charts: for each (ctx, h), plot metric vs threshold, for step1 and mean_horizon.

    Saved under _comparisons/sensitivity/sensitivity_ctx{ctx}_h{h}_{metric}.png
    """
    m = metrics_df.copy()
    m["context"] = pd.to_numeric(m["context"], errors="coerce")
    m["h"] = pd.to_numeric(m["h"], errors="coerce")

    for (ctx, h), group in m.groupby(["context", "h"]):
        ctx_int = int(ctx)
        h_int = int(h)
        fig, ax = plt.subplots(figsize=(6, 4))

        for variant, linestyle in [("step1", "-"), ("mean_horizon", "--")]:
            g = group[group["variant"] == variant].sort_values("threshold")
            if g.empty or metric not in g.columns:
                continue
            ax.plot(
                g["threshold"],
                g[metric],
                marker="o",
                linestyle=linestyle,
                color=ctx_colors.get(ctx_int, None),
                label=variant,
            )

        if not ax.has_data():
            plt.close(fig)
            continue

        ax.set_title(f"ctx{ctx_int}_h{h_int} · {metric} vs threshold")
        ax.set_xlabel("threshold")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            SENS_DIR / f"sensitivity_ctx{ctx_int}_h{h_int}_{metric}.png",
            dpi=150,
        )
        plt.close(fig)


# ====================== global leaders ======================

def write_global_leaders(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """
    One-row-per (ctx, h) summary of 'best' config:

      - best_threshold
      - best_variant
      - ir_star (if available) used as ranking metric,
        with fallback to final_equity_no_tc.
    """
    m = metrics_df.copy()
    m["context"] = pd.to_numeric(m["context"], errors="coerce")
    m["h"] = pd.to_numeric(m["h"], errors="coerce")

    leaders = []
    for (ctx, h), group in m.groupby(["context", "h"]):
        ctx_int = int(ctx)
        h_int = int(h)

        group = group.copy()
        if "ir_star" in group.columns:
            group["score"] = group["ir_star"]
        else:
            group["score"] = group["final_equity_no_tc"]
        # fallback where IR is NaN
        group["score"] = group["score"].fillna(group["final_equity_no_tc"])
        idx = group["score"].idxmax()
        best = group.loc[idx]

        leaders.append(
            {
                "context": ctx_int,
                "h": h_int,
                "best_variant": best["variant"],
                "best_threshold": best["threshold"],
                "score": best["score"],
                "ir_star": best.get("ir_star", np.nan),
                "final_equity_no_tc": best.get("final_equity_no_tc", np.nan),
                "final_equity_tc": best.get("final_equity_tc", np.nan),
            }
        )

    df_lead = pd.DataFrame(leaders).sort_values(["context", "h"])
    df_lead.to_csv(out_path, index=False)
    print(f"[compare] wrote leaders table -> {out_path}")


# ====================== driver ======================

def main():
    runs = discover_runs(RESULTS_ROOT)
    print(f"[compare] found {len(runs)} runs")
    metrics_df = load_metrics_table(runs)
    equities = load_equity_tables(runs)

    if metrics_df.empty:
        print("[compare] no metrics found; nothing to do")
        return

    # consistent colour per context
    ctx_colors = build_context_colors(metrics_df)

    # HEATMAPS (non-TC)
    heatmaps_all(metrics_df, variant="step1", use_tc=False)
    heatmaps_all(metrics_df, variant="mean_horizon", use_tc=False)

    # OVERLAYS (non-TC)
    overlays_all(
        equities,
        metrics_df,
        ctx_colors,
        variant="step1",
        use_tc=False,
        top_k=TOP_K,
        group_by=GROUP_BY,
        group_value=GROUP_VALUE,
    )
    overlays_all(
        equities,
        metrics_df,
        ctx_colors,
        variant="mean_horizon",
        use_tc=False,
        top_k=TOP_K,
        group_by=GROUP_BY,
        group_value=GROUP_VALUE,
    )

    # SMALL MULTIPLES (non-TC)
    for thr in THRESHOLDS:
        small_multiples(
            equities,
            metrics_df,
            ctx_colors,
            threshold=thr,
            variant="step1",
            use_tc=False,
        )
        small_multiples(
            equities,
            metrics_df,
            ctx_colors,
            threshold=thr,
            variant="mean_horizon",
            use_tc=False,
        )

    # THRESHOLD-SENSITIVITY CHARTS (non-TC)
    threshold_sensitivity(metrics_df, ctx_colors, metric="final_equity_no_tc")

    # GLOBAL LEADERS TABLE
    write_global_leaders(metrics_df, BASE_OUT / "leaders_by_ctx_h.csv")

    print(f"[compare] outputs in: {BASE_OUT.resolve()}")
    print(f"          heatmaps:       {HEATMAPS_DIR.resolve()}")
    print(f"          overlays:       {OVERLAYS_DIR.resolve()}")
    print(f"          small-multiples:{SMALLS_DIR.resolve()}")
    print(f"          sensitivity:    {SENS_DIR.resolve()}")

if __name__ == "__main__":
    main()