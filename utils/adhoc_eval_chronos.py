from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import evaluate_forecast
from thresholds import DEFAULT_THRESHOLD


# config

RESULTS_ROOT = Path("chronos-results")
OUT_ROOT = RESULTS_ROOT / "_adhoc_eval"

CTX: int = 512          # context length used in sweep
H: int = 1              # 1, 3, 5, 20
THRESHOLD: float = DEFAULT_THRESHOLD
USE_TC: bool = False    # False -> equity_no_tc; True -> equity_tc

# Use timezone-aware cutoff to match the UTC timestamps loaded from CSVs
CUT_TIMESTAMP = pd.Timestamp("2018-03-27 07:00:00", tz="UTC")


# helpers

def discover_runs(root: Path) -> List[Path]:
    runs = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name.startswith("_"):
            continue
        if (p / "chronos_step1.csv").exists() and (p / "chronos_wide.csv").exists():
            runs.append(p)
    return runs

def parse_meta(name: str) -> Tuple[int | None, int | None]:
    name = name.lower()
    try:
        ctx = int(name.split("ctx", 1)[1].split("_", 1)[0])
        h = int(name.split("_h", 1)[1].split("_", 1)[0])
        return ctx, h
    except Exception:
        return None, None

def pick_run_dir(root: Path, ctx: int, h: int) -> Path | None:
    for p in discover_runs(root):
        c, hh = parse_meta(p.name)
        if c == ctx and hh == h:
            return p
    return None

def list_available(root: Path) -> Dict[int, List[int]]:
    m: Dict[int, set] = {}
    for p in discover_runs(root):
        c, hh = parse_meta(p.name)
        if c is None or hh is None:
            continue
        m.setdefault(c, set()).add(hh)
    return {c: sorted(list(hs)) for c, hs in sorted(m.items())}

def load_run(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    step = pd.read_csv(run_dir / "chronos_step1.csv")
    step["timestamp"] = pd.to_datetime(step["timestamp"], errors="coerce")

    wide = pd.read_csv(run_dir / "chronos_wide.csv")
    if "base_timestamp" in wide.columns:
        wide["base_timestamp"] = pd.to_datetime(wide["base_timestamp"], errors="coerce")

    if len(step) != len(wide):
        raise ValueError(f"Row mismatch in {run_dir}: step1={len(step)}, wide={len(wide)}")
    return step, wide

def hard_cut(step: pd.DataFrame, wide: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask = step["timestamp"] >= CUT_TIMESTAMP
    return step.loc[mask].reset_index(drop=True), wide.loc[mask].reset_index(drop=True)

def mean_h_pred(wide: pd.DataFrame) -> np.ndarray:
    pred_cols = [c for c in wide.columns if c.startswith("y_pred_h")]
    if not pred_cols:
        raise ValueError("No y_pred_h* columns in chronos_wide.csv")

    def _idx(c: str) -> int:
        try:
            return int(c.split("h", 1)[1])
        except Exception:
            return 10**9

    pred_cols = sorted(pred_cols, key=_idx)
    return wide[pred_cols].astype(float).to_numpy().mean(axis=1)


def json_ready(x):
    import numpy as np
    import pandas as pd

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(x).isoformat()
    if isinstance(x, pd.Series):
        return [json_ready(v) for v in x.tolist()]
    if isinstance(x, (np.ndarray, list, tuple)):
        return [json_ready(v) for v in x]
    if isinstance(x, dict):
        return {k: json_ready(v) for k, v in x.items()}
    return x

def summarize_metrics(m: dict) -> dict:
    want = [
        "final_equity_no_tc",
        "final_equity_tc",
        "total_fees_paid",
        "arc",
        "asd",
        "mdd",
        "ir_star",
        "ir_starstar",
        "pct_long", "pct_short", "pct_flat",
        "trades_long", "trades_short", "trades_flat", "trades_total",
    ]
    out = {}
    for k in want:
        if k in m and np.isscalar(m[k]):
            out[k] = float(m[k])
    for k in ("n_samples", "bars", "period_days"):
        if k in m and np.isscalar(m[k]):
            out[k] = float(m[k])
    return out


# core

def eval_one(root: Path, ctx: int, h: int, thr: float, use_tc: bool) -> None:
    run_dir = pick_run_dir(root, ctx, h)
    if run_dir is None:
        avail = list_available(root)
        print(f"[adhoc] No run folder for ctx{ctx}, h{h}. "
              f"Available (context -> horizons): {avail}")
        return

    subdir = OUT_ROOT / f"ctx{ctx}_h{h}_thr{thr:.5f}_{'tc' if use_tc else 'notc'}"
    subdir.mkdir(parents=True, exist_ok=True)

    print(f"[adhoc] Evaluating ctx{ctx}, h{h}, thr={thr:.5f}, "
          f"{'with TC' if use_tc else 'no TC'}")
    print(f"[adhoc] Input run dir: {run_dir}")
    print(f"[adhoc] Output dir: {subdir}")

    step, wide = load_run(run_dir)
    step, wide = hard_cut(step, wide)

    timestamps = step["timestamp"]
    y_true = step["y_true"].astype(float).to_numpy()
    y_pred_step1 = step["y_pred"].astype(float).to_numpy()

    start_cap, fee_rate = 100_000.0, 0.001
    try:
        meta = json.loads((run_dir / "metrics.json").read_text())
        start_cap = float(meta.get("starting_capital", start_cap))
        fee_rate = float(meta.get("fee_rate", fee_rate))
    except Exception:
        pass

    m_step = evaluate_forecast(
        y_true=y_true,
        y_pred=y_pred_step1,
        timestamps=timestamps,
        starting_capital=start_cap,
        threshold=float(thr),
        fee_rate=fee_rate,
    )

    has_mean = len([c for c in wide.columns if c.startswith("y_pred_h")]) > 1
    m_mean = None
    if has_mean:
        y_pred_mean = mean_h_pred(wide)
        m_mean = evaluate_forecast(
            y_true=y_true,
            y_pred=y_pred_mean,
            timestamps=timestamps,
            starting_capital=start_cap,
            threshold=float(thr),
            fee_rate=fee_rate,
        )

    use_key = "equity_tc" if use_tc else "equity_no_tc"
    df_out = pd.DataFrame({"timestamp": timestamps, "y_true": y_true})
    df_out[f"eq_step1_thr{thr:.5f}_{'tc' if use_tc else 'no_tc'}"] = np.asarray(m_step[use_key])
    if m_mean is not None:
        df_out[f"eq_mean_h_thr{thr:.5f}_{'tc' if use_tc else 'no_tc'}"] = np.asarray(m_mean[use_key])
    df_out.to_csv(subdir / "adhoc_equity.csv", index=False)

    dump = {
        "ctx": int(ctx),
        "h": int(h),
        "threshold": float(thr),
        "use_tc": bool(use_tc),
        "starting_capital": float(start_cap),
        "fee_rate": float(fee_rate),
        "step_metrics": summarize_metrics(m_step),
        "mean_metrics": summarize_metrics(m_mean) if m_mean else None,
    }
    (subdir / "summary.json").write_text(json.dumps(json_ready(dump), indent=2))

    try:
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, m_step[use_key], label="step1", linewidth=1.5)
        if m_mean is not None:
            plt.plot(timestamps, m_mean[use_key], label="mean_h", linewidth=1.2)
        plt.xlabel("timestamp")
        plt.ylabel("equity")
        plt.title(f"ctx{ctx}, h{h}, thr={thr:.5f}, {'TC' if use_tc else 'no TC'}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(subdir / "equity.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"[adhoc] Plot failed: {e}")


def main():
    print("[adhoc] Chronos adhoc eval")
    print(f"[adhoc] Available runs (context -> horizons): {list_available(RESULTS_ROOT)}")
    eval_one(RESULTS_ROOT, ctx=CTX, h=H, thr=THRESHOLD, use_tc=USE_TC)


if __name__ == "__main__":
    main()
