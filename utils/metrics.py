"""
metrics.py
Simplified backtest + risk metrics for direction-only trading strategies.

Assumptions:
- We derive a trading signal from forecasted price vs current price:
      edge[t] = y_pred[t] / y_true[t] - 1
  If |edge[t]| > threshold, we take a directional view (+1 long, -1 short),
  else we stay flat (0).

- We do NOT trade immediately. The position we actually hold at time t
  is yesterday's signal (one-bar execution lag). This avoids lookahead.

- Per-period asset return:
      ret[t] = y_true[t] / y_true[t-1] - 1
      ret[0] = 0
  Strategy P&L contribution at t is exec_pos[t] * ret[t].

- Transaction costs:
  When we CHANGE executed position, we pay:
      fee = fee_rate * equity_before_change * |Δpos|
  where Δpos is the step change in exec_pos. Going from +1 → -1
  has |Δpos| = 2 (close long, open short).

Metrics we output:
- equity_no_tc : equity curve without transaction costs
- equity_tc    : equity curve with transaction costs
- fee          : per-step fees paid
- cum_fee      : cumulative fees
- ARC          : annualized return of capital (CAGR-style)
- ASD          : annualized stdev of per-step strategy returns
                 (this is what you called SAD)
- MDD          : max drawdown
- IR*          : ARC / ASD
- IR**         : sign(ARC) * ARC^2 / (ASD * MDD)
- pct_long / pct_short / pct_flat : fraction of traded bars spent in long/short/flat
- trades_long / trades_short / trades_flat / trades_total :
  how many times we ENTERED each regime. Staying long for 200 bars counts as 1.

All "percentage time" stats ignore t=0 because exec_pos[0] is always 0 by design.
"""

from typing import Tuple, Dict, Union, Optional, Sequence
import numpy as np
import pandas as pd

ArrayLike = Union[pd.Series, np.ndarray, list, tuple]

# We need seconds/year for ARC annualization when timestamps are real datetimes.
SECONDS_PER_YEAR = 365.2425 * 24 * 3600.0  # ~31,556,952 seconds


# ---------------------------------------------------------------------------
# Helpers to coerce inputs
# ---------------------------------------------------------------------------

def _as_np(x: ArrayLike) -> np.ndarray:
    """Return x as a 1D float64 numpy array."""
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float)
    else:
        arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected 1D array-like.")
    return arr


def _coerce_series_pair(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> Tuple[np.ndarray, np.ndarray, Optional[pd.Index]]:
    """
    Cast y_true, y_pred to same-length 1D numpy arrays (float).
    Also return index (timestamps etc.) if y_true was a Series.
    """
    idx = y_true.index if isinstance(y_true, pd.Series) else None
    y = _as_np(y_true)
    yp = _as_np(y_pred)
    if y.shape != yp.shape:
        raise ValueError(
            f"Expected same shape, got y_true={y.shape}, y_pred={yp.shape}"
        )
    return y, yp, idx


def _as_datetime_index(
    t: Optional[Union[pd.Series, pd.Index, Sequence]]
) -> Optional[pd.DatetimeIndex]:
    """Best-effort conversion of timestamps to a DatetimeIndex."""
    if t is None:
        return None
    if isinstance(t, (pd.Series, pd.Index)):
        vals = t.values if isinstance(t, pd.Series) else t
        return pd.to_datetime(vals)
    return pd.to_datetime(pd.Index(t))


def periods_per_year_from_timestamps(
    timestamps: Union[pd.Series, pd.Index, Sequence]
) -> float:
    """
    Infer sampling frequency (bars per year) from timestamps.
    Fallback: 252 if we can't infer.
    """
    ts = _as_datetime_index(timestamps)
    if ts is None or len(ts) < 2:
        return 252.0

    # Compute median spacing in seconds
    diffs_ns = (ts[1:].view("i8") - ts[:-1].view("i8")).astype("float64")
    step_seconds = float(np.nanmedian(diffs_ns / 1e9))

    if not np.isfinite(step_seconds) or step_seconds <= 0:
        return 252.0

    ppy = SECONDS_PER_YEAR / step_seconds
    # safety clamp (avoid extreme nonsense from dirty timestamps)
    return float(np.clip(ppy, 12.0, 1_000_000.0))


# ---------------------------------------------------------------------------
# Trading mechanics
# ---------------------------------------------------------------------------

def compute_positions_and_returns(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    threshold: float
) -> Dict[str, np.ndarray]:
    """
    Returns:
        ret      : asset simple returns, shape [T]
        pos_raw  : desired position at t, based on edge[t], shape [T] in {-1,0,+1}
        exec_pos : executed position at t, lagged by 1 bar, shape [T] in {-1,0,+1}
    """
    y, yp, _ = _coerce_series_pair(y_true, y_pred)
    T = len(y)

    # asset simple returns
    ret = np.zeros(T, dtype=float)
    if T >= 2:
        ret[1:] = (y[1:] / y[:-1]) - 1.0

    # edge[t] = expected relative mispricing for next bar
    edge = np.zeros(T, dtype=float)
    nz = (y != 0.0)
    edge[nz] = (yp[nz] / y[nz]) - 1.0

    # raw position (what we'd LIKE to hold next bar)
    pos_raw = np.zeros(T, dtype=float)
    pos_raw[edge >  threshold] = +1.0
    pos_raw[edge < -threshold] = -1.0

    # executed position (what we ACTUALLY hold). We apply yesterday's pos_raw today.
    exec_pos = np.zeros(T, dtype=float)
    if T >= 2:
        exec_pos[1:] = pos_raw[:-1]

    return {
        "ret": ret,
        "pos_raw": pos_raw,
        "exec_pos": exec_pos,
    }


def equity_no_tc(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    starting_capital: float,
    threshold: float
) -> pd.Series:
    """
    Equity curve WITHOUT transaction costs.
    """
    if starting_capital <= 0:
        raise ValueError("starting_capital must be positive.")

    y, yp, idx = _coerce_series_pair(y_true, y_pred)
    tech = compute_positions_and_returns(y, yp, threshold)
    ret, exec_pos = tech["ret"], tech["exec_pos"]

    equity = np.empty_like(y, dtype=float)
    equity[0] = float(starting_capital)
    for t in range(1, len(y)):
        equity[t] = equity[t-1] * (1.0 + exec_pos[t] * ret[t])

    return pd.Series(equity, index=idx)


def equity_with_tc(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    starting_capital: float,
    threshold: float,
    fee_rate: float,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Equity curve WITH transaction costs.

    At each step t>0:
    - Compute position change Δpos = exec_pos[t] - exec_pos[t-1].
    - Fee = fee_rate * |Δpos| * equity_{t-1}.
      (Note flipping long<->short is |Δpos|=2, so double fee.)
    - Apply fee, THEN apply P&L for that bar.
    """
    if starting_capital <= 0:
        raise ValueError("starting_capital must be positive.")
    if fee_rate < 0:
        raise ValueError("fee_rate must be non-negative.")

    y, yp, idx = _coerce_series_pair(y_true, y_pred)
    tech = compute_positions_and_returns(y, yp, threshold)
    ret, exec_pos = tech["ret"], tech["exec_pos"]

    T = len(y)
    equity_tc = np.empty(T, dtype=float)
    fee = np.zeros(T, dtype=float)

    equity_tc[0] = float(starting_capital)

    for t in range(1, T):
        dpos = abs(exec_pos[t] - exec_pos[t-1])        # 0,1,2
        step_fee = fee_rate * dpos * equity_tc[t-1]    # pay once when switching
        equity_tc[t] = (equity_tc[t-1] - step_fee) * (1.0 + exec_pos[t] * ret[t])
        fee[t] = step_fee

    equity_tc_s = pd.Series(equity_tc, index=idx)
    fee_s       = pd.Series(fee, index=idx)
    cum_fee_s   = fee_s.cumsum()

    return equity_tc_s, fee_s, cum_fee_s


# ---------------------------------------------------------------------------
# Risk / performance metrics
# ---------------------------------------------------------------------------

def compute_arc(
    equity: ArrayLike,
    timestamps: Optional[Union[pd.Series, pd.Index, Sequence]] = None,
    periods_per_year: Optional[float] = None,
) -> float:
    """
    ARC = annualized return of capital (CAGR-style).
    """
    eq = _as_np(equity)
    if eq.size < 2 or not np.isfinite(eq[[0, -1]]).all() or eq[0] <= 0 or eq[-1] <= 0:
        return float("nan")

    total_growth = eq[-1] / eq[0]  # final / initial
    years = None

    # If we have real timestamps, measure real elapsed time in years:
    if timestamps is not None:
        ts = _as_datetime_index(timestamps)
        if ts is not None and len(ts) == len(eq):
            years = (ts[-1] - ts[0]).total_seconds() / SECONDS_PER_YEAR

    # Otherwise assume uniform spacing and infer bar frequency:
    if years is None or years <= 0:
        ppy = periods_per_year or 252.0
        years = (len(eq) - 1) / ppy

    if years <= 0:
        return float("nan")

    return float(total_growth ** (1.0 / years) - 1.0)


def compute_asd(
    returns: ArrayLike,
    periods_per_year: Optional[float] = None
) -> float:
    """
    ASD = annualized standard deviation of strategy returns.
    (This is what you referred to as SAD.)
    """
    r = _as_np(returns)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")

    ppy = periods_per_year or 252.0
    return float(np.std(r, ddof=1) * np.sqrt(ppy))


def compute_mdd(equity: ArrayLike) -> float:
    """
    MDD = maximum drawdown fraction, e.g. 0.25 == 25% drawdown.
    """
    eq = _as_np(equity)
    if eq.size < 2:
        return float("nan")

    # Avoid division by <=0 equity
    safe_eq = np.where(eq <= 0, np.nan, eq)
    running_max = np.maximum.accumulate(np.nan_to_num(safe_eq, nan=-np.inf))

    with np.errstate(invalid="ignore", divide="ignore"):
        drawdown = 1.0 - (safe_eq / running_max)

    return float(np.nanmax(drawdown))


def compute_ir_star(arc: float, asd: float) -> float:
    """IR* = ARC / ASD."""
    if (not np.isfinite(arc)) or (not np.isfinite(asd)) or (asd <= 0):
        return float("nan")
    return float(arc / asd)


def compute_ir_starstar(arc: float, asd: float, mdd: float) -> float:
    """IR** = sign(ARC) * ARC^2 / (ASD * MDD)."""
    if (not np.isfinite(arc)
        or not np.isfinite(asd)
        or not np.isfinite(mdd)
        or asd <= 0
        or mdd <= 0):
        return float("nan")
    return float(np.sign(arc) * (arc ** 2) / (asd * mdd))


# ---------------------------------------------------------------------------
# Position usage / trade statistics
# ---------------------------------------------------------------------------

def compute_position_statistics(exec_pos: np.ndarray) -> Dict[str, float]:
    """
    Given exec_pos (shape [T], values in {-1,0,+1} AFTER lag),
    compute:
      - pct_long, pct_short, pct_flat : fraction of bars spent in each regime
      - trades_long, trades_short, trades_flat : how many times we ENTERED that regime
      - trades_total : total transitions counted

    Notes:
    - We ignore t=0 for percentage time (because exec_pos[0] is always 0).
    - We *do* count the first transition at t=1 as a trade.
    """
    pos = np.asarray(exec_pos, dtype=float)
    T = len(pos)
    if T <= 1:
        return {
            "pct_long": np.nan,
            "pct_short": np.nan,
            "pct_flat": np.nan,
            "trades_long": 0,
            "trades_short": 0,
            "trades_flat": 0,
            "trades_total": 0,
        }

    # --- percentage time in each regime (ignoring t=0 warmup) ---
    active = pos[1:]  # actual traded bars
    denom = active.size if active.size > 0 else 1
    pct_long  = float(np.sum(active ==  1.0) / denom)
    pct_short = float(np.sum(active == -1.0) / denom)
    pct_flat  = float(np.sum(active ==  0.0) / denom)

    # --- trade counts (count transitions in exec_pos) ---
    trades_long = 0
    trades_short = 0
    trades_flat = 0

    for t in range(1, T):
        if pos[t] != pos[t-1]:
            if pos[t] ==  1.0:
                trades_long += 1
            elif pos[t] == -1.0:
                trades_short += 1
            elif pos[t] ==  0.0:
                trades_flat += 1

    trades_total = trades_long + trades_short + trades_flat

    return {
        "pct_long": pct_long,
        "pct_short": pct_short,
        "pct_flat": pct_flat,
        "trades_long": trades_long,
        "trades_short": trades_short,
        "trades_flat": trades_flat,
        "trades_total": trades_total,
    }


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

def evaluate_forecast(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    timestamps: Optional[Union[pd.Series, pd.Index, Sequence]] = None,
    starting_capital: float = 100_000.0,
    threshold: float = 0.01,
    fee_rate: float = 0.001,
) -> Dict[str, Union[float, pd.Series]]:

    # --- equity curves ---
    eq_no = equity_no_tc(y_true, y_pred, starting_capital, threshold)
    eq_tc, fee_s, cum_fee_s = equity_with_tc(
        y_true, y_pred, starting_capital, threshold, fee_rate
    )

    # --- final scalars ---
    final_equity_no_tc = float(eq_no.iloc[-1]) if len(eq_no) else float("nan")
    final_equity_tc    = float(eq_tc.iloc[-1]) if len(eq_tc) else float("nan")
    total_fees_paid    = float(cum_fee_s.iloc[-1]) if len(cum_fee_s) else float("nan")

    # --- strategy returns from fee-adjusted equity ---
    rets = np.zeros(len(eq_no), dtype=float)
    if len(eq_no) >= 2:
        rets[1:] = (eq_no.values[1:] / eq_no.values[:-1]) - 1.0

    # --- frequency for annualization ---
    ppy = periods_per_year_from_timestamps(timestamps) if timestamps is not None else None

    # --- performance stats ---
    arc = compute_arc(eq_no.values, timestamps=timestamps, periods_per_year=ppy)
    asd = compute_asd(rets, periods_per_year=ppy or 252.0)
    mdd = compute_mdd(eq_no.values)
    ir1 = compute_ir_star(arc, asd)
    ir2 = compute_ir_starstar(arc, asd, mdd)

    # --- position usage / trade stats ---
    tech = compute_positions_and_returns(y_true, y_pred, threshold)
    exec_pos = tech["exec_pos"]
    pos_stats = compute_position_statistics(exec_pos)

    return {
        # curves / series
        "equity_no_tc": eq_no,
        "equity_tc": eq_tc,
        "fee": fee_s,
        "cum_fee": cum_fee_s,

        # final scalar summaries
        "final_equity_no_tc": final_equity_no_tc,
        "final_equity_tc": final_equity_tc,
        "total_fees_paid": total_fees_paid,

        # risk stats
        "arc": arc,
        "asd": asd,
        "mdd": mdd,
        "ir_star": ir1,
        "ir_starstar": ir2,

        # position usage / trade stats
        "pct_long": pos_stats["pct_long"],
        "pct_short": pos_stats["pct_short"],
        "pct_flat": pos_stats["pct_flat"],
        "trades_long": pos_stats["trades_long"],
        "trades_short": pos_stats["trades_short"],
        "trades_flat": pos_stats["trades_flat"],
        "trades_total": pos_stats["trades_total"],
    }


__all__ = [
    "compute_positions_and_returns",
    "equity_no_tc",
    "equity_with_tc",
    "compute_arc",
    "compute_asd",
    "compute_mdd",
    "compute_ir_star",
    "compute_ir_starstar",
    "compute_position_statistics",
    "evaluate_forecast",
]