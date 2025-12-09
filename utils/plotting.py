# utils/plotting.py
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def _get_time_index(df: pd.DataFrame, timestamp_col: Optional[str]) -> pd.Series:
    if timestamp_col is not None and timestamp_col in df.columns:
        return pd.to_datetime(df[timestamp_col])
    # fallback: use index
    if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        return pd.to_datetime(df.index)
    # last resort: make a simple range index
    return pd.Series(np.arange(len(df)), index=df.index)


def _compute_bh_equity(y_true: pd.Series, starting_capital: float) -> np.ndarray:
    y = y_true.to_numpy(dtype=float)
    if len(y) == 0 or y[0] == 0:
        raise ValueError("Cannot compute buy-and-hold equity: empty series or y[0]==0.")
    return starting_capital * (y / y[0])


# 1) Price: Actual vs Predicted (1-step)
def plot_price_actual_vs_pred(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    timestamp_col: Optional[str] = "timestamp",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    if y_true_col not in df or y_pred_col not in df:
        raise KeyError(f"Expected columns '{y_true_col}' and '{y_pred_col}' in DataFrame.")

    t = _get_time_index(df, timestamp_col)
    y_true = df[y_true_col].astype(float)
    y_pred = df[y_pred_col].astype(float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4.5))
    else:
        fig = ax.figure

    ax.plot(t, y_true, label="Actual", linewidth=1.2)
    ax.plot(t, y_pred, label="Predicted", alpha=0.85)
    ax.set_title(title or "Actual vs Predicted (1-step)")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax


# Generic equity plotter (strategy vs buy-and-hold), with optional log scale
def plot_equity_vs_bh(
    df: pd.DataFrame,
    equity_col: str = "equity",                 # use "equity_tc" for with-fees
    y_true_col: str = "y_true",
    timestamp_col: Optional[str] = "timestamp",
    starting_capital: Optional[float] = None,   # if None, uses first equity value (if available)
    label_strategy: Optional[str] = None,
    logy: bool = False,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    if y_true_col not in df:
        raise KeyError(f"Expected column '{y_true_col}' for buy-and-hold computation.")
    if equity_col not in df:
        raise KeyError(f"Expected column '{equity_col}' for strategy equity.")

    t = _get_time_index(df, timestamp_col)
    eq_series = df[equity_col].astype(float)
    sc = float(eq_series.iloc[0]) if starting_capital is None else float(starting_capital)
    bh_equity = _compute_bh_equity(df[y_true_col].astype(float), sc)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4.5))
    else:
        fig = ax.figure

    ax.plot(t, eq_series, label=label_strategy or f"Strategy ({equity_col})")
    ax.plot(t, bh_equity, label="Buy & Hold")
    ax.set_ylabel("Equity")
    ax.set_xlabel("Time")
    if logy:
        ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_title(title or ("Equity vs Buy & Hold" + (" (log scale)" if logy else "")))
    fig.tight_layout()
    return fig, ax


# 2) Equity curve vs Buy-and-Hold (no fees)
def plot_equity_no_tc_vs_bh(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    equity_col: str = "equity",
    timestamp_col: Optional[str] = "timestamp",
    starting_capital: Optional[float] = None,
    logy: bool = False,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    return plot_equity_vs_bh(
        df=df,
        equity_col=equiry_col if (equiry_col:=equity_col) else "equity",
        y_true_col=y_true_col,
        timestamp_col=timestamp_col,
        starting_capital=starting_capital,
        label_strategy="Strategy Equity",
        logy=logy,
        title=title or ("Equity vs Buy & Hold" + (" (log)" if logy else "")),
        ax=ax,
    )


# 3) Equity curve (with fees) vs Buy-and-Hold
def plot_equity_tc_vs_bh(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    equity_tc_col: str = "equity_tc",
    timestamp_col: Optional[str] = "timestamp",
    starting_capital: Optional[float] = None,
    logy: bool = False,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    return plot_equity_vs_bh(
        df=df,
        equity_col=equiry_col if (equiry_col:=equity_tc_col) else "equity_tc",
        y_true_col=y_true_col,
        timestamp_col=timestamp_col,
        starting_capital=starting_capital,
        label_strategy="Strategy Equity (with fees)",
        logy=logy,
        title=title or ("Equity (with fees) vs Buy & Hold" + (" (log)" if logy else "")),
        ax=ax,
    )


# 4) Convenience wrappers for log-scaled versions
def plot_equity_no_tc_vs_bh_log(df: pd.DataFrame, **kwargs):
    kwargs = dict(kwargs)
    kwargs["logy"] = True
    return plot_equity_no_tc_vs_bh(df=df, **kwargs)

def plot_equity_tc_vs_bh_log(df: pd.DataFrame, **kwargs):
    kwargs = dict(kwargs)
    kwargs["logy"] = True
    return plot_equity_tc_vs_bh(df=df, **kwargs)


# 5) Two-panel combo (price on top, equity vs B&H below)
def plot_price_and_equity_panel(
    df: pd.DataFrame,
    equity_col: str = "equity",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    timestamp_col: Optional[str] = "timestamp",
    title_price: Optional[str] = None,
    title_equity: Optional[str] = None,
    logy_equity: bool = False,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    t = _get_time_index(df, timestamp_col)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: price
    plot_price_actual_vs_pred(
        df=df,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
        timestamp_col=timestamp_col,
        title=title_price,
        ax=axes[0],
    )

    # Bottom: equity vs B&H
    plot_equity_vs_bh(
        df=df,
        equity_col=equity_col,
        y_true_col=y_true_col,
        timestamp_col=timestamp_col,
        label_strategy=f"Strategy ({equity_col})",
        logy=logy_equity,
        title=title_equity,
        ax=axes[1],
    )

    axes[1].set_xlabel("Time")
    fig.tight_layout()
    return fig, (axes[0], axes[1])



ASDFrameLike = Union[pd.DataFrame, str, Path]

def _coerce_asd_monthly(
    obj: ASDFrameLike,
    *,
    month_col: str = "month_end",
    asd_col: str = "ASD",
    start: str = "2013-01-01",
) -> pd.DataFrame:
    """
    Coerce an ASD monthly dataset (1-minute bars) into a tidy DataFrame:
    - If `obj` is a DataFrame: copy and normalize.
    - If `obj` is a path: read CSV.
    - Ensures DateTimeIndex on `month_col` and presence of `asd_col`.
    - Returns df.loc[start:, [asd_col]] sorted by index.
    """
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        if not p.exists():
            raise FileNotFoundError(f"ASD file not found: {p}")
        df = pd.read_csv(p)
    elif isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        raise TypeError("obj must be a DataFrame or a path to a CSV file.")

    if month_col in df.columns:
        df[month_col] = pd.to_datetime(df[month_col])
        df = df.set_index(month_col)

    if asd_col not in df.columns:
        raise ValueError(f"Expected column '{asd_col}' in ASD data.")

    df = df[[asd_col]].dropna().sort_index()
    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError("Expected a datetime-like index or a 'month_end' column.")

    # Normalize PeriodIndex -> Timestamp
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp(how="end")

    return df.loc[pd.to_datetime(start):]


def plot_monthly_asd_1min(
    asd_source: ASDFrameLike,
    *,
    month_col: str = "month_end",
    asd_col: str = "ASD",
    start: str = "2013-01-01",
    title: str = "Monthly Annualised Std Dev (ASD) — 1-minute data",
    marker: str = "o",
    linewidth: float = 1.2,
    label: str = "ASD (1-minute)",
    show_tail: int = 0,          # e.g. 6 to print last 6 rows
    x_start: Optional[pd.Timestamp] = None,
    x_end: Optional[pd.Timestamp] = None,
    y_pad: float = 0.10,         # 10% headroom when zooming
    rotate_xticks: int = 30,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Plot month-to-month ASD (1-minute). Accepts a DataFrame or a CSV path.

    Returns
    -------
    fig, ax, df_asd : the Matplotlib objects and the normalized ASD DataFrame used.
    """
    df_asd = _coerce_asd_monthly(asd_source, month_col=month_col, asd_col=asd_col, start=start)

    if show_tail and show_tail > 0:
        # compact tail preview (like your print)
        print(df_asd.tail(show_tail))

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    ax.plot(df_asd.index, df_asd[asd_col], marker=marker, linewidth=linewidth, label=label)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(asd_col)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if rotate_xticks:
        plt.setp(ax.get_xticklabels(), rotation=rotate_xticks)

    # Optional zoom
    if x_start is not None or x_end is not None:
        xmin = pd.to_datetime(x_start) if x_start is not None else df_asd.index.min()
        xmax = pd.to_datetime(x_end)   if x_end is not None else df_asd.index.max()
        ax.set_xlim(xmin, xmax)
        # y-limits based on the visible slice
        visible = df_asd.loc[(df_asd.index >= xmin) & (df_asd.index <= xmax), asd_col]
        if not visible.empty:
            ymax = float(visible.max())
            ymin = float(visible.min())
            pad = max(1e-12, y_pad)   # avoid zero
            ax.set_ylim(ymin * (1 - pad), ymax * (1 + pad))

    fig.tight_layout()
    return fig, ax, df_asd

