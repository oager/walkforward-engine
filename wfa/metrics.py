"""Stitched OOS metrics and return-degradation calculation.

All metrics computed on stitched out-of-sample (OOS) trades only.
IS metrics are stored per-fold for comparison but never used as the headline result.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wfa.objectives import (
    _sharpe,
    _sortino,
    _calmar,
    _total_return,
    _win_rate,
    _profit_factor,
)

logger = logging.getLogger(__name__)


@dataclass
class OOSMetrics:
    """Headline metrics for a stitched OOS trade stream."""

    n_trades: int
    total_return: float       # (final_equity - initial) / initial
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float   # 0-1, higher = worse
    win_rate: float
    profit_factor: float

    def as_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
        }


def build_equity_curve(trades: pd.DataFrame, initial_capital: float = 10_000.0) -> list[float]:
    """Build a compounding equity curve from stitched OOS trades.

    Trades must have a 'pnl' column. Equity starts at initial_capital and
    each trade's pnl is added sequentially (preserves OOS order).
    """
    if trades.empty:
        return [initial_capital]
    eq: list[float] = [initial_capital]
    for pnl in trades["pnl"]:
        eq.append(eq[-1] + float(pnl))
    return eq


def max_drawdown(equity: list[float]) -> float:
    """Return max drawdown as a fraction 0-1 (0 = no drawdown)."""
    arr = np.array(equity, dtype=float)
    if len(arr) < 2:
        return 0.0
    peak = np.maximum.accumulate(arr)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.where(peak > 0, (peak - arr) / peak, 0.0)
    return float(dd.max())


def compute_oos_metrics(
    stitched_trades: pd.DataFrame,
    initial_capital: float = 10_000.0,
) -> OOSMetrics:
    """Compute all headline metrics on stitched OOS trades."""
    equity = build_equity_curve(stitched_trades, initial_capital)

    return OOSMetrics(
        n_trades=len(stitched_trades),
        total_return=_total_return(stitched_trades, equity),
        sharpe=_sharpe(stitched_trades, equity),
        sortino=_sortino(stitched_trades, equity),
        calmar=_calmar(stitched_trades, equity),
        max_drawdown_pct=max_drawdown(equity),
        win_rate=_win_rate(stitched_trades, equity),
        profit_factor=_profit_factor(stitched_trades, equity),
    )


def return_degradation(
    oos_return: float,
    mean_is_return: float,
) -> float | None:
    """Fraction of IS return that was curve-fit.

    degradation = 1 - (OOS_return / IS_return)

    Returns None if IS return is <= 0 or not finite — the ratio is undefined for a
    non-positive IS baseline (a negative IS flips the sign: OOS-beats-IS would read
    as "lost too much of IS" and vice versa).
    0.75 = 75% of IS gains did not transfer OOS.
    Negative = OOS outperformed IS (check for noise/small sample).
    """
    if not math.isfinite(mean_is_return):
        return None
    if mean_is_return < 0.0:
        # Distinct from the zero/non-finite skip below: there WAS an IS baseline,
        # it was just net-negative, so the ratio is undefined (sign flips) and the
        # overfit filter is skipped — surface that explicitly so "couldn't compute"
        # doesn't read as "no overfit" in the logs.
        logger.warning(
            "return_degradation: skipping overfit filter — mean_is_return is "
            "negative (%.4f), degradation ratio is undefined for a net-losing "
            "IS baseline", mean_is_return)
        return None
    if mean_is_return == 0.0:
        return None
    if not math.isfinite(oos_return):
        return None
    return 1.0 - (oos_return / mean_is_return)
