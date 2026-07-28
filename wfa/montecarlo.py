"""Monte Carlo path-risk analysis for stitched OOS trade streams.

Core math ported from mybot/scripts/monte_carlo.py.
Input is always stitched OOS trades from a WFA run (or raw trades for legacy mode).

Per-trade-close equity curves — not per-bar. Sharpe/Sortino/Calmar here are NOT
comparable to per-bar baseline numbers. Portal labels every stat with its granularity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from wfa.objectives import _ann_factor


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class MCPercentiles:
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    mean: float
    std: float

    def as_dict(self) -> dict:
        return {
            "p5": self.p5, "p25": self.p25, "p50": self.p50,
            "p75": self.p75, "p95": self.p95,
            "mean": self.mean, "std": self.std,
        }


@dataclass
class MCResult:
    """Summary of a Monte Carlo simulation run."""

    source: Literal["stitched_oos", "raw_trades"]
    method: str                          # "reshuffle" | "bootstrap"
    n_sims: int
    n_trades: int
    initial_capital: float
    seed: int
    compounding: bool

    probability_of_ruin: float           # fraction of sims that hit ruin_threshold
    ruin_threshold: float                # as fraction (e.g. 0.20)

    total_return_pct: MCPercentiles      # distribution of final return %
    max_drawdown_pct: MCPercentiles      # distribution of max DD %
    losing_streak: MCPercentiles         # distribution of longest losing streak
    sharpe: MCPercentiles                # per-trade Sharpe
    sortino: MCPercentiles               # per-trade Sortino

    # Raw arrays for portal charting
    equity_curves: np.ndarray = field(repr=False)  # (n_sims, n_trades+1)
    all_return_pct: np.ndarray = field(repr=False)  # (n_sims,)
    all_max_dd_pct: np.ndarray = field(repr=False)  # (n_sims,)
    all_losing_streak: np.ndarray = field(repr=False)  # (n_sims,)
    all_sharpe: np.ndarray = field(repr=False)       # (n_sims,)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "method": self.method,
            "n_sims": self.n_sims,
            "n_trades": self.n_trades,
            "initial_capital": self.initial_capital,
            "seed": self.seed,
            "compounding": self.compounding,
            "ruin_threshold": self.ruin_threshold,
            "probability_of_ruin": self.probability_of_ruin,
            "total_return_pct": self.total_return_pct.as_dict(),
            "max_drawdown_pct": self.max_drawdown_pct.as_dict(),
            "losing_streak": self.losing_streak.as_dict(),
            "sharpe": self.sharpe.as_dict(),
            "sortino": self.sortino.as_dict(),
        }


# ── core simulation math ──────────────────────────────────────────────────────

def _resample_paths(
    returns: np.ndarray,
    n_sims: int,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (n_sims, n_trades) resampled return matrix.

    reshuffle: permutation — tests path/sequence risk, preserves exact edge.
    bootstrap: resample with replacement — also captures sampling uncertainty.
    """
    n = len(returns)
    if method == "reshuffle":
        tiled = np.tile(returns, (n_sims, 1))
        return rng.permuted(tiled, axis=1)
    if method == "bootstrap":
        idx = rng.integers(0, n, size=(n_sims, n))
        return returns[idx]
    raise ValueError(f"Unknown method {method!r}. Choose 'reshuffle' or 'bootstrap'.")


def _build_equity_curves(
    paths: np.ndarray,
    initial_capital: float,
    compounding: bool,
) -> np.ndarray:
    """Return (n_sims, n_trades+1) equity curves starting at initial_capital."""
    n_sims = paths.shape[0]
    start = np.full((n_sims, 1), initial_capital)
    if compounding:
        tail = initial_capital * np.cumprod(1.0 + paths, axis=1)
    else:
        tail = initial_capital * (1.0 + np.cumsum(paths, axis=1))
    return np.concatenate([start, tail], axis=1)


def _max_dd_rows(equity: np.ndarray) -> np.ndarray:
    """Max drawdown % per row (vectorized)."""
    peak = np.maximum.accumulate(equity, axis=1)
    safe_peak = np.where(peak > 0, peak, 1.0)
    dd = (peak - equity) / safe_peak * 100.0
    return np.max(dd, axis=1)


def _max_losing_streaks(paths: np.ndarray) -> np.ndarray:
    """Max consecutive losing streak per row (vectorized island-detection)."""
    losses = (paths < 0).astype(np.int32)
    # Pad with a zero column on each side so streak islands are always bounded
    padded = np.pad(losses, ((0, 0), (1, 1)), constant_values=0)
    # Rising edges (+1): streak starts; falling edges (-1): streak ends
    diff = np.diff(padded, axis=1)
    n_sims = paths.shape[0]
    streaks = np.empty(n_sims, dtype=np.int32)
    for i in range(n_sims):
        starts = np.where(diff[i] == 1)[0]
        ends = np.where(diff[i] == -1)[0]
        if len(starts) == 0:
            streaks[i] = 0
        else:
            streaks[i] = int((ends - starts).max())
    return streaks


def _pct_stats(arr: np.ndarray) -> MCPercentiles:
    ps = np.percentile(arr, [5, 25, 50, 75, 95])
    return MCPercentiles(
        p5=round(float(ps[0]), 4),
        p25=round(float(ps[1]), 4),
        p50=round(float(ps[2]), 4),
        p75=round(float(ps[3]), 4),
        p95=round(float(ps[4]), 4),
        mean=round(float(arr.mean()), 4),
        std=round(float(arr.std()), 4),
    )


# ── public API ────────────────────────────────────────────────────────────────

def run_mc(
    trades: pd.DataFrame,
    initial_capital: float = 10_000.0,
    n_sims: int = 10_000,
    method: str = "reshuffle",
    ruin_threshold: float = 0.20,
    compounding: bool = True,
    seed: int = 42,
    source: Literal["stitched_oos", "raw_trades"] = "stitched_oos",
) -> MCResult:
    """Run Monte Carlo on *trades* (must have a 'pnl' column).

    Args:
        trades:          Stitched OOS trades from a WFA run (or raw trades).
        initial_capital: Starting equity. Fractional returns = pnl / initial_capital.
        n_sims:          Number of resampled paths.
        method:          "reshuffle" (sequence risk only) or "bootstrap" (+ sample risk).
        ruin_threshold:  Max-DD fraction that counts as ruin (e.g. 0.20 = 20%).
        compounding:     True = compound equity model; False = additive.
        seed:            RNG seed for reproducibility.
        source:          Tag for the output: "stitched_oos" or "raw_trades".
    """
    if trades.empty:
        raise ValueError("trades DataFrame is empty — cannot run Monte Carlo.")
    if "pnl" not in trades.columns:
        raise ValueError("trades DataFrame must have a 'pnl' column.")

    pnls = trades["pnl"].astype(float).values
    returns = pnls / initial_capital

    rng = np.random.default_rng(seed)
    paths = _resample_paths(returns, n_sims, method, rng)
    equity = _build_equity_curves(paths, initial_capital, compounding)

    final_equity = equity[:, -1]
    total_return_pct = (final_equity - initial_capital) / initial_capital * 100.0
    max_dd_pct = _max_dd_rows(equity)
    hit_ruin = max_dd_pct >= ruin_threshold * 100.0
    losing_streaks = _max_losing_streaks(paths)

    mean_r = np.mean(paths, axis=1)
    std_r = np.std(paths, axis=1, ddof=1)  # sample std — consistent with objectives._sharpe
    downside_sq = np.minimum(paths, 0.0) ** 2
    downside_dev = np.sqrt(np.mean(downside_sq, axis=1))

    ann = _ann_factor(trades)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std_r > 0, mean_r / std_r * ann, 0.0)
        sortino = np.where(downside_dev > 0, mean_r / downside_dev * ann, 0.0)

    return MCResult(
        source=source,
        method=method,
        n_sims=n_sims,
        n_trades=len(trades),
        initial_capital=initial_capital,
        seed=seed,
        compounding=compounding,
        ruin_threshold=ruin_threshold,
        probability_of_ruin=round(float(hit_ruin.mean()), 4),
        total_return_pct=_pct_stats(total_return_pct),
        max_drawdown_pct=_pct_stats(max_dd_pct),
        losing_streak=_pct_stats(losing_streaks.astype(float)),
        sharpe=_pct_stats(sharpe),
        sortino=_pct_stats(sortino),
        equity_curves=equity,
        all_return_pct=total_return_pct,
        all_max_dd_pct=max_dd_pct,
        all_losing_streak=losing_streaks,
        all_sharpe=sharpe,
    )
