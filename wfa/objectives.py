"""Objective function registry for WFA parameter optimisation.

Each objective takes a trades DataFrame and equity curve, returning a scalar.
Higher is better. Invalid/empty inputs return -inf so they are never selected.

Min-trades gate: if trade count < min_trades, return -inf regardless of objective.
This prevents the optimiser picking params that fire 0 trades and 'win' by NaN->inf Sharpe.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from statistics import NormalDist

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ObjectiveFn = Callable[[pd.DataFrame, list[float]], float]

# One-shot flag: warn loudly (but only once per process — objectives run thousands
# of times per WFA search) when annualisation silently degrades to factor 1.0.
_ANN_FALLBACK_WARNED = False


def _warn_ann_fallback(reason: str) -> None:
    global _ANN_FALLBACK_WARNED
    if not _ANN_FALLBACK_WARNED:
        logger.warning(
            "annualisation degraded to factor 1.0 (%s) — Sharpe/Sortino/Calmar are "
            "per-trade UNANNUALISED and NOT comparable to annualised baselines", reason)
        _ANN_FALLBACK_WARNED = True


def _safe(val: float) -> float:
    return val if math.isfinite(val) else -math.inf


def _ann_factor(trades: pd.DataFrame) -> float:
    """sqrt(trades_per_year) — per-trade annualisation factor derived from actual date span."""
    n = len(trades)
    if n < 2:
        return 1.0
    if "exit_time" not in trades.columns:
        _warn_ann_fallback("trades have no 'exit_time' column")
        return 1.0
    try:
        span_days = float(
            (pd.to_datetime(trades["exit_time"]).max() - pd.to_datetime(trades["exit_time"]).min()).days
        )
    except Exception as exc:
        _warn_ann_fallback(f"'exit_time' unparseable: {exc}")
        return 1.0
    if span_days <= 0:
        return 1.0
    return math.sqrt(n / (span_days / 365.25))


def _span_years(trades: pd.DataFrame) -> float:
    """Actual date span in years (for Calmar annualisation). Falls back to n/6 if no timestamps."""
    n = len(trades)
    if n < 2:
        return max(n / 6.0, 1e-9)
    if "exit_time" not in trades.columns:
        _warn_ann_fallback("trades have no 'exit_time' column")
        return max(n / 6.0, 1e-9)
    try:
        span_days = float(
            (pd.to_datetime(trades["exit_time"]).max() - pd.to_datetime(trades["exit_time"]).min()).days
        )
    except Exception as exc:
        _warn_ann_fallback(f"'exit_time' unparseable: {exc}")
        return max(n / 6.0, 1e-9)
    return max(span_days / 365.25, 1e-9)


def _sharpe(trades: pd.DataFrame, equity: list[float]) -> float:
    if len(equity) < 2:
        return -math.inf
    if min(equity[:-1]) <= 0:
        return -math.inf  # equity touched/crossed zero — %-returns past that point are sign-inverted
    returns = np.diff(equity) / np.array(equity[:-1])
    mean_r = float(returns.mean())
    std_r = float(returns.std(ddof=1))
    if std_r == 0:
        return -math.inf
    return _safe(mean_r / std_r * _ann_factor(trades))


def _sortino(trades: pd.DataFrame, equity: list[float]) -> float:
    if len(equity) < 2:
        return -math.inf
    if min(equity[:-1]) <= 0:
        return -math.inf  # equity touched/crossed zero — %-returns past that point are sign-inverted
    returns = np.diff(equity) / np.array(equity[:-1])
    mean_r = float(returns.mean())
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 100.0 if mean_r > 0 else -math.inf
    downside_std = float(np.sqrt((downside ** 2).mean()))
    if downside_std == 0:
        return -math.inf
    return _safe(mean_r / downside_std * _ann_factor(trades))


def _calmar(trades: pd.DataFrame, equity: list[float]) -> float:
    if len(equity) < 2:
        return -math.inf
    arr = np.array(equity, dtype=float)
    initial = arr[0]
    if initial <= 0:
        return -math.inf
    total_return = (arr[-1] - initial) / initial
    years = _span_years(trades)
    ann_return = (1 + total_return) ** (1.0 / years) - 1
    peak = np.maximum.accumulate(arr)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.where(peak > 0, (peak - arr) / peak, 0.0)
    max_dd = float(dd.max())
    if max_dd == 0:
        return 100.0 if ann_return > 0 else -math.inf
    return _safe(ann_return / max_dd)


def _total_return(trades: pd.DataFrame, equity: list[float]) -> float:
    if len(equity) < 2:
        return -math.inf
    initial = equity[0]
    if initial <= 0:
        return -math.inf
    return _safe((equity[-1] - initial) / initial)


def _win_rate(trades: pd.DataFrame, equity: list[float]) -> float:
    if trades.empty:
        return -math.inf
    wins = int((trades["pnl"] > 0).sum())
    return _safe(wins / len(trades))


def _profit_factor(trades: pd.DataFrame, equity: list[float]) -> float:
    if trades.empty:
        return -math.inf
    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gross_loss = abs(float(trades.loc[trades["pnl"] < 0, "pnl"].sum()))
    if gross_loss == 0:
        return 100.0 if gross_profit > 0 else -math.inf
    return _safe(gross_profit / gross_loss)


def _psr(trades: pd.DataFrame, equity: list[float]) -> float:
    """Probabilistic Sharpe Ratio — P(true Sharpe > 0 | observed SR, n, skew, kurt)."""
    if len(equity) < 4:
        return -math.inf
    if min(equity[:-1]) <= 0:
        return -math.inf  # equity touched/crossed zero — %-returns past that point are sign-inverted
    returns = np.diff(equity) / np.array(equity[:-1])
    n = len(returns)
    sr = _sharpe(trades, equity)
    if not math.isfinite(sr):
        return -math.inf
    ann = _ann_factor(trades)
    sr_obs = sr / ann if ann > 0 else 0.0
    s = pd.Series(returns)
    skew = float(s.skew()) if n >= 4 else 0.0
    kurt = float(s.kurtosis()) + 3.0 if n >= 4 else 3.0  # pandas gives EXCESS kurtosis; PSR wants non-excess
    var_term = (1 - skew * sr_obs + (kurt - 1) / 4 * sr_obs ** 2) / (n - 1)
    if var_term <= 0:
        return -math.inf
    denom = math.sqrt(var_term)
    psr = NormalDist().cdf(sr_obs / denom)
    return _safe(psr)


# ── registry ───────────────────────────────────────────────────────────────────

OBJECTIVES: dict[str, dict] = {
    "sortino": {
        "fn": _sortino,
        "label": "Sortino Ratio",
        "help": (
            "Default. Penalises downside volatility only — ignores upside vol. "
            "Best for strategies where you accept some up-vol but hate drawdowns. "
            "Higher = better risk-adjusted return on the downside dimension."
        ),
    },
    "calmar": {
        "fn": _calmar,
        "label": "Calmar Ratio",
        "help": (
            "Annualised return ÷ max drawdown. Directly rewards small max-DD. "
            "Best for 'will this survive real money?' — the only metric that "
            "explicitly prices drawdown in the denominator."
        ),
    },
    "sharpe": {
        "fn": _sharpe,
        "label": "Sharpe Ratio",
        "help": (
            "Classical risk-adjusted return. Punishes all volatility equally — "
            "penalises upside AND downside. Favours smooth equity curves. "
            "Can pick low-return strategies that happen to be smooth."
        ),
    },
    "total_return": {
        "fn": _total_return,
        "label": "Total Return",
        "help": (
            "Simple (final equity − initial) / initial. "
            "WARNING: ignores risk entirely — will pick the most aggressive params. "
            "Useful as a sanity check but dangerous as an optimiser target."
        ),
    },
    "win_rate": {
        "fn": _win_rate,
        "label": "Win Rate",
        "help": (
            "Fraction of winning trades. Rewards many small wins. "
            "WARNING: 90% WR with one large loss still loses money. "
            "Useful as a secondary diagnostic, not a primary optimiser target."
        ),
    },
    "profit_factor": {
        "fn": _profit_factor,
        "label": "Profit Factor",
        "help": (
            "Gross profit ÷ gross loss. PF > 1.5 is generally considered robust. "
            "Less timeframe-dependent than Sharpe. Tolerant of non-normal returns."
        ),
    },
    "psr": {
        "fn": _psr,
        "label": "Probabilistic Sharpe Ratio",
        "help": (
            "Sharpe adjusted for sample size, return skew, and kurtosis "
            "(Bailey & Lopez de Prado 2012). Returns P(true Sharpe > 0). "
            "Most honest metric — slower and harder to game than raw Sharpe. "
            "Best when sample is small (< 100 trades)."
        ),
    },
}

OBJECTIVE_NAMES = list(OBJECTIVES.keys())
DEFAULT_OBJECTIVE = "sortino"


def compute_objective(
    name: str,
    trades: pd.DataFrame,
    equity: list[float],
    min_trades: int = 10,
) -> float:
    """Compute named objective; return -inf if fewer than min_trades."""
    if name not in OBJECTIVES:
        raise ValueError(f"Unknown objective '{name}'. Choose from: {OBJECTIVE_NAMES}")
    if len(trades) < min_trades:
        return -math.inf
    return OBJECTIVES[name]["fn"](trades, equity)
