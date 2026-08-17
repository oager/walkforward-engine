"""Tests for wfa/metrics.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from wfa.metrics import (
    OOSMetrics,
    build_equity_curve,
    compute_oos_metrics,
    max_drawdown,
    return_degradation,
)


def make_trades(pnls: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"pnl": pnls})


# ── build_equity_curve ────────────────────────────────────────────────────────

def test_equity_curve_empty_trades() -> None:
    eq = build_equity_curve(make_trades([]), initial_capital=10_000.0)
    assert eq == [10_000.0]


def test_equity_curve_known_values() -> None:
    eq = build_equity_curve(make_trades([1000.0, -500.0, 200.0]), initial_capital=10_000.0)
    assert eq == [10_000.0, 11_000.0, 10_500.0, 10_700.0]


def test_equity_curve_length() -> None:
    trades = make_trades([100.0] * 10)
    eq = build_equity_curve(trades)
    assert len(eq) == 11  # initial + 10 trades


# ── max_drawdown ──────────────────────────────────────────────────────────────

def test_max_drawdown_no_drawdown() -> None:
    eq = [10_000.0, 10_100.0, 10_200.0, 10_300.0]
    assert max_drawdown(eq) == 0.0


def test_max_drawdown_known_value() -> None:
    # Peak = 10_200, trough = 9_800 → dd = 400/10_200 ≈ 0.0392
    eq = [10_000.0, 10_200.0, 9_800.0, 10_500.0]
    dd = max_drawdown(eq)
    assert abs(dd - 400 / 10_200) < 1e-9


def test_max_drawdown_single_bar() -> None:
    assert max_drawdown([10_000.0]) == 0.0


def test_max_drawdown_all_losses() -> None:
    eq = [10_000.0, 9_000.0, 8_000.0, 7_000.0]
    dd = max_drawdown(eq)
    assert abs(dd - 3_000 / 10_000) < 1e-9


# ── compute_oos_metrics ───────────────────────────────────────────────────────

def test_compute_oos_metrics_returns_oos_metrics() -> None:
    trades = make_trades([200.0, -50.0] * 15)
    result = compute_oos_metrics(trades)
    assert isinstance(result, OOSMetrics)


def test_compute_oos_metrics_n_trades() -> None:
    trades = make_trades([100.0] * 20)
    result = compute_oos_metrics(trades)
    assert result.n_trades == 20


def test_compute_oos_metrics_total_return_known() -> None:
    # 10 trades of +1000 each, starting from 10_000 → final = 20_000 → return = 1.0
    trades = make_trades([1000.0] * 10)
    result = compute_oos_metrics(trades, initial_capital=10_000.0)
    assert abs(result.total_return - 1.0) < 1e-9


def test_compute_oos_metrics_max_dd_upward_curve() -> None:
    trades = make_trades([100.0] * 20)
    result = compute_oos_metrics(trades)
    assert result.max_drawdown_pct == 0.0


def test_compute_oos_metrics_as_dict_has_all_keys() -> None:
    trades = make_trades([100.0, -50.0] * 10)
    d = compute_oos_metrics(trades).as_dict()
    expected_keys = {
        "n_trades", "total_return", "sharpe", "sortino", "calmar",
        "max_drawdown_pct", "win_rate", "profit_factor",
    }
    assert set(d.keys()) == expected_keys


# ── return_degradation ────────────────────────────────────────────────────────

def test_degradation_zero_when_equal() -> None:
    result = return_degradation(oos_return=0.5, mean_is_return=0.5)
    assert result == 0.0


def test_degradation_full_when_oos_zero() -> None:
    result = return_degradation(oos_return=0.0, mean_is_return=0.5)
    assert result == 1.0


def test_degradation_known_value() -> None:
    # OOS got 25% of IS return → degradation = 1 - 0.25/1.0 = 0.75
    result = return_degradation(oos_return=0.25, mean_is_return=1.0)
    assert abs(result - 0.75) < 1e-9


def test_degradation_negative_when_oos_beats_is() -> None:
    result = return_degradation(oos_return=1.5, mean_is_return=1.0)
    assert result is not None
    assert result < 0


def test_degradation_none_when_is_zero() -> None:
    assert return_degradation(oos_return=0.5, mean_is_return=0.0) is None


def test_degradation_none_when_is_inf() -> None:
    assert return_degradation(oos_return=0.5, mean_is_return=math.inf) is None


def test_degradation_none_when_oos_inf() -> None:
    assert return_degradation(oos_return=math.inf, mean_is_return=1.0) is None


def test_degradation_none_when_is_negative() -> None:
    # Review MED: with mean_is_return < 0 the ratio flips sign — OOS beating a
    # losing IS (oos=+0.10, is=-0.02) computed 1-(0.10/-0.02)=6.0, misread as
    # "OOS lost too much of IS". Undefined baseline -> None (filter skipped).
    assert return_degradation(oos_return=0.10, mean_is_return=-0.02) is None
    assert return_degradation(oos_return=-0.30, mean_is_return=-0.02) is None


def test_degradation_negative_is_baseline_logs_warning(caplog) -> None:
    # Residual LOW: negative-IS None must be distinguishable in logs from the
    # zero/non-finite None cases — "couldn't compute" != "no overfit". Old code
    # returned None with no log signal at all (fails this test).
    with caplog.at_level("WARNING", logger="wfa.metrics"):
        result = return_degradation(oos_return=0.10, mean_is_return=-0.02)
    assert result is None
    assert any("negative" in r.message and "return_degradation" in r.message
               for r in caplog.records)


def test_degradation_none_when_is_zero_no_warning(caplog) -> None:
    # The plain zero/non-finite skip path is unchanged — no negative-IS warning.
    with caplog.at_level("WARNING", logger="wfa.metrics"):
        result = return_degradation(oos_return=0.5, mean_is_return=0.0)
    assert result is None
    assert not any("negative" in r.message for r in caplog.records)
