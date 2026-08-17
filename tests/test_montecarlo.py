"""Tests for wfa/montecarlo.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import wfa.objectives as objectives
from wfa.montecarlo import MCResult, _ann_factor, _max_dd_rows, run_mc


def make_trades(pnls: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"pnl": pnls})


TRADES_30 = make_trades([200.0, -50.0, 100.0, -30.0, 150.0] * 6)


# ── validation ────────────────────────────────────────────────────────────────

def test_run_mc_empty_trades_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        run_mc(make_trades([]))


def test_run_mc_missing_pnl_column_raises() -> None:
    with pytest.raises(ValueError, match="pnl"):
        run_mc(pd.DataFrame({"side": ["long"]}))


# ── basic structure ───────────────────────────────────────────────────────────

def test_run_mc_returns_mc_result() -> None:
    result = run_mc(TRADES_30, n_sims=100, seed=0)
    assert isinstance(result, MCResult)


def test_run_mc_n_trades_correct() -> None:
    result = run_mc(TRADES_30, n_sims=100, seed=0)
    assert result.n_trades == 30


def test_run_mc_equity_curves_shape() -> None:
    result = run_mc(TRADES_30, n_sims=100, seed=0)
    assert result.equity_curves.shape == (100, 31)  # n_sims × (n_trades+1)


def test_run_mc_equity_curves_start_at_initial() -> None:
    result = run_mc(TRADES_30, n_sims=100, initial_capital=5_000.0, seed=0)
    assert np.all(result.equity_curves[:, 0] == 5_000.0)


def test_run_mc_source_tag() -> None:
    r = run_mc(TRADES_30, n_sims=50, source="stitched_oos", seed=0)
    assert r.source == "stitched_oos"
    r2 = run_mc(TRADES_30, n_sims=50, source="raw_trades", seed=0)
    assert r2.source == "raw_trades"


# ── reproducibility ───────────────────────────────────────────────────────────

def test_run_mc_same_seed_reproducible() -> None:
    r1 = run_mc(TRADES_30, n_sims=500, seed=42)
    r2 = run_mc(TRADES_30, n_sims=500, seed=42)
    assert r1.probability_of_ruin == r2.probability_of_ruin
    np.testing.assert_array_equal(r1.equity_curves, r2.equity_curves)


def test_run_mc_different_seed_different_result() -> None:
    r1 = run_mc(TRADES_30, n_sims=500, seed=42)
    r2 = run_mc(TRADES_30, n_sims=500, seed=99)
    assert not np.array_equal(r1.equity_curves, r2.equity_curves)


# ── reshuffle vs bootstrap ────────────────────────────────────────────────────

def test_reshuffle_preserves_total_return() -> None:
    """Reshuffle preserves exact realized sum per path — total return pct is constant."""
    trades = make_trades([100.0, -50.0, 200.0, -30.0, 80.0] * 6)
    result = run_mc(trades, n_sims=100, method="reshuffle", seed=0, compounding=False)
    # All paths have same final return (additive, permutation = same sum)
    ret_min = result.all_return_pct.min()
    ret_max = result.all_return_pct.max()
    assert abs(ret_max - ret_min) < 1e-6


def test_bootstrap_has_return_variance() -> None:
    """Bootstrap varies total return across paths."""
    result = run_mc(TRADES_30, n_sims=500, method="bootstrap", seed=0, compounding=False)
    assert result.all_return_pct.std() > 0.01


# ── ruin probability ──────────────────────────────────────────────────────────

def test_ruin_prob_all_wins_is_zero() -> None:
    """Monotonically rising equity → no sim hits ruin threshold."""
    trades = make_trades([100.0] * 30)
    result = run_mc(trades, n_sims=500, ruin_threshold=0.20, seed=0)
    assert result.probability_of_ruin == 0.0


def test_ruin_prob_all_losses_is_one() -> None:
    """All-loss trades → every sim hits ruin."""
    trades = make_trades([-500.0] * 30)
    result = run_mc(trades, n_sims=100, initial_capital=10_000.0, ruin_threshold=0.20, seed=0)
    assert result.probability_of_ruin == 1.0


# ── percentile structure ──────────────────────────────────────────────────────

def test_percentiles_ordered() -> None:
    result = run_mc(TRADES_30, n_sims=500, seed=0)
    ret = result.total_return_pct
    assert ret.p5 <= ret.p25 <= ret.p50 <= ret.p75 <= ret.p95


def test_as_dict_keys() -> None:
    result = run_mc(TRADES_30, n_sims=100, seed=0)
    d = result.as_dict()
    expected = {
        "source", "method", "n_sims", "n_trades", "initial_capital",
        "seed", "compounding", "ruin_threshold", "probability_of_ruin",
        "total_return_pct", "max_drawdown_pct", "losing_streak", "sharpe", "sortino",
    }
    assert set(d.keys()) == expected


# ── _max_dd_rows ──────────────────────────────────────────────────────────────

def test_max_dd_rows_no_drawdown() -> None:
    eq = np.array([[10_000.0, 10_100.0, 10_200.0, 10_300.0]])
    assert _max_dd_rows(eq)[0] == 0.0


def test_max_dd_rows_known_value() -> None:
    # Peak at 10_200, trough at 9_800 → DD = 400/10_200 × 100 ≈ 3.92%
    eq = np.array([[10_000.0, 10_200.0, 9_800.0, 10_500.0]])
    dd = _max_dd_rows(eq)[0]
    assert abs(dd - 400 / 10_200 * 100) < 1e-6


# ── per-path Sharpe uses sample std (ddof=1), consistent with objectives ──────

def test_mc_sharpe_uses_sample_std_ddof1() -> None:
    # Reshuffle permutes each path, so per-path mean/std equal the input's —
    # every path's Sharpe is exactly mean_r / std_r(ddof=1) (ann factor 1.0,
    # no exit_time). The old ddof=0 population std inflated it by sqrt(n/(n-1)).
    pnls = np.array([200.0, -50.0, 100.0, -30.0, 150.0] * 6)
    returns = pnls / 10_000.0
    expected = returns.mean() / returns.std(ddof=1)
    result = run_mc(make_trades(list(pnls)), n_sims=50, method="reshuffle", seed=0)
    assert abs(result.sharpe.p50 - expected) < 1e-3
    assert abs(result.sharpe.p5 - result.sharpe.p95) < 1e-6  # permutation-invariant


# ── _ann_factor is the shared objectives implementation (residual LOW) ─────────

def test_montecarlo_ann_factor_is_objectives_ann_factor() -> None:
    # montecarlo.py used to define its own silent duplicate. Dedup means the two
    # names must now be the identical function object, not just equal behavior.
    assert _ann_factor is objectives._ann_factor


def test_montecarlo_ann_factor_warns_on_missing_exit_time(caplog) -> None:
    # Old montecarlo._ann_factor copy returned 1.0 on missing 'exit_time' with NO
    # log signal. The shared objectives._ann_factor fires _warn_ann_fallback once
    # per process — fails on the old duplicate (no warning record emitted).
    objectives._ANN_FALLBACK_WARNED = False
    trades = make_trades([100.0, -50.0, 100.0])  # no 'exit_time' column
    with caplog.at_level("WARNING", logger="wfa.objectives"):
        factor = _ann_factor(trades)
    assert factor == 1.0
    assert any("annualisation degraded to factor 1.0" in r.message for r in caplog.records)
