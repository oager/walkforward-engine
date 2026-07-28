"""Synthetic adapter tests for wfa/runner.py.

Uses a fake adapter that returns deterministic trades — no real bot needed.
Tests the fold orchestration, IS/OOS split, stitching, and degradation calc.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from wfa.runner import WFAResult, run_wfa


def _dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


# ── synthetic adapter ─────────────────────────────────────────────────────────

class SyntheticAdapter:
    """Returns a fixed number of profitable trades per run — deterministic."""

    bot_name = "synthetic"
    timeframe = "1h"

    def param_schema(self) -> dict:
        return {
            "threshold": {"type": "float", "choices": [0.5, 1.0, 1.5], "default": 1.0, "help": ""},
        }

    def recommended_windows(self) -> tuple[int, int]:
        return (6, 3)

    def default_objective(self) -> str:
        return "sortino"

    def data_snapshot_path(self) -> Path:
        return Path("/dev/null")

    def data_range(self) -> tuple[datetime, datetime]:
        return _dt(2018), _dt(2022)

    def run(self, params: dict, start: datetime, end: datetime) -> pd.DataFrame:
        """Return N trades where N scales with params['threshold'].

        Higher threshold → more trades (to make it clearly "winning" in IS).
        """
        n_months = max(1, round((end - start).days / 30))
        n_trades = max(1, int(params["threshold"] * n_months * 3))
        pnls = [100.0 if i % 3 != 0 else -50.0 for i in range(n_trades)]
        return pd.DataFrame({"pnl": pnls})


ADAPTER = SyntheticAdapter()


# ── basic structure ───────────────────────────────────────────────────────────

def test_run_wfa_returns_wfa_result() -> None:
    result = run_wfa(ADAPTER, seed=42)
    assert isinstance(result, WFAResult)


def test_run_wfa_correct_bot_name() -> None:
    result = run_wfa(ADAPTER, seed=42)
    assert result.bot_name == "synthetic"


def test_run_wfa_has_folds() -> None:
    result = run_wfa(ADAPTER, seed=42)
    assert len(result.folds) > 0


def test_run_wfa_uses_recommended_windows() -> None:
    result = run_wfa(ADAPTER, seed=42)
    assert result.train_months == 6
    assert result.test_months == 3


def test_run_wfa_accepts_custom_windows() -> None:
    result = run_wfa(ADAPTER, train_months=12, test_months=6, seed=42)
    assert result.train_months == 12
    assert result.test_months == 6


# ── fold integrity ────────────────────────────────────────────────────────────

def test_oos_trades_non_empty() -> None:
    result = run_wfa(ADAPTER, seed=42)
    for fr in result.folds:
        assert not fr.oos_trades.empty, f"Fold {fr.fold.index} has empty OOS trades"


def test_best_params_have_correct_keys() -> None:
    result = run_wfa(ADAPTER, seed=42)
    for fr in result.folds:
        assert "threshold" in fr.best_params


def test_is_objective_finite_for_valid_folds() -> None:
    result = run_wfa(ADAPTER, min_trades_per_fold=5, seed=42)
    for fr in result.folds:
        assert math.isfinite(fr.is_objective), f"Fold {fr.fold.index} IS obj not finite"


# ── stitched OOS ─────────────────────────────────────────────────────────────

def test_stitched_trades_count_equals_sum_of_fold_oos() -> None:
    result = run_wfa(ADAPTER, seed=42)
    expected = sum(fr.oos_n_trades for fr in result.folds)
    assert len(result.stitched_trades) == expected


def test_stitched_metrics_has_pnl_column() -> None:
    result = run_wfa(ADAPTER, seed=42)
    assert "pnl" in result.stitched_trades.columns


# ── degradation ───────────────────────────────────────────────────────────────

def test_degradation_is_finite_for_profitable_synthetic() -> None:
    result = run_wfa(ADAPTER, seed=42)
    if result.mean_is_return > 0 and math.isfinite(result.stitched_metrics.total_return):
        assert result.degradation is not None
        assert math.isfinite(result.degradation)


# ── reproducibility ───────────────────────────────────────────────────────────

def test_same_seed_same_result() -> None:
    r1 = run_wfa(ADAPTER, seed=42)
    r2 = run_wfa(ADAPTER, seed=42)
    assert len(r1.folds) == len(r2.folds)
    for fr1, fr2 in zip(r1.folds, r2.folds):
        assert fr1.best_params == fr2.best_params
        assert fr1.is_objective == fr2.is_objective


# ── no-folds edge case ────────────────────────────────────────────────────────

def test_empty_result_when_range_too_short() -> None:
    class TinyAdapter(SyntheticAdapter):
        def data_range(self):
            return _dt(2020), _dt(2020, 6)  # only 6 months, can't fit 6+3=9 months

    result = run_wfa(TinyAdapter(), seed=42)
    assert result.folds == []
    assert result.stitched_trades.empty


# ── min-trades gate ───────────────────────────────────────────────────────────

def test_min_trades_gate_with_impossible_threshold() -> None:
    """Setting min_trades=9999 → no valid IS params → folds are skipped."""
    result = run_wfa(ADAPTER, min_trades_per_fold=9999, seed=42)
    # All folds skipped → no fold results
    assert result.folds == []
