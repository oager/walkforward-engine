"""Fail-loud runner tests — findings #13/#14 + IS/OOS asymmetry (review 2026-07-02).

#13: an OOS adapter exception must be recorded per-fold (FoldResult.oos_error,
     WFAResult.n_failed_folds) — a broken fold must be distinguishable from a
     legitimately quiet market.
#14: degradation must compare IS and OOS on the same per-month basis over
     matched folds, not raw train-window return vs stitched multi-fold return.
MED runner.py:164: an IS re-run failure must not silently drop the fold from
     the IS mean while its OOS trades stay in the comparison.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from wfa.runner import run_wfa


def _dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


class SteadyAdapter:
    """One +100 trade per month, regardless of params — constant per-month edge."""

    bot_name = "steady"
    timeframe = "4h"

    def param_schema(self) -> dict:
        return {
            "threshold": {"type": "float", "choices": [1.0], "default": 1.0, "help": ""},
        }

    def recommended_windows(self) -> tuple[int, int]:
        return (6, 3)

    def default_objective(self) -> str:
        return "sortino"

    def data_snapshot_path(self) -> Path:
        return Path(".")

    def data_range(self) -> tuple[datetime, datetime]:
        return _dt(2018), _dt(2022)

    def run(self, params: dict, start: datetime, end: datetime) -> pd.DataFrame:
        n_months = max(1, round((end - start).days / 30.44))
        return pd.DataFrame({"pnl": [100.0] * n_months})


class BrokenOOSAdapter(SteadyAdapter):
    """OOS (short) windows raise; IS (long) windows are fine — broken recent data."""

    bot_name = "broken_oos"

    def run(self, params: dict, start: datetime, end: datetime) -> pd.DataFrame:
        if (end - start).days < 120:  # test windows are 3 months
            raise RuntimeError("simulated OOS data breakage")
        return super().run(params, start, end)


class FlakyISRerunAdapter(SteadyAdapter):
    """Fold 0's IS re-run (2nd call on the first train window) raises transiently."""

    bot_name = "flaky_is"

    def __init__(self) -> None:
        self._train_calls: dict[datetime, int] = {}

    def run(self, params: dict, start: datetime, end: datetime) -> pd.DataFrame:
        if (end - start).days > 120:  # train window
            self._train_calls[start] = self._train_calls.get(start, 0) + 1
            if start == _dt(2018) and self._train_calls[start] == 2:
                raise RuntimeError("transient IS re-run failure")
        return super().run(params, start, end)


class RecentDataBrokenAdapter(SteadyAdapter):
    """Any window touching 2020+ raises — trailing folds skip (IS) or fail (OOS)."""

    bot_name = "recent_broken"

    def run(self, params: dict, start: datetime, end: datetime) -> pd.DataFrame:
        if start >= _dt(2020):
            raise RuntimeError("data broken after 2020")
        return super().run(params, start, end)


# ── #14: per-month degradation basis ─────────────────────────────────────────

def test_degradation_zero_for_constant_per_month_edge() -> None:
    """Identical per-month edge IS and OOS must give degradation ~0, not the
    window-length ratio (old code: 1 - 42mo/6mo = -600%)."""
    result = run_wfa(SteadyAdapter(), search_method="grid", min_trades_per_fold=1, seed=7)
    assert len(result.folds) > 1
    assert result.degradation is not None
    assert abs(result.degradation) < 1e-9


def test_per_month_fields_populated() -> None:
    result = run_wfa(SteadyAdapter(), search_method="grid", min_trades_per_fold=1, seed=7)
    # 100 $/month on 10k capital = 0.01 return per month, both sides
    assert abs(result.mean_is_return_per_month - 0.01) < 1e-9
    assert abs(result.oos_return_per_month - 0.01) < 1e-9


# ── #13: OOS failure recorded per fold ────────────────────────────────────────

def test_oos_failure_recorded_per_fold() -> None:
    result = run_wfa(BrokenOOSAdapter(), search_method="grid", min_trades_per_fold=1, seed=7)
    assert len(result.folds) > 0
    assert result.n_failed_folds == len(result.folds)
    for fr in result.folds:
        assert fr.oos_error is not None
        assert "breakage" in fr.oos_error
        assert fr.oos_n_trades == 0


def test_all_oos_failed_gives_no_degradation_verdict() -> None:
    """All-broken OOS must NOT read as 'strategy stopped trading' (old code:
    degradation = 1 - 0/IS = 1.0, a clean-looking full-decay verdict)."""
    result = run_wfa(BrokenOOSAdapter(), search_method="grid", min_trades_per_fold=1, seed=7)
    assert result.degradation is None


def test_clean_run_has_no_failures_flagged() -> None:
    result = run_wfa(SteadyAdapter(), search_method="grid", min_trades_per_fold=1, seed=7)
    assert result.n_failed_folds == 0
    for fr in result.folds:
        assert fr.is_error is None
        assert fr.oos_error is None


# ── MED runner.py:164: IS re-run failure asymmetry ────────────────────────────

def test_is_rerun_failure_flagged_and_excluded_from_degradation() -> None:
    result = run_wfa(
        FlakyISRerunAdapter(), search_method="grid", min_trades_per_fold=1, seed=7
    )
    flagged = [fr for fr in result.folds if fr.is_error is not None]
    assert len(flagged) == 1
    # An IS re-run failure with a VALID OOS is degradation-only — it must NOT
    # count as a failed fold (which would force the whole run INCONCLUSIVE);
    # it's tracked separately (re-review 2026-07-02 regression fix).
    assert result.n_failed_folds == 0
    assert result.n_degradation_unavailable == 1
    # Fold 0's OOS trades must not skew the comparison while its IS is missing:
    # matched-fold degradation stays ~0 for the constant edge.
    assert result.degradation is not None
    assert abs(result.degradation) < 1e-9


# ── fold accounting (feeds #11's CLI guard) ───────────────────────────────────

def test_generated_vs_survived_vs_failed_accounting() -> None:
    result = run_wfa(
        RecentDataBrokenAdapter(), search_method="grid", min_trades_per_fold=1, seed=7
    )
    # 2018-2022, train=6/test=3 → 14 generated folds. Train windows starting
    # 2020+ are skipped (6 folds); folds whose OOS window starts 2020+ fail (2).
    assert result.n_folds_generated == 14
    assert len(result.folds) == 8
    assert result.n_failed_folds == 2


def test_n_trials_total_counts_all_generated_folds() -> None:
    result = run_wfa(SteadyAdapter(), search_method="grid", min_trades_per_fold=1, seed=7)
    # grid over a single choice = 1 trial per generated fold
    assert result.n_trials_total == result.n_folds_generated
    assert result.n_trials_total > 0


# ── purge_days wiring: run_wfa must thread the embargo into generate_folds ────
# folds.py implements the embargo correctly, but run_wfa historically called
# generate_folds() with no purge_days arg (defaulted 0) → the embargo was
# unreachable from the runner/CLI. These lock the thread in place.

def test_purge_days_threaded_from_runner_to_generate_folds(monkeypatch) -> None:
    import wfa.runner as _runner
    captured: dict = {}
    real = _runner.generate_folds

    def _spy(*args, **kwargs):
        captured["purge_days"] = kwargs.get("purge_days")
        return real(*args, **kwargs)

    monkeypatch.setattr(_runner, "generate_folds", _spy)
    run_wfa(SteadyAdapter(), search_method="grid", min_trades_per_fold=1, seed=7, purge_days=5)
    assert captured["purge_days"] == 5


def test_purge_days_defaults_to_zero_from_runner(monkeypatch) -> None:
    import wfa.runner as _runner
    captured: dict = {}
    real = _runner.generate_folds

    def _spy(*args, **kwargs):
        captured["purge_days"] = kwargs.get("purge_days")
        return real(*args, **kwargs)

    monkeypatch.setattr(_runner, "generate_folds", _spy)
    run_wfa(SteadyAdapter(), search_method="grid", min_trades_per_fold=1, seed=7)
    assert captured["purge_days"] == 0
