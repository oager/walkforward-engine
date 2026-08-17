"""CLI guard tests for scripts/run_wfa.py — finding #11 + MED/LOW rows (review 2026-07-02).

#11: exit non-zero + LOUD INCONCLUSIVE when stitched OOS trades == 0, any fold
     failed, or >30% of generated folds were skipped — zero-evidence must never
     exit 0 with a clean-looking JSON.
MED run_wfa.py:120: wfa_result.json must be strictly valid JSON (no -Infinity/NaN).
MED run_wfa.py:84:  JSON must carry search_budget / n_trials_total for DSR.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


class SteadyAdapter:
    """One +100 trade per month — clean run."""

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


class QuietOOSAdapter(SteadyAdapter):
    """IS windows trade; every OOS window returns 0 trades WITHOUT raising."""

    bot_name = "quiet_oos"

    def run(self, params: dict, start: datetime, end: datetime) -> pd.DataFrame:
        if (end - start).days < 120:  # test windows are 3 months
            return pd.DataFrame(columns=["pnl"])
        return super().run(params, start, end)


class BrokenOOSAdapter(SteadyAdapter):
    """Every OOS window raises — broken data path."""

    bot_name = "broken_oos"

    def run(self, params: dict, start: datetime, end: datetime) -> pd.DataFrame:
        if (end - start).days < 120:
            raise RuntimeError("simulated OOS data breakage")
        return super().run(params, start, end)


class RecentDataBrokenAdapter(SteadyAdapter):
    """Windows starting 2020+ raise — mass fold-skip in the trailing data."""

    bot_name = "recent_broken"

    def run(self, params: dict, start: datetime, end: datetime) -> pd.DataFrame:
        if start >= _dt(2020):
            raise RuntimeError("data broken after 2020")
        return super().run(params, start, end)


class FlakyISRerunAdapter(SteadyAdapter):
    """Fold 0's IS re-run raises → is_return would be -inf in the JSON."""

    bot_name = "flaky_is"

    def __init__(self) -> None:
        self._train_calls: dict[datetime, int] = {}

    def run(self, params: dict, start: datetime, end: datetime) -> pd.DataFrame:
        if (end - start).days > 120:
            self._train_calls[start] = self._train_calls.get(start, 0) + 1
            if start == _dt(2018) and self._train_calls[start] == 2:
                raise RuntimeError("transient IS re-run failure")
        return super().run(params, start, end)


# ── harness ───────────────────────────────────────────────────────────────────

def _run_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, adapter: object) -> tuple[int, str]:
    """Load scripts/run_wfa.py fresh, patch its adapter loader, run main().

    Returns (exit_code, raw wfa_result.json text).
    """
    spec = importlib.util.spec_from_file_location(
        "run_wfa_cli_under_test", REPO / "scripts" / "run_wfa.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_wfa_cli_under_test"] = mod
    spec.loader.exec_module(mod)

    monkeypatch.setattr(mod, "load_adapter", lambda bot: adapter)
    monkeypatch.setattr(sys, "argv", [
        "run_wfa.py", "--bot", "x", "--search", "grid",
        "--min-trades", "1", "--out", str(tmp_path),
    ])

    exit_code = 0
    try:
        mod.main()
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1

    return exit_code, (tmp_path / "wfa_result.json").read_text()


def _strict_load(text: str) -> dict:
    """json.loads that rejects the non-standard Infinity/NaN constants."""
    def _reject(const: str) -> None:
        raise ValueError(f"non-strict JSON constant: {const}")
    return json.loads(text, parse_constant=_reject)


# ── clean run stays exit 0 ────────────────────────────────────────────────────

def test_clean_run_exits_zero_with_status_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, text = _run_cli(monkeypatch, tmp_path, SteadyAdapter())
    assert code == 0
    data = _strict_load(text)
    assert data["status"] == "OK"
    assert data["inconclusive_reasons"] == []
    assert data["n_folds_failed"] == 0


# ── #11: zero stitched OOS trades ─────────────────────────────────────────────

def test_zero_stitched_trades_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, text = _run_cli(monkeypatch, tmp_path, QuietOOSAdapter())
    assert code != 0
    data = _strict_load(text)
    assert data["status"] == "INCONCLUSIVE"
    assert data["summary"]["n_trades"] == 0
    assert any("0 trades" in r for r in data["inconclusive_reasons"])


# ── #11: failed folds ─────────────────────────────────────────────────────────

def test_failed_folds_exit_nonzero_and_are_counted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, text = _run_cli(monkeypatch, tmp_path, BrokenOOSAdapter())
    assert code != 0
    data = _strict_load(text)
    assert data["status"] == "INCONCLUSIVE"
    assert data["n_folds_failed"] == data["n_folds"] > 0
    # broken fold must be distinguishable from a quiet market in the artifact
    assert all(f["oos_error"] for f in data["folds"])


# ── #11: mass fold-skip (>30%) ────────────────────────────────────────────────

def test_mass_fold_skip_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, text = _run_cli(monkeypatch, tmp_path, RecentDataBrokenAdapter())
    assert code != 0
    data = _strict_load(text)
    assert data["status"] == "INCONCLUSIVE"
    assert data["n_folds_generated"] == 14
    assert data["n_folds_skipped"] == 6   # 6/14 > 30%
    assert data["n_folds_failed"] == 2


# ── MED run_wfa.py:120: strictly valid JSON ───────────────────────────────────

def test_json_never_contains_infinity_or_nan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, text = _run_cli(monkeypatch, tmp_path, FlakyISRerunAdapter())
    data = _strict_load(text)  # raises on -Infinity/NaN constants (old code emitted them)
    # the failed fold's is_return is null, with the reason preserved
    flagged = [f for f in data["folds"] if f["is_error"]]
    assert len(flagged) == 1
    assert flagged[0]["is_return"] is None


# ── MED run_wfa.py:84: DSR inputs present ─────────────────────────────────────

def test_json_carries_search_budget_and_trial_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, text = _run_cli(monkeypatch, tmp_path, SteadyAdapter())
    data = _strict_load(text)
    assert data["search_budget"] == 100          # argparse default
    assert data["min_trades_per_fold"] == 1
    assert data["n_trials_total"] == data["n_folds_generated"] > 0
    assert data["degradation_basis"] == "per_month_matched_folds"
