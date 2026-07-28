"""Tests for wfa/survival.py — survival gate, per-symbol breadth, DSR helper.

Added per the 2026-07-02 code review: survival.py emits the go/no-go verdict and
previously had zero test coverage.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import wfa.montecarlo
from wfa.survival import (
    SurvivalThresholds,
    deflated_sharpe_from_trade_sets,
    evaluate_survival,
    per_symbol_breakdown,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def sym_trades(spec: dict[str, list[float]]) -> pd.DataFrame:
    """Build a trades frame from {symbol: [pnls]}, interleaved deterministically."""
    rows = []
    for sym, pnls in spec.items():
        for p in pnls:
            rows.append({"symbol": sym, "pnl": p})
    return pd.DataFrame(rows)


def edge_pnls(n: int, win: float = 100.0, loss: float = -30.0) -> list[float]:
    """Alternating win/loss stream with a clear positive edge."""
    return [win if i % 2 == 0 else loss for i in range(n)]


GOOD_BOOK = sym_trades({"BTC": edge_pnls(45), "ETH": edge_pnls(30, win=80.0, loss=-25.0)})


# ── basic funnel behaviour ────────────────────────────────────────────────────

def test_empty_trades_fail() -> None:
    v = evaluate_survival(pd.DataFrame())
    assert v.passed is False
    assert v.n_trades == 0


def test_good_book_passes() -> None:
    v = evaluate_survival(GOOD_BOOK)
    assert v.passed is True
    assert v.psr is not None and v.psr >= 0.95


def test_min_trades_filter_fails_small_sample() -> None:
    v = evaluate_survival(sym_trades({"BTC": edge_pnls(10)}))
    f = {x.name: x for x in v.filters}
    assert f["min_trades"].passed is False
    assert v.passed is False


# ── #19: breadth gate must not hard-fail symbols below the min-trade floor ────

def test_breadth_incidental_symbol_is_skipped_not_failed() -> None:
    # BTC 45 + ETH 30 good trades, plus ONE incidental DOGE trade.
    # Review finding #19: DOGE n=1 -> PSR -inf -> 0.0 -> breadth 2/3 -> whole verdict FAIL.
    # Fixed: DOGE is an insufficient-data skip, excluded from the breadth denominator.
    trades = pd.concat(
        [GOOD_BOOK, pd.DataFrame([{"symbol": "DOGE", "pnl": 50.0}])],
        ignore_index=True,
    )
    v = evaluate_survival(trades)
    assert v.per_symbol is not None
    assert v.per_symbol["DOGE"]["passed"] is None          # skipped, not failed
    assert v.per_symbol["DOGE"]["psr"] is None             # not scored 0.0
    breadth = next(f for f in v.filters if f.name == "breadth")
    assert breadth.passed is True                          # 2/2 eligible symbols
    assert "2 of 2" in breadth.threshold
    assert "DOGE" in breadth.note                          # skip is visible, not silent
    assert v.passed is True


def test_breadth_all_symbols_below_floor_is_skipped_filter() -> None:
    trades = sym_trades({"BTC": edge_pnls(4), "ETH": edge_pnls(4)})
    t = SurvivalThresholds(min_trades=8, run_montecarlo=False)
    v = evaluate_survival(trades, thresholds=t)
    breadth = next(f for f in v.filters if f.name == "breadth")
    assert breadth.passed is None
    assert "no symbol" in breadth.note


def test_breadth_eligible_symbol_can_still_fail() -> None:
    # A losing symbol with enough trades must still hard-fail breadth.
    trades = sym_trades({"BTC": edge_pnls(40), "XRP": edge_pnls(40, win=30.0, loss=-100.0)})
    v = evaluate_survival(trades)
    assert v.per_symbol is not None
    assert v.per_symbol["XRP"]["passed"] is False
    breadth = next(f for f in v.filters if f.name == "breadth")
    assert breadth.passed is False


def test_per_symbol_breakdown_floor_uses_threshold() -> None:
    t = SurvivalThresholds(breadth_min_trades=20)
    out = per_symbol_breakdown(sym_trades({"BTC": edge_pnls(15)}), t)
    assert out is not None
    assert out["BTC"]["passed"] is None


def test_summary_renders_skipped_symbols() -> None:
    trades = pd.concat(
        [GOOD_BOOK, pd.DataFrame([{"symbol": "DOGE", "pnl": 50.0}])],
        ignore_index=True,
    )
    text = evaluate_survival(trades).summary()
    assert "[skip] DOGE" in text
    assert "PSR=n/a" in text


# ── MC gate: an MC crash must fail loud, never read as a legitimate skip ──────

def test_mc_exception_fails_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("numpy exploded")

    monkeypatch.setattr(wfa.montecarlo, "run_mc", boom)
    v = evaluate_survival(GOOD_BOOK)
    mc = next(f for f in v.filters if f.name == "mc_p_ruin")
    assert mc.passed is False                              # NOT None/skip
    assert "MC ERROR" in mc.note and "numpy exploded" in mc.note
    assert v.passed is False                               # gate unevaluated -> no PASS


def test_mc_legitimate_skip_below_min_trades_stays_skip() -> None:
    trades = sym_trades({"BTC": edge_pnls(5)})
    t = SurvivalThresholds(min_trades=5, mc_min_trades=10, run_montecarlo=True)
    v = evaluate_survival(trades, thresholds=t)
    mc = next(f for f in v.filters if f.name == "mc_p_ruin")
    assert mc.passed is None
    assert "skipped" in mc.note


# ── deflated Sharpe: defaulted trial count must be flagged ────────────────────

def _trade_sets() -> dict[str, pd.DataFrame]:
    return {
        "a": pd.DataFrame({"pnl": edge_pnls(30)}),
        "b": pd.DataFrame({"pnl": edge_pnls(30, win=60.0, loss=-40.0)}),
        "c": pd.DataFrame({"pnl": edge_pnls(30, win=90.0, loss=-35.0)}),
    }


def test_dsr_defaulted_trials_carries_loud_note() -> None:
    out = deflated_sharpe_from_trade_sets(_trade_sets(), best="a")
    assert out is not None
    assert "trials_note" in out
    assert "UNDER-deflated" in out["trials_note"]


def test_dsr_explicit_trials_no_note_and_deflates_more() -> None:
    out3 = deflated_sharpe_from_trade_sets(_trade_sets(), best="a", trials=800)
    assert out3 is not None
    assert "trials_note" not in out3
    assert out3["trials"] == 800
    out_default = deflated_sharpe_from_trade_sets(_trade_sets(), best="a")
    assert out3["DSR"] <= out_default["DSR"]
