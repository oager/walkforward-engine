"""Tests for wfa/objectives.py and wfa/search.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from wfa.objectives import (
    OBJECTIVE_NAMES,
    _calmar,
    _profit_factor,
    _sharpe,
    _sortino,
    _total_return,
    _win_rate,
    compute_objective,
)
from wfa.search import FullGrid, OptunaSearch, RandomSearch, make_search

# ── helpers ───────────────────────────────────────────────────────────────────

def make_trades(pnls: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"pnl": pnls})


def make_equity(pnls: list[float], initial: float = 10_000.0) -> list[float]:
    eq = [initial]
    for p in pnls:
        eq.append(eq[-1] + p)
    return eq


# ── objectives registry ───────────────────────────────────────────────────────

def test_all_objectives_registered() -> None:
    expected = {"sortino", "calmar", "sharpe", "total_return", "win_rate", "profit_factor", "psr"}
    assert expected == set(OBJECTIVE_NAMES)


def test_unknown_objective_raises() -> None:
    trades = make_trades([100.0] * 20)
    equity = make_equity([100.0] * 20)
    with pytest.raises(ValueError):
        compute_objective("nonexistent", trades, equity)


# ── min-trades gate ───────────────────────────────────────────────────────────

def test_min_trades_gate_returns_neg_inf() -> None:
    trades = make_trades([100.0] * 5)
    equity = make_equity([100.0] * 5)
    result = compute_objective("sortino", trades, equity, min_trades=10)
    assert result == -math.inf


def test_min_trades_gate_passes_at_threshold() -> None:
    trades = make_trades([100.0] * 10)
    equity = make_equity([100.0] * 10)
    result = compute_objective("sortino", trades, equity, min_trades=10)
    assert math.isfinite(result)


# ── total_return ──────────────────────────────────────────────────────────────

def test_total_return_known_value() -> None:
    trades = make_trades([1000.0] * 20)
    equity = make_equity([1000.0] * 20, initial=10_000.0)
    # final = 30_000, initial = 10_000 → return = 2.0
    result = _total_return(trades, equity)
    assert abs(result - 2.0) < 1e-9


def test_total_return_loss() -> None:
    trades = make_trades([-100.0] * 20)
    equity = make_equity([-100.0] * 20, initial=10_000.0)
    result = _total_return(trades, equity)
    assert result < 0


def test_total_return_empty_equity() -> None:
    assert _total_return(make_trades([]), []) == -math.inf


# ── win_rate ──────────────────────────────────────────────────────────────────

def test_win_rate_all_wins() -> None:
    trades = make_trades([100.0] * 20)
    equity = make_equity([100.0] * 20)
    assert abs(_win_rate(trades, equity) - 1.0) < 1e-9


def test_win_rate_half() -> None:
    trades = make_trades([100.0, -100.0] * 10)
    equity = make_equity([100.0, -100.0] * 10)
    assert abs(_win_rate(trades, equity) - 0.5) < 1e-9


def test_win_rate_empty_trades() -> None:
    assert _win_rate(make_trades([]), []) == -math.inf


# ── profit_factor ─────────────────────────────────────────────────────────────

def test_profit_factor_known_value() -> None:
    trades = make_trades([200.0, 200.0, -100.0, -100.0])
    equity = make_equity([200.0, 200.0, -100.0, -100.0])
    # gross profit = 400, gross loss = 200 → PF = 2.0
    assert abs(_profit_factor(trades, equity) - 2.0) < 1e-9


def test_profit_factor_no_losses() -> None:
    trades = make_trades([100.0] * 10)
    equity = make_equity([100.0] * 10)
    assert _profit_factor(trades, equity) == 100.0


def test_profit_factor_all_losses() -> None:
    trades = make_trades([-100.0] * 10)
    equity = make_equity([-100.0] * 10)
    assert _profit_factor(trades, equity) == 0.0  # no wins → PF = 0/loss = 0


# ── sharpe ────────────────────────────────────────────────────────────────────

def test_sharpe_positive_for_upward_curve() -> None:
    pnls = [100.0] * 30
    equity = make_equity(pnls)
    trades = make_trades(pnls)
    # Constant positive returns → positive Sharpe (std > 0 due to floating point won't apply)
    # Actually constant returns → std=0 → -inf; use variable
    pnls = [100.0 + (i % 3) * 10 for i in range(30)]
    equity = make_equity(pnls)
    trades = make_trades(pnls)
    result = _sharpe(trades, equity)
    assert math.isfinite(result)
    assert result > 0


def test_sharpe_neg_inf_for_empty() -> None:
    assert _sharpe(make_trades([]), [10_000.0]) == -math.inf


# ── sortino ───────────────────────────────────────────────────────────────────

def test_sortino_all_positive_returns_high_value() -> None:
    pnls = [100.0, 120.0, 80.0, 110.0, 90.0] * 8
    equity = make_equity(pnls)
    trades = make_trades(pnls)
    result = _sortino(trades, equity)
    assert result == 100.0  # no downside returns → returns sentinel 100.0


def test_sortino_positive_for_mixed() -> None:
    pnls = [200.0, -50.0] * 20
    equity = make_equity(pnls)
    trades = make_trades(pnls)
    result = _sortino(trades, equity)
    assert math.isfinite(result)
    assert result > 0


# ── calmar ────────────────────────────────────────────────────────────────────

def test_calmar_pos_for_profitable_low_dd() -> None:
    pnls = [100.0 + i for i in range(40)]
    equity = make_equity(pnls)
    trades = make_trades(pnls)
    result = _calmar(trades, equity)
    assert math.isfinite(result)
    assert result > 0


def test_calmar_neg_inf_for_empty() -> None:
    assert _calmar(make_trades([]), [10_000.0]) == -math.inf


# ── RandomSearch ─────────────────────────────────────────────────────────────

SCHEMA_SMALL = {
    "a": {"choices": [1, 2, 3]},
    "b": {"choices": ["x", "y"]},
}  # 6 combos total


def test_random_search_seed_deterministic() -> None:
    r1 = list(RandomSearch(SCHEMA_SMALL, budget=6, seed=42))
    r2 = list(RandomSearch(SCHEMA_SMALL, budget=6, seed=42))
    assert r1 == r2


def test_random_search_different_seeds() -> None:
    r1 = list(RandomSearch(SCHEMA_SMALL, budget=6, seed=42))
    r2 = list(RandomSearch(SCHEMA_SMALL, budget=6, seed=99))
    assert r1 != r2


def test_random_search_budget_under_total_no_duplicates() -> None:
    results = list(RandomSearch(SCHEMA_SMALL, budget=4, seed=0))
    assert len(results) == 4
    tuples = [tuple(sorted(d.items())) for d in results]
    assert len(set(tuples)) == 4  # no duplicates


def test_random_search_budget_equals_total_all_covered() -> None:
    results = list(RandomSearch(SCHEMA_SMALL, budget=6, seed=0))
    assert len(results) == 6
    tuples = {tuple(sorted(d.items())) for d in results}
    assert len(tuples) == 6  # all 6 unique combos


def test_random_search_yields_dicts_with_correct_keys() -> None:
    for params in RandomSearch(SCHEMA_SMALL, budget=3, seed=0):
        assert set(params.keys()) == {"a", "b"}
        assert params["a"] in [1, 2, 3]
        assert params["b"] in ["x", "y"]


def test_random_search_total_combos_property() -> None:
    rs = RandomSearch(SCHEMA_SMALL, budget=6)
    assert rs.total_combos == 6


# ── FullGrid ──────────────────────────────────────────────────────────────────

def test_full_grid_all_combos() -> None:
    results = list(FullGrid(SCHEMA_SMALL))
    assert len(results) == 6


def test_full_grid_no_duplicates() -> None:
    results = list(FullGrid(SCHEMA_SMALL))
    tuples = {tuple(sorted(d.items())) for d in results}
    assert len(tuples) == 6


def test_full_grid_larger_schema() -> None:
    schema = {
        "x": {"choices": list(range(4))},
        "y": {"choices": list(range(4))},
        "z": {"choices": list(range(3))},
    }
    results = list(FullGrid(schema))
    assert len(results) == 4 * 4 * 3  # 48


def test_full_grid_total_combos() -> None:
    assert FullGrid(SCHEMA_SMALL).total_combos == 6


# ── make_search factory ───────────────────────────────────────────────────────

def test_make_search_random() -> None:
    s = make_search("random", SCHEMA_SMALL, budget=4)
    assert isinstance(s, RandomSearch)


def test_make_search_grid() -> None:
    s = make_search("grid", SCHEMA_SMALL, budget=4)
    assert isinstance(s, FullGrid)


def test_make_search_optuna() -> None:
    pytest.importorskip("optuna")
    s = make_search("optuna", SCHEMA_SMALL, budget=4)
    assert isinstance(s, OptunaSearch)


def test_make_search_unknown_raises() -> None:
    with pytest.raises(ValueError):
        make_search("nonexistent", SCHEMA_SMALL, budget=4)


# ── PSR — review #7: formula requires NON-excess kurtosis (+3 vs pandas) ──────

def test_psr_hand_computed_non_excess_kurtosis() -> None:
    """Regression for review finding #7 (PSR systematically inflated).

    Fat-tailed 60-trade stream; expected PSR hand-computed with the Bailey &
    Lopez de Prado variance term using NON-excess kurtosis (pandas .kurtosis()+3):
      sr_obs = mean/std(ddof=1), skew = pandas .skew(), n = 60
      var = (1 - skew*sr + (kurt-1)/4*sr^2) / (n-1);  PSR = Phi(sr/sqrt(var))
    Buggy excess-kurtosis value for the same series is 0.7495998602410865.
    """
    from wfa.objectives import _psr

    pnls = ([90.0, 110.0, 100.0, 95.0, 105.0, 85.0, 115.0, 92.0, 108.0, -700.0] * 6)[:60]
    trades = make_trades(pnls)
    equity = make_equity(pnls)
    psr = _psr(trades, equity)
    assert abs(psr - 0.7489852888452224) < 1e-9
    assert psr < 0.7495998602410865  # excess-kurtosis bug inflated PSR


# ── equity crossing zero — %-returns are sign-inverted past that point ────────

BLOWN_PNLS = [-6_000.0, -8_000.0, 1_000.0, -500.0, 800.0, -300.0, 400.0, -200.0]


def test_sharpe_neg_inf_when_equity_crosses_zero() -> None:
    trades = make_trades(BLOWN_PNLS)
    equity = make_equity(BLOWN_PNLS)  # 10k -> 4k -> -4k -> ...
    assert min(equity) < 0
    assert _sharpe(trades, equity) == -math.inf


def test_sortino_neg_inf_when_equity_crosses_zero() -> None:
    trades = make_trades(BLOWN_PNLS)
    equity = make_equity(BLOWN_PNLS)
    assert _sortino(trades, equity) == -math.inf


def test_psr_neg_inf_when_equity_crosses_zero() -> None:
    from wfa.objectives import _psr

    trades = make_trades(BLOWN_PNLS)
    equity = make_equity(BLOWN_PNLS)
    assert _psr(trades, equity) == -math.inf


# ── silent annualisation fallback must warn ───────────────────────────────────

def test_ann_fallback_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    import wfa.objectives as obj

    obj._ANN_FALLBACK_WARNED = False  # reset one-shot flag
    pnls = [100.0, -30.0] * 10
    with caplog.at_level("WARNING", logger="wfa.objectives"):
        _sharpe(make_trades(pnls), make_equity(pnls))  # no exit_time column
    assert "UNANNUALISED" in caplog.text
    # second call: warned once only
    caplog.clear()
    with caplog.at_level("WARNING", logger="wfa.objectives"):
        _sharpe(make_trades(pnls), make_equity(pnls))
    assert caplog.text == ""


# ── RandomSearch must not materialise huge Cartesian products ─────────────────

def test_random_search_huge_space_does_not_materialise() -> None:
    # 20^10 ≈ 1e13 combos — the old list(itertools.product(...)) would OOM/hang.
    schema = {f"p{i}": {"choices": list(range(20))} for i in range(10)}
    rs = RandomSearch(schema, budget=8, seed=7)
    assert rs.total_combos == 20 ** 10
    results = list(rs)
    assert len(results) == 8
    tuples = {tuple(sorted(d.items())) for d in results}
    assert len(tuples) == 8  # without replacement
    for params in results:
        assert set(params.keys()) == set(schema.keys())
        for k, v in params.items():
            assert v in schema[k]["choices"]
    # deterministic per seed
    assert results == list(RandomSearch(schema, budget=8, seed=7))
    assert results != list(RandomSearch(schema, budget=8, seed=8))
