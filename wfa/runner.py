"""WFA orchestrator — runs the full walk-forward optimization loop.

For each fold:
  1. Build a search iterator over the param space.
  2. For each trial: run adapter.run(params, train_start, train_end) → IS trades.
  3. Score with the chosen objective; gate on min_trades.
  4. Pick the best IS params for this fold.
  5. Run adapter.run(best_params, test_start, test_end) → OOS trades.

Stitch all OOS trades chronologically. Compute stitched-OOS metrics and
return degradation (fraction of IS return that was curve-fit, compared on a
per-month basis over folds where both IS re-run and OOS run succeeded).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from wfa.adapter import BacktestAdapter
from wfa.folds import Fold, generate_folds
from wfa.metrics import OOSMetrics, compute_oos_metrics, return_degradation
from wfa.objectives import compute_objective
from wfa.search import make_search

logger = logging.getLogger(__name__)


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold: Fold
    best_params: dict
    is_objective: float      # best IS objective value
    is_return: float         # IS total return (for degradation calc)
    oos_trades: pd.DataFrame
    oos_n_trades: int
    # Additive fields (defaults keep old positional construction working):
    n_trials: int = 0              # IS trials actually run for this fold
    is_error: str | None = None    # IS re-run raised — is_return is NOT market data
    oos_error: str | None = None   # OOS run raised — oos_trades empty is NOT a quiet market


@dataclass
class WFAResult:
    bot_name: str
    objective: str
    search_method: str
    train_months: int
    test_months: int
    folds: list[FoldResult]
    stitched_trades: pd.DataFrame
    stitched_metrics: OOSMetrics
    mean_is_return: float
    degradation: float | None  # per-month basis over matched folds; None if not computable
    seed: int
    # Additive fields (defaults keep old positional construction working):
    n_folds_generated: int = 0        # folds generated (incl. skipped "no valid IS params" folds)
    n_failed_folds: int = 0           # folds whose OOS blind test raised — LOST EVIDENCE (verdict-invalidating)
    n_degradation_unavailable: int = 0  # folds whose IS re-run raised but OOS succeeded — degradation-only, verdict still valid
    n_trials_total: int = 0           # total IS trials across ALL generated folds (for DSR)
    mean_is_return_per_month: float = 0.0   # mean IS return / train_months (matched folds)
    oos_return_per_month: float = 0.0       # matched-fold OOS return / (n_matched * test_months)


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_wfa(
    adapter: BacktestAdapter,
    train_months: int | None = None,
    test_months: int | None = None,
    objective_name: str | None = None,
    search_method: str = "random",
    search_budget: int = 100,
    min_trades_per_fold: int = 10,
    seed: int = 42,
    initial_capital: float = 10_000.0,
    purge_days: int = 0,
) -> WFAResult:
    """Run walk-forward analysis on the given adapter.

    Args:
        adapter:           Bot adapter implementing BacktestAdapter.
        train_months:      Training window length. Defaults to adapter.recommended_windows()[0].
        test_months:       Blind-test window length. Defaults to adapter.recommended_windows()[1].
        objective_name:    Objective to optimise. Defaults to adapter.default_objective().
        search_method:     "random", "grid", or "optuna".
        search_budget:     Number of IS trials per fold (ignored for "grid").
        min_trades_per_fold: Trials with fewer trades get objective = -inf.
        seed:              Random seed for reproducibility.
        initial_capital:   Starting equity for metrics calc only (adapter handles $ sizing).
        purge_days:        Embargo — trim this many days off each fold's train END so a
                           boundary-straddling trade cannot leak IS info into OOS. Default 0 (off).
    """
    rec_train, rec_test = adapter.recommended_windows()
    if train_months is None:
        train_months = rec_train
    if test_months is None:
        test_months = rec_test
    if objective_name is None:
        objective_name = adapter.default_objective()

    data_start, data_end = adapter.data_range()
    schema = adapter.param_schema()

    folds = generate_folds(data_start, data_end, train_months, test_months, purge_days=purge_days)
    if not folds:
        logger.warning(
            "No folds generated for %s (data %s–%s, train=%d, test=%d)",
            adapter.bot_name, data_start, data_end, train_months, test_months,
        )
        empty_trades = pd.DataFrame(columns=["pnl"])
        empty_metrics = compute_oos_metrics(empty_trades, initial_capital)
        return WFAResult(
            bot_name=adapter.bot_name,
            objective=objective_name,
            search_method=search_method,
            train_months=train_months,
            test_months=test_months,
            folds=[],
            stitched_trades=empty_trades,
            stitched_metrics=empty_metrics,
            mean_is_return=0.0,
            degradation=None,
            seed=seed,
        )

    fold_results: list[FoldResult] = []
    fold_seed = seed
    n_trials_total = 0

    for fold in folds:
        logger.info(
            "Fold %d/%d  IS %s→%s  OOS %s→%s",
            fold.index + 1, len(folds),
            fold.train_start.date(), fold.train_end.date(),
            fold.test_start.date(), fold.test_end.date(),
        )

        searcher = make_search(search_method, schema, search_budget, seed=fold_seed)
        fold_seed += 1  # different seed each fold for random/optuna

        best_params: dict = {}
        best_score: float = float("-inf")
        trial_num = 0

        for params in searcher:
            try:
                is_trades = adapter.run(params, fold.train_start, fold.train_end)
            except Exception as exc:
                logger.warning("IS run failed (fold %d, params %s): %s", fold.index, params, exc)
                score = float("-inf")
            else:
                is_equity = _build_equity_list(is_trades, initial_capital)
                score = compute_objective(
                    objective_name, is_trades, is_equity, min_trades=min_trades_per_fold
                )

            if hasattr(searcher, "report_score"):
                searcher.report_score(trial_num, score)

            if score > best_score:
                best_score = score
                best_params = params
            trial_num += 1

        n_trials_total += trial_num

        if not best_params:
            logger.warning("Fold %d: no valid IS params found — skipping OOS.", fold.index)
            continue

        # IS return for degradation calc (re-run best params on IS window)
        is_error: str | None = None
        try:
            is_trades_best = adapter.run(best_params, fold.train_start, fold.train_end)
            is_equity = _build_equity_list(is_trades_best, initial_capital)
            is_ret = _total_return(is_equity)
        except Exception as exc:
            logger.warning("IS re-run failed (fold %d): %s", fold.index, exc)
            is_ret = float("-inf")
            is_error = f"{type(exc).__name__}: {exc}"

        # OOS blind test with best IS params — a raised OOS run is recorded as a
        # FOLD FAILURE, not silently counted as a 0-trade fold (swallowed-exception
        # incident class).
        oos_error: str | None = None
        try:
            oos_trades = adapter.run(best_params, fold.test_start, fold.test_end)
        except Exception as exc:
            logger.error("OOS run failed (fold %d): %s", fold.index, exc)
            oos_trades = pd.DataFrame(columns=["pnl"])
            oos_error = f"{type(exc).__name__}: {exc}"

        oos_metrics = compute_oos_metrics(oos_trades, initial_capital)

        fold_results.append(FoldResult(
            fold=fold,
            best_params=best_params,
            is_objective=best_score,
            is_return=is_ret,
            oos_trades=oos_trades,
            oos_n_trades=len(oos_trades),
            n_trials=trial_num,
            is_error=is_error,
            oos_error=oos_error,
        ))

        logger.info(
            "Fold %d: best=%s  IS_obj=%.3f  OOS_n=%d  OOS_sortino=%.3f",
            fold.index, best_params, best_score, len(oos_trades), oos_metrics.sortino,
        )

    # Stitch OOS trades chronologically
    if fold_results:
        parts = [fr.oos_trades for fr in fold_results if not fr.oos_trades.empty]
        if parts:
            stitched_trades = pd.concat(parts, ignore_index=True)
            if "exit_time" in stitched_trades.columns:
                stitched_trades = stitched_trades.sort_values("exit_time").reset_index(drop=True)
        else:
            stitched_trades = pd.DataFrame(columns=["pnl"])
    else:
        logger.error("All folds skipped — no valid IS params found across any fold.")
        stitched_trades = pd.DataFrame(columns=["pnl"])

    stitched_metrics = compute_oos_metrics(stitched_trades, initial_capital)

    # A fold's OOS blind test raising = LOST EVIDENCE — its verdict is invalid
    # (the swallowed-exception class). An IS re-run raising while OOS succeeded is
    # DEGRADATION-ONLY: the OOS blind test still stands, only the curve-fit
    # comparison for that fold is unavailable — it must NOT invalidate the run.
    n_failed_folds = sum(1 for fr in fold_results if fr.oos_error)
    n_degradation_unavailable = sum(1 for fr in fold_results if fr.is_error and not fr.oos_error)

    is_returns = [fr.is_return for fr in fold_results if math.isfinite(fr.is_return)]
    mean_is_return = float(sum(is_returns) / len(is_returns)) if is_returns else 0.0

    # Degradation on a PER-MONTH basis over MATCHED folds only:
    #   - mean per-fold IS return spans train_months; stitched OOS spans
    #     len(folds)*test_months — dividing them raw measures the window-length
    #     ratio, not curve-fit decay. Normalize both sides to return-per-month.
    #   - a fold contributes only if BOTH its IS re-run and its OOS run succeeded,
    #     so a broken side cannot skew the comparison asymmetrically.
    matched = [
        fr for fr in fold_results
        if fr.is_error is None and fr.oos_error is None and math.isfinite(fr.is_return)
    ]
    mean_is_return_per_month = 0.0
    oos_return_per_month = 0.0
    degradation: float | None = None
    if matched:
        mean_is_matched = float(sum(fr.is_return for fr in matched) / len(matched))
        mean_is_return_per_month = mean_is_matched / train_months
        oos_pnl_matched = sum(
            float(fr.oos_trades["pnl"].sum())
            for fr in matched
            if not fr.oos_trades.empty and "pnl" in fr.oos_trades.columns
        )
        oos_return_per_month = (oos_pnl_matched / initial_capital) / (len(matched) * test_months)
        degradation = return_degradation(oos_return_per_month, mean_is_return_per_month)

    return WFAResult(
        bot_name=adapter.bot_name,
        objective=objective_name,
        search_method=search_method,
        train_months=train_months,
        test_months=test_months,
        folds=fold_results,
        stitched_trades=stitched_trades,
        stitched_metrics=stitched_metrics,
        mean_is_return=mean_is_return,
        degradation=degradation,
        seed=seed,
        n_folds_generated=len(folds),
        n_failed_folds=n_failed_folds,
        n_degradation_unavailable=n_degradation_unavailable,
        n_trials_total=n_trials_total,
        mean_is_return_per_month=mean_is_return_per_month,
        oos_return_per_month=oos_return_per_month,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_equity_list(trades: pd.DataFrame, initial_capital: float) -> list[float]:
    if trades.empty or "pnl" not in trades.columns:
        return [initial_capital]
    eq = [initial_capital]
    for pnl in trades["pnl"]:
        eq.append(eq[-1] + float(pnl))
    return eq


def _total_return(equity: list[float]) -> float:
    if len(equity) < 2 or equity[0] <= 0:
        return float("-inf")
    return (equity[-1] - equity[0]) / equity[0]
