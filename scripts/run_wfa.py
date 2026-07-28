#!/usr/bin/env python3
"""Headless CLI runner for Walk-Forward Analysis.

Usage:
    python scripts/run_wfa.py --bot mybot --objective sortino \
        --search random --budget 100 --train-mo 18 --test-mo 6 \
        --seed 42 --out runs/mybot_v1/
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from wfa.registry import load_adapter
from wfa.runner import run_wfa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_wfa")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WFA headless runner")
    p.add_argument("--bot", required=True, help="Bot name from ~/.wfa/config.toml")
    p.add_argument("--objective", default=None, help="Objective (default: adapter default)")
    p.add_argument("--search", default="random", choices=["random", "grid", "optuna"])
    p.add_argument("--budget", type=int, default=100, help="Trials per fold (random/optuna)")
    p.add_argument("--train-mo", type=int, default=None)
    p.add_argument("--test-mo", type=int, default=None)
    p.add_argument("--min-trades", type=int, default=10)
    p.add_argument("--purge-days", type=int, default=0, help="Embargo days trimmed off each fold's train end (IS->OOS leakage guard). Default 0 = off.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--out", type=str, default=None, help="Output directory for wfa_result.json")
    return p.parse_args()


def _json_safe(obj: object) -> object:
    """Replace non-finite floats with None so the emitted JSON is strictly valid.

    json.dumps would otherwise emit -Infinity/NaN, which is invalid JSON and
    breaks strict downstream parsers (jq, JS JSON.parse). A null + the fold's
    is_error/oos_error field keeps failures distinguishable from data.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def main() -> None:
    args = _parse_args()

    adapter = load_adapter(args.bot)
    logger.info("Loaded adapter: %s", adapter.bot_name)

    result = run_wfa(
        adapter=adapter,
        train_months=args.train_mo,
        test_months=args.test_mo,
        objective_name=args.objective,
        search_method=args.search,
        search_budget=args.budget,
        min_trades_per_fold=args.min_trades,
        seed=args.seed,
        initial_capital=args.capital,
        purge_days=args.purge_days,
    )

    m = result.stitched_metrics
    n_generated = result.n_folds_generated
    n_survived = len(result.folds)
    n_skipped = n_generated - n_survived
    n_failed = result.n_failed_folds

    # Inconclusive-run detection (fail-loud): zero evidence or partial/broken
    # runs must never exit 0 looking like a clean verdict.
    inconclusive: list[str] = []
    if n_survived == 0:
        inconclusive.append("zero folds survived — every fold was skipped/gated out")
    if m.n_trades == 0:
        inconclusive.append("stitched OOS contains 0 trades — zero evidence either way")
    if n_failed > 0:
        inconclusive.append(
            f"{n_failed}/{n_survived} folds FAILED (OOS blind test raised — lost evidence)"
        )
    if n_generated > 0 and n_skipped / n_generated > 0.30:
        inconclusive.append(
            f"{n_skipped}/{n_generated} generated folds were skipped (>30% attrition)"
        )
    status = "INCONCLUSIVE" if inconclusive else "OK"

    # Summary
    print("\n" + "=" * 60)
    print(f"WFA complete: {result.bot_name}")
    print(f"  Folds:         {n_survived} survived / {n_generated} generated"
          f"  ({n_skipped} skipped, {n_failed} failed)")
    print(f"  OOS trades:    {len(result.stitched_trades)}")
    print(f"  OOS return:    {m.total_return * 100:.2f}%")
    if m.sortino is not None:
        print(f"  OOS Sortino:   {m.sortino:.3f}")
    if m.calmar is not None:
        print(f"  OOS Calmar:    {m.calmar:.3f}")
    print(f"  OOS Max DD:    {m.max_drawdown_pct * 100:.2f}%")
    if result.degradation is not None:
        print(f"  Degradation:   {result.degradation * 100:.1f}%  (per-month basis, matched folds)")
    n_deg_na = result.n_degradation_unavailable
    if n_deg_na > 0:
        print(f"  Note:          {n_deg_na} fold(s) had an IS re-run error (OOS valid) — "
              f"degradation computed on the remaining matched folds only")
    print(f"  Status:        {status}")
    print("=" * 60)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "wfa_result.json"

        data: dict = {
            "bot_name": result.bot_name,
            "objective": result.objective,
            "search_method": result.search_method,
            "train_months": result.train_months,
            "test_months": result.test_months,
            "seed": result.seed,
            "n_folds": len(result.folds),
            "mean_is_return": result.mean_is_return,
            "degradation": result.degradation,
            # Additive fields (2026-07): run health + degradation basis + DSR inputs.
            "status": status,
            "inconclusive_reasons": inconclusive,
            "n_folds_generated": n_generated,
            "n_folds_skipped": n_skipped,
            "n_folds_failed": n_failed,
            "n_folds_degradation_unavailable": n_deg_na,
            "search_budget": args.budget,
            "min_trades_per_fold": args.min_trades,
            "n_trials_total": result.n_trials_total,
            "degradation_basis": "per_month_matched_folds",
            "mean_is_return_per_month": result.mean_is_return_per_month,
            "oos_return_per_month": result.oos_return_per_month,
            "summary": {
                "n_trades": m.n_trades,
                "total_return": m.total_return,
                "sharpe": m.sharpe,
                "sortino": m.sortino,
                "calmar": m.calmar,
                "max_drawdown_pct": m.max_drawdown_pct,
                "win_rate": m.win_rate,
                "profit_factor": m.profit_factor,
            },
            "folds": [
                {
                    "fold": fr.fold.index,
                    "train_start": str(fr.fold.train_start.date()),
                    "train_end": str(fr.fold.train_end.date()),
                    "test_start": str(fr.fold.test_start.date()),
                    "test_end": str(fr.fold.test_end.date()),
                    "best_params": fr.best_params,
                    "is_objective": fr.is_objective,
                    "is_return": fr.is_return,
                    "oos_n_trades": fr.oos_n_trades,
                    "n_trials": fr.n_trials,
                    "is_error": fr.is_error,
                    "oos_error": fr.oos_error,
                }
                for fr in result.folds
            ],
        }

        out_path.write_text(
            json.dumps(_json_safe(data), indent=2, default=str, allow_nan=False)
        )
        logger.info("Saved to %s", out_path)

    # Inconclusive guard: a run with zero stitched evidence, failed folds, or
    # mass fold-skips (most commonly the real-CVD coverage gate raising
    # DataQualityError per symbol) must NOT exit 0 looking like a clean verdict.
    if inconclusive:
        bar = "!" * 60
        lines = "".join(f"  - {r}\n" for r in inconclusive)
        remediation = ""
        if n_survived == 0 or n_skipped > 0:
            remediation = (
                f"Most likely cause: {result.bot_name}'s adapter data-quality gate "
                "rejected folds — check the adapter's backtest data coverage/date range.\n"
                "If your adapter enforces a data-coverage gate, refresh the underlying "
                "market data for the requested window before re-running."
                "scripts/merge_cvd_sources.py into data/ohlc/{BTC,ETH}_real_cvd.csv — "
                "then re-run.\n"
            )
        print(
            f"\n{bar}\n"
            "WFA INCONCLUSIVE — DO NOT USE THIS RUN AS A GO/NO-GO VERDICT\n"
            f"{lines}{remediation}{bar}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
