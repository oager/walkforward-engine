"""Strategy-survival gate — absolute OOS acceptance funnel + breadth + trial-deflation.

OPT-IN. Import and call `evaluate_survival()` on stitched OOS trades (or
`evaluate_wfa_result()` on a WFAResult). Changes no existing WFA behavior.

Inspired by AI Pathways' 9k-strategy funnel
(sources/claude-9000-trading-strategies-ai-pathways.md), but deliberately uses our
SCALE-FREE metrics instead of the video's raw thresholds: our Sharpe/Sortino are
per-trade-annualised (`sqrt(trades/yr)`), NOT comparable to the video's per-bar
daily Sharpe, so porting "Sharpe > 0.5" would be a metric-scale error
(see wfa_param_tiebreaker). Instead:
  - "edge is real"  -> PSR  (P(true Sharpe > 0); scale-free, skew/kurtosis-aware)
  - "not overfit"   -> return degradation band (OOS vs IS)
  - "survivable"    -> max drawdown % + Monte-Carlo P(ruin)
  - "meaningful"    -> minimum trade count
  - "holds broadly" -> per-symbol breadth (no single symbol carrying the pool)

Thresholds are all configurable; defaults are conservative starting points. Calibrate
against a known-good baseline before treating the verdict as a hard gate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from wfa.metrics import build_equity_curve, max_drawdown
from wfa.objectives import _psr, _sharpe

_EULER = 0.5772156649015329
# Candidate column names that identify which asset a trade belongs to.
SYMBOL_COLS = ("symbol", "symbol_hl", "coin", "pair", "asset", "ticker")


# ── config ──────────────────────────────────────────────────────────────────────

@dataclass
class SurvivalThresholds:
    """Absolute OOS acceptance thresholds. All scale-free or %-based."""
    min_trades: int = 30                 # statistical viability
    max_drawdown_pct: float = 0.35       # OOS max DD fraction (0-1)
    min_psr: float = 0.95                # P(true Sharpe > 0) on stitched OOS
    max_degradation: float = 0.50        # OOS kept >= 50% of IS return (overfit gap)
    min_degradation: float = -1.00       # OOS not absurdly > IS (lucky-split / "too good")
    require_positive_return: bool = True
    run_montecarlo: bool = True
    mc_max_p_ruin: float = 0.05          # bootstrap P(ruin) ceiling
    mc_ruin_threshold: float = 0.20      # DD fraction that counts as ruin
    mc_n_sims: int = 10_000
    mc_min_trades: int = 10              # MC on fewer trades is meaningless
    breadth_min_psr: float = 0.50        # per-symbol floor (looser than pooled)
    breadth_min_trades: int = 10         # symbols with fewer trades are EXCLUDED from the breadth
                                         # denominator (reported as insufficient-data skips) — a
                                         # single incidental trade must not hard-fail the verdict
    breadth_min_pass: int | None = None  # None = ALL symbols must clear the floor (pooled book).
                                         # K = at least K must. A "situational" edge that fails breadth
                                         # is NOT discarded — the per-symbol map reports which assets it's
                                         # valid on (edges are asset/regime-specific; a BTC/ETH miss != dead).


@dataclass
class FilterResult:
    name: str
    passed: bool | None      # None = skipped (insufficient data, not a fail)
    value: float | None
    threshold: str
    note: str = ""

    def __post_init__(self):
        # Comparisons on numpy floats yield np.bool_, for which `x is False` is False —
        # silently hiding a real failure from callers that test identity. Coerce to a
        # Python bool (leave None as-is) so `is True/False` and truthiness both behave.
        if self.passed is not None:
            self.passed = bool(self.passed)


@dataclass
class SurvivalVerdict:
    passed: bool
    filters: list[FilterResult]
    psr: float | None
    n_trades: int
    per_symbol: dict | None = field(default=None)

    def summary(self) -> str:
        lines = [f"SURVIVAL: {'PASS' if self.passed else 'FAIL'}  (n_trades={self.n_trades})"]
        for f in self.filters:
            mark = "skip" if f.passed is None else ("pass" if f.passed else "FAIL")
            val = "n/a" if f.value is None else f"{f.value:.4f}"
            lines.append(f"  [{mark}] {f.name:<16} {val:>10}  ({f.threshold})"
                         + (f"  — {f.note}" if f.note else ""))
        if self.per_symbol:
            lines.append("  per-symbol breadth:")
            for sym, d in self.per_symbol.items():
                mark = "skip" if d["passed"] is None else ("pass" if d["passed"] else "FAIL")
                psr_s = "n/a" if d.get("psr") is None else f"{d['psr']:.3f}"
                lines.append(f"    [{mark}] {sym:<10} n={d['n_trades']:<4} "
                             f"PSR={psr_s} ret={d['total_return']:+.4f} maxDD={d['max_dd']:.3f}"
                             + (f"  — {d['note']}" if d.get("note") else ""))
        return "\n".join(lines)


# ── core ──────────────────────────────────────────────────────────────────────

def evaluate_survival(
    trades: pd.DataFrame,
    *,
    degradation: float | None = None,
    thresholds: SurvivalThresholds | None = None,
    initial_capital: float = 10_000.0,
) -> SurvivalVerdict:
    """Apply the absolute-OOS survival funnel to a stitched OOS trade stream.

    `degradation` (1 - OOS_ret/IS_ret) gates the overfit filter; pass None when there
    is no IS comparison (e.g. a fixed-param full-history run) and that filter is skipped.
    """
    t = thresholds or SurvivalThresholds()
    filters: list[FilterResult] = []

    if trades is None or trades.empty or "pnl" not in trades.columns:
        return SurvivalVerdict(passed=False, filters=[
            FilterResult("data", False, None, "non-empty trades with 'pnl'", "no usable trades")
        ], psr=None, n_trades=0)

    n = len(trades)
    equity = build_equity_curve(trades, initial_capital)
    total_return = (equity[-1] - equity[0]) / equity[0] if equity[0] > 0 else float("-inf")
    dd = max_drawdown(equity)
    psr = _psr(trades, equity)
    psr = psr if math.isfinite(psr) else None

    # 1. minimum trades
    filters.append(FilterResult(
        "min_trades", n >= t.min_trades, float(n), f">= {t.min_trades}"))
    # 2. max drawdown
    filters.append(FilterResult(
        "max_drawdown", dd <= t.max_drawdown_pct, dd, f"<= {t.max_drawdown_pct:.0%}"))
    # 3. PSR — "edge is real" (scale-free)
    filters.append(FilterResult(
        "psr", (psr is not None and psr >= t.min_psr), psr, f">= {t.min_psr}",
        "" if psr is not None else "PSR not computable (n<4 or zero-vol)"))
    # 4. positive return
    if t.require_positive_return:
        filters.append(FilterResult(
            "positive_return", total_return > 0, total_return, "> 0"))
    # 5. overfit / degradation band (only when IS comparison exists)
    if degradation is None:
        filters.append(FilterResult(
            "degradation", None, None, f"[{t.min_degradation}, {t.max_degradation}]",
            "skipped — no IS baseline (fixed-param run)"))
    else:
        ok = t.min_degradation <= degradation <= t.max_degradation
        filters.append(FilterResult(
            "degradation", ok, degradation, f"in [{t.min_degradation}, {t.max_degradation}]",
            "OOS lost too much of IS" if degradation > t.max_degradation
            else ("OOS >> IS (lucky split?)" if degradation < t.min_degradation else "")))
    # 6. Monte-Carlo P(ruin)
    if t.run_montecarlo and n >= t.mc_min_trades:
        try:
            from wfa.montecarlo import run_mc
            mc = run_mc(trades, initial_capital=initial_capital, n_sims=t.mc_n_sims,
                        method="bootstrap", ruin_threshold=t.mc_ruin_threshold)
            filters.append(FilterResult(
                "mc_p_ruin", mc.probability_of_ruin <= t.mc_max_p_ruin,
                mc.probability_of_ruin, f"<= {t.mc_max_p_ruin}",
                f"bootstrap {t.mc_n_sims} sims, ruin@{t.mc_ruin_threshold:.0%} DD; "
                f"p95 maxDD={mc.max_drawdown_pct.p95:.1f}%"))
        except Exception as exc:
            # Fail LOUD: an MC crash is not "insufficient data" — a verdict must never
            # PASS with its ruin gate silently unevaluated (swallowed-exception class).
            filters.append(FilterResult("mc_p_ruin", False, None,
                                        f"<= {t.mc_max_p_ruin}",
                                        f"MC ERROR — gate not evaluated, failing loud: {exc}"))
    else:
        filters.append(FilterResult("mc_p_ruin", None, None, f"<= {t.mc_max_p_ruin}",
                                    f"skipped — need >= {t.mc_min_trades} trades"))

    per_symbol = per_symbol_breakdown(trades, t, initial_capital)
    if per_symbol:
        # Symbols below the min-trade floor are insufficient-data skips (passed=None),
        # excluded from the breadth denominator — not scored PSR=0.0.
        eligible = {s: d for s, d in per_symbol.items() if d["passed"] is not None}
        skipped = [s for s, d in per_symbol.items() if d["passed"] is None]
        n_sym = len(eligible)
        if n_sym == 0:
            filters.append(FilterResult(
                "breadth", None, None, f"PSR >= {t.breadth_min_psr} per symbol",
                f"skipped — no symbol has >= {t.breadth_min_trades} trades ({skipped})"))
        else:
            n_pass = sum(1 for d in eligible.values() if d["passed"])
            required = t.breadth_min_pass if t.breadth_min_pass is not None else n_sym
            ok = n_pass >= required
            winners = [s for s, d in eligible.items() if d["passed"]]
            note = f"{n_pass}/{n_sym} symbols clear PSR>={t.breadth_min_psr}"
            if skipped:
                note += f"; insufficient data (n<{t.breadth_min_trades}): {skipped}"
            if not ok and winners:
                # The video's core lesson: a non-broad edge is SITUATIONAL, not dead.
                note += f" — NOT broadly robust, but valid on {winners}: deploy per-asset, don't discard"
            elif not ok:
                note += " — edge holds on no single symbol"
            filters.append(FilterResult(
                "breadth", ok, float(n_pass), f">= {required} of {n_sym} symbols", note))

    passed = all(f.passed for f in filters if f.passed is not None)
    return SurvivalVerdict(passed=passed, filters=filters, psr=psr,
                           n_trades=n, per_symbol=per_symbol)


def evaluate_wfa_result(result, thresholds: SurvivalThresholds | None = None,
                        initial_capital: float = 10_000.0) -> SurvivalVerdict:
    """Convenience wrapper: pull stitched trades + degradation off a WFAResult."""
    return evaluate_survival(
        result.stitched_trades,
        degradation=result.degradation,
        thresholds=thresholds,
        initial_capital=initial_capital,
    )


def per_symbol_breakdown(trades: pd.DataFrame, thresholds: SurvivalThresholds,
                         initial_capital: float = 10_000.0) -> dict | None:
    """PSR / return / max-DD per traded symbol. Returns None if no symbol column.

    Guards the pooled-result trap: a pooled BTC+ETH edge can pass while one symbol
    carries it. Each symbol must independently clear `breadth_min_psr`.
    """
    col = next((c for c in SYMBOL_COLS if c in trades.columns), None)
    if col is None:
        return None
    out: dict = {}
    for sym, grp in trades.groupby(col):
        if grp.empty:
            continue
        eq = build_equity_curve(grp, initial_capital)
        ret = (eq[-1] - eq[0]) / eq[0] if eq[0] > 0 else float("-inf")
        psr = _psr(grp, eq)
        psr_val = psr if math.isfinite(psr) else None
        if len(grp) < thresholds.breadth_min_trades:
            # Insufficient data = skipped (passed=None), matching FilterResult convention —
            # PSR on a handful of trades is a coin flip, not a verdict on the symbol.
            passed = None
            note = f"insufficient data (n<{thresholds.breadth_min_trades}) — excluded from breadth"
        else:
            passed = (psr_val if psr_val is not None else 0.0) >= thresholds.breadth_min_psr
            note = "" if psr_val is not None else "PSR not computable (zero-vol)"
        out[str(sym)] = {
            "n_trades": len(grp),
            "psr": psr_val,
            "total_return": ret,
            "max_dd": max_drawdown(eq),
            "passed": passed,
            "note": note,
        }
    return out or None


# ── trial-deflated Sharpe across candidate param sets ───────────────────────────

def deflated_sharpe_from_trade_sets(
    trade_sets: dict[str, pd.DataFrame],
    best: str,
    trials: int | None = None,
    cap: float = 10_000.0,
) -> dict | None:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) for the best of several
    candidate param sets. Same math as scripts/drivers/regime_overfit_check.py but
    operating on a {label: trades_df} dict instead of the regime-sweep JSONs, and with
    no scipy dependency (uses NormalDist + pandas skew/kurtosis like objectives._psr).

    `trials` = the TRUE search budget (e.g. WFA search_budget x folds), since the few
    finalists in `trade_sets` understate the real number of configs scanned -> passing
    only the finalist count makes the DSR too lenient. We never go below the observed count.
    Returns None if <2 sets have returns or cross-trial Sharpe variance is degenerate.
    """
    sr: dict[str, float] = {}
    for label, t in trade_sets.items():
        if t is None or t.empty or "pnl" not in t.columns:
            continue
        r = t["pnl"].astype(float).to_numpy() / cap
        if len(r) <= 2:
            continue
        sd = r.std(ddof=1)
        sr[label] = float(r.mean() / sd) if sd > 0 else 0.0
    if best not in sr or len(sr) < 2:
        return None
    n_trials = max(trials, len(sr)) if trials else len(sr)
    var_sr = float(np.var(list(sr.values()), ddof=1)) if n_trials > 1 else 0.0
    if var_sr <= 0:
        return None
    nd = NormalDist()
    z1 = nd.inv_cdf(1 - 1.0 / n_trials)
    z2 = nd.inv_cdf(1 - 1.0 / (n_trials * math.e))
    sr_star = math.sqrt(var_sr) * ((1 - _EULER) * z1 + _EULER * z2)

    r = trade_sets[best]["pnl"].astype(float).to_numpy() / cap
    n = len(r)
    sd = r.std(ddof=1)
    sr_hat = float(r.mean() / sd) if sd > 0 else 0.0
    s = pd.Series(r)
    g3 = float(s.skew()) if n >= 4 else 0.0
    g4 = float(s.kurtosis()) + 3.0 if n >= 4 else 3.0   # pandas gives EXCESS kurtosis; DSR wants non-excess
    denom = math.sqrt(max(1 - g3 * sr_hat + (g4 - 1) / 4.0 * sr_hat ** 2, 1e-9))
    dsr = float(nd.cdf((sr_hat - sr_star) * math.sqrt(n - 1) / denom))
    out = {"best": best, "trials": n_trials, "sr_hat": round(sr_hat, 4),
           "sr_star": round(sr_star, 4), "skew": round(g3, 3), "kurt": round(g4, 3),
           "n_trades": n, "DSR": round(dsr, 4)}
    if not trials:
        # Fail-loud marker: defaulting to the finalist count UNDER-deflates the DSR
        # (the true search budget is almost always far larger).
        out["trials_note"] = (f"trials defaulted to finalist count ({n_trials}) — "
                              "DSR is UNDER-deflated; pass the true search budget")
    return out
