"""Central registry of all help tooltip copy for the WFA portal.

Edit copy here; widgets import by key. Keeps UI and content separate.
"""
from __future__ import annotations

HELP: dict[str, str] = {
    # ── sidebar — bot + windows ────────────────────────────────────────────────
    "bot": (
        "Which bot strategy and data to analyse. Each bot has its own param schema, "
        "recommended train/test windows, and data files. Adapters are registered in "
        "~/.wfa/config.toml."
    ),
    "train_months": (
        "Months of history used to optimise parameters per fold. "
        "Longer = more stable params but staler. Shorter = adapts faster but each fold "
        "has fewer trades and noisier metrics. "
        "Rule of thumb: must produce ≥20 trades for stable Sortino/Calmar."
    ),
    "test_months": (
        "Months of blind-test data per fold (the OOS window). "
        "Shorter = more folds (better aggregate confidence) but each fold is noisier. "
        "Longer = cleaner per-fold signal but fewer folds total."
    ),
    # ── sidebar — objective ────────────────────────────────────────────────────
    "objective_sortino": (
        "Default. Penalises downside volatility only — ignores upside vol. "
        "Best for strategies where you accept some up-vol but hate drawdowns. "
        "Higher = better risk-adjusted return on the downside dimension."
    ),
    "objective_calmar": (
        "Annualised return ÷ max drawdown. Directly rewards small max-DD. "
        "Best for 'will this survive real money?' — the only metric that "
        "explicitly prices drawdown in the denominator."
    ),
    "objective_sharpe": (
        "Classical risk-adjusted return. Punishes all volatility equally — "
        "penalises upside AND downside. Favours smooth equity curves. "
        "Can pick low-return strategies that happen to be smooth."
    ),
    "objective_total_return": (
        "Simple (final equity − initial) / initial. "
        "WARNING: ignores risk entirely — will pick the most aggressive params. "
        "Useful as a sanity check but dangerous as an optimiser target."
    ),
    "objective_win_rate": (
        "Fraction of winning trades. Rewards many small wins. "
        "WARNING: 90% WR with one large loss still loses money. "
        "Useful as a secondary diagnostic, not a primary optimiser target."
    ),
    "objective_profit_factor": (
        "Gross profit ÷ gross loss. PF > 1.5 is generally considered robust. "
        "Less timeframe-dependent than Sharpe. Tolerant of non-normal returns."
    ),
    "objective_psr": (
        "Sharpe adjusted for sample size, return skew, and kurtosis "
        "(Bailey & Lopez de Prado 2012). Returns P(true Sharpe > 0). "
        "Most honest metric — slower and harder to game than raw Sharpe. "
        "Best when sample is small (< 100 trades)."
    ),
    # ── sidebar — search method ────────────────────────────────────────────────
    "search_random": (
        "Random N trials without replacement. "
        "Better generalisation than full-grid in most cases (Bergstra & Bengio 2012) "
        "and faster. Default for all bots."
    ),
    "search_grid": (
        "Exhaustive Cartesian product — tries every param combination. "
        "Use when total combos < 50 (e.g. mybot with 3 params = 36 combos). "
        "Guaranteed coverage but slow for large spaces."
    ),
    "search_optuna": (
        "Bayesian TPE search (Tree-structured Parzen Estimator). "
        "Intelligently focuses trials on promising regions. "
        "Best for > 5 params or continuous spaces. Requires: pip install optuna."
    ),
    "search_budget": (
        "Number of param combinations to try per fold (random / optuna). "
        "Ignored for full-grid. "
        "For mybot (36 combos) random N=36 gives full coverage. "
        "Higher budget improves IS optimum quality but takes longer."
    ),
    # ── sidebar — guards ───────────────────────────────────────────────────────
    "min_trades": (
        "Minimum closed trades required per IS fold window for a param combo to be valid. "
        "Without this guard, tight params fire 0 trades and 'win' by NaN→∞ Sharpe. "
        "Default 10. Set higher (20+) if your strategy fires infrequently."
    ),
    # ── sidebar — Monte Carlo ──────────────────────────────────────────────────
    "mc_sims": (
        "Number of resampled equity paths. 10 000 is sufficient for stable percentiles. "
        "50 000 for tight CI on P(ruin) — takes ~3× longer."
    ),
    "mc_method_reshuffle": (
        "Permutes the same trades in random order — same edge, different sequence. "
        "Tests path risk ONLY (sequence of wins/losses). "
        "All paths have identical total return (additive model). "
        "Use this first."
    ),
    "mc_method_bootstrap": (
        "Resamples trades with replacement — draws a new 'sample' of N trades each path. "
        "Tests BOTH sequence risk AND sample uncertainty (wider tails). "
        "Use after reshuffle to see how sensitive the edge is to which trades happened."
    ),
    "ruin_threshold": (
        "Max drawdown % that counts as 'ruin'. "
        "Matches our internal threshold (BTC_ETH_TRADING_KNOWLEDGE_BASE.md). "
        "Default 20%. A sim 'hits ruin' if its equity ever falls ≥ threshold from its peak."
    ),
    # ── sidebar — misc ─────────────────────────────────────────────────────────
    "seed": (
        "Random seed for param search and MC sampling. "
        "Same seed + same data = identical results every run. "
        "Change to verify results aren't seed-dependent (try 42, 123, 999)."
    ),
    "initial_capital": (
        "Starting equity for metrics calculation. "
        "Does not affect which params are selected (objective is unitless). "
        "Affects displayed $ values and P(ruin) thresholds."
    ),
    # ── metric explanations ────────────────────────────────────────────────────
    "degradation": (
        "Return Degradation = 1 − (OOS return / mean IS return). "
        "How much of the IS gains did NOT transfer to the blind test. "
        "75% = only 25% of the edge was real; 75% was curve-fit to the training data. "
        "Negative = OOS outperformed IS (lucky OOS sample or genuine robustness)."
    ),
    "stitched_oos": (
        "All OOS (blind-test) trades stitched together chronologically across folds. "
        "This is the only curve shown as the headline — IS metrics are labelled as such."
    ),
}
