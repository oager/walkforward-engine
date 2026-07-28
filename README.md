# WFA Engine

[![tests](https://github.com/oager/walkforward-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/oager/walkforward-engine/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A **bot-agnostic walk-forward analysis engine** for trading strategies. Any strategy that
implements the `BacktestAdapter` contract gets the same overfit-resistant validation: rolling
walk-forward optimization, an absolute out-of-sample survival gate, Monte Carlo path risk, and
selection-overfit backstops (PBO/CSCV and Deflated Sharpe).

The engine never imports a strategy directly. Strategies plug in through an adapter, which keeps
validation honest and lets one engine grade very different books.

> **Metric-scale gotcha, read this first.** All Sharpe/Sortino/Calmar here are
> **per-trade-annualised** (`× sqrt(trades/year)`), computed on a per-trade-close equity curve.
> They are **not comparable** to a per-bar baseline Sharpe, nor to an external study's per-bar
> daily Sharpe. Never cross-compare across scales. That is why the survival gate keys on
> scale-free **PSR** (probability true Sharpe > 0) rather than a raw Sharpe threshold.

---

## Scope

**This engine validates strategies. It does not implement them.**

There are no signals, indicators, or parameter sets in this repository, and no broker or exchange
integrations. A strategy plugs in through an adapter that lives in its own repo, and the engine
never imports it directly. That boundary is what keeps the validation credible: the thing being
graded cannot reach into the grader.

See [CONTRIBUTING.md](CONTRIBUTING.md) for what belongs here and what does not.

---

## Why this exists

A backtest that tunes and scores on the same data will always look good. This engine is built
around the assumption that most apparent edges are selection artifacts, so every layer exists to
try to kill a result before it reaches production:

- Tune on in-sample, score on **blind** out-of-sample, stitch the OOS tails, report OOS only.
- Gate on scale-free statistics, so results stay comparable across strategies and timeframes.
- Deflate for the number of trials actually run, because searching harder inflates the best Sharpe.
- Simulate path risk, because a positive expectancy that can ruin you first is not tradeable.

## Architecture (`wfa/`)

| Module | Role |
|---|---|
| `adapter.py` | The `BacktestAdapter` Protocol, the contract every strategy implements |
| `registry.py` | Loads adapters dynamically from `~/.wfa/config.toml` |
| `folds.py` | `generate_folds()` produces rolling walk-forward train/test `Fold` windows |
| `search.py` | In-sample parameter search: `RandomSearch` (default), `FullGrid`, `OptunaSearch`, via a `make_search()` factory |
| `objectives.py` | Seven objectives (`sortino` default, `calmar`, `sharpe`, `total_return`, `win_rate`, `profit_factor`, `psr`) behind `compute_objective()` with a min-trades gate |
| `runner.py` | `run_wfa()` orchestrator, drives the loop, stitches OOS, computes degradation into a `WFAResult` |
| `metrics.py` | `compute_oos_metrics()` over stitched OOS trades, plus `max_drawdown` and `return_degradation` |
| `montecarlo.py` | `run_mc()` path simulation (reshuffle or bootstrap) producing `MCResult`, including probability of ruin |
| `survival.py` | The absolute OOS survival gate: multi-filter funnel, per-symbol breadth classifier, trial-deflated Sharpe |

### Data flow

```
registry.load_adapter(bot) -> BacktestAdapter
  -> folds.generate_folds(data_range, train_mo, test_mo) -> [Fold]
    for each Fold:
      search.make_search(method, schema, budget) -> param sets
        adapter.run(params, train_start, train_end)   -> IS trades
        objectives.compute_objective(...)             -> score (best IS params kept)
      adapter.run(best_params, test_start, test_end)  -> OOS trades (blind)
  runner: stitch all OOS trades chronologically
  metrics.compute_oos_metrics(stitched) + return_degradation -> WFAResult
  -> (optional) montecarlo.run_mc(stitched)            -> MCResult
  -> (optional) survival.evaluate_wfa_result(result)   -> SurvivalVerdict
  -> portal/app.py (Streamlit) | scripts/run_wfa.py -> runs/<name>/wfa_result.json
```

---

## The adapter contract

Implement a class named `Adapter` satisfying `wfa/adapter.py`'s `BacktestAdapter` Protocol in your
strategy's own repo, then register it in `~/.wfa/config.toml`.

| Member | Returns | Notes |
|---|---|---|
| `bot_name` | `str` | e.g. `"mybot"` |
| `timeframe` | `str` | e.g. `"4h"` |
| `param_schema()` | `dict[str, ParamSpec]` | `{param: {type, choices, default, help}}`, where `choices` is the sweep grid |
| `recommended_windows()` | `(train_months, test_months)` | per-strategy default windows |
| `default_objective()` | `str` | e.g. `"sortino"` |
| `data_snapshot_path()` | `Path` | **frozen** OHLC data, never a live API |
| `data_range()` | `(first_bar_utc, last_bar_utc)` | |
| `run(params, start, end)` | `pd.DataFrame` | trades over `[start, end)` |

**Required trades schema** from `run()`: `trade_id, symbol, side, entry_time, entry_price,
exit_time, exit_price, size, pnl, fees, status`. `symbol` enables the per-symbol breadth check in
`survival.py`; `pnl` drives all metrics and Monte Carlo.

Requiring a frozen data snapshot rather than a live API is deliberate. A validation run that can
silently pull different data on re-run is not a validation run.

---

## Running it

**Headless CLI**

```bash
python scripts/run_wfa.py --bot <name> --objective sortino --seed 42 --out runs/my_run/
```

Options: `--objective` (sortino/calmar/sharpe/total_return/win_rate/profit_factor/psr) ·
`--search` (random/grid/optuna) · `--budget` (trials per fold, default 100) · `--train-mo` /
`--test-mo` · `--min-trades` (IS gate, default 10) · `--seed` (default 42) · `--capital`
(default 10000) · `--out` (writes `wfa_result.json`).

Exits non-zero with an explicit message if zero folds were produced, rather than reporting an
empty success.

**Portal**

```bash
streamlit run portal/app.py
```

Tabs: Setup (param schema) · WFA Run (run plus per-fold table) · MC Run (Monte Carlo over the
stitched trades) · Compare (multi-strategy Sortino chart). Tooltips live in `portal/help_text.py`.

**Rolling tripwire**

`scripts/rolling_psr_tripwire.py` re-checks live strategies' rolling PSR on a schedule and alerts
when an edge decays. Systemd unit and timer are in `scripts/systemd/`. Telegram credentials are
read from the environment (`TELEGRAM_BOT_TOKEN`, `HEALTH_TG_CHAT`) and scrubbed from error output.

---

## Validation layers

1. **Walk-forward** (`runner` + `folds`) — tune on IS, score on blind OOS, stitch OOS tails. The
   headline metric is OOS-only. `return_degradation` = `1 − OOS/IS` flags curve-fitting.

2. **Survival gate** (`survival.py`) — an absolute OOS acceptance funnel on scale-free metrics:
   `min_trades` · `max_drawdown_pct` · **PSR ≥ 0.95** · positive return · a degradation band ·
   Monte Carlo probability of ruin. Plus a **per-symbol breadth classifier**:
   `breadth_min_pass=None` means every symbol must clear the floor (pooled book); `=K` means at
   least K must. A non-broad edge is reported as *situational* ("valid on X, deploy per-asset")
   rather than discarded. Entry points: `evaluate_survival(trades, …)` and
   `evaluate_wfa_result(result, …)`. Thresholds live in `SurvivalThresholds` and are configurable.

3. **Trial-deflated Sharpe** — `survival.deflated_sharpe_from_trade_sets()` across candidate
   parameter sets, with no scipy dependency. Pass the true search budget, or deflation is too
   lenient and you are back to fooling yourself.

4. **Monte Carlo** (`montecarlo.py`) — reshuffle (sequence risk) or bootstrap (sequence plus
   sample risk), 10k paths by default, yielding probability of ruin and a worst-case drawdown
   distribution.

---

## Tests

```bash
python -m pytest -q
```

**226 passed, 1 skipped** (the skip is optuna, which is import-gated). Coverage includes folds,
metrics, Monte Carlo, objectives, search, the full `run_wfa` loop against a synthetic adapter,
portal logic, tripwire alerting and secret scrubbing, and a CI guard that fails the build if
acausal `filtfilt` (lookahead) is ever introduced.

## Layout

```
wfa/                 core package
scripts/             run_wfa.py (CLI), rolling_psr_tripwire.py, systemd units
portal/              Streamlit app (app.py) and help_text.py
tests/               pytest suite
~/.wfa/config.toml   adapter registry
```

## Requirements

Python 3.10+, numpy, pandas, streamlit (portal only), optuna (optional, for `--search optuna`).

## Contributing

Issues and pull requests are welcome, particularly on validation methodology and statistical
correctness. See [CONTRIBUTING.md](CONTRIBUTING.md) for scope and setup.

If you use this in research or production, I would genuinely like to hear about it — open a
discussion and tell me what you pointed it at.

## License

MIT. See [LICENSE](LICENSE).
