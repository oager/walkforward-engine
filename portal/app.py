"""WFA + Monte Carlo Streamlit portal.

Launch: streamlit run portal/app.py --server.address localhost

The portal is unauthenticated and refuses to serve on a non-loopback address
unless WFA_PORTAL_ALLOW_REMOTE=1 is set explicitly.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import math

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from portal.help_text import HELP
from wfa.metrics import OOSMetrics
from wfa.montecarlo import MCResult, run_mc
from wfa.registry import list_bots, load_adapter
from wfa.runner import WFAResult, run_wfa

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WFA Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state keys ────────────────────────────────────────────────────────

_WFA_KEY = "wfa_result"
_MC_KEY = "mc_result"
_BOT_KEY = "last_wfa_bot"
_MC_BOT_KEY = "last_mc_bot"
_CAPITAL_KEY = "last_wfa_capital"
_COMPARE_KEY = "compare_results"

_ALLOW_REMOTE_ENV = "WFA_PORTAL_ALLOW_REMOTE"
_LOOPBACK_ADDRESSES = ("localhost", "127.0.0.1", "::1")


# ── pure state/config logic (tested in tests/test_portal_logic.py) ────────────

def _binding_allowed(address: str | None, allow_remote: str | None) -> bool:
    """True when the portal may serve on *address*.

    The portal is unauthenticated and Run WFA is CPU-heavy, so it must bind
    loopback-only. Streamlit's default (address unset → None) binds ALL
    interfaces, so unset is refused. Set WFA_PORTAL_ALLOW_REMOTE=1 to accept
    remote exposure explicitly.
    """
    if allow_remote:
        return True
    return address in _LOOPBACK_ADDRESSES


def _stored_wfa_result(state: Mapping, bot_name: str) -> WFAResult | None:
    """The stored WFA result, or None unless it was produced for *bot_name*.

    A result from one bot must never render under another selection — _BOT_KEY
    records which sidebar bot produced _WFA_KEY and is checked here. A missing
    _BOT_KEY counts as a mismatch (fail-safe: never show possibly-wrong data).
    """
    result = state.get(_WFA_KEY)
    if result is None or state.get(_BOT_KEY) != bot_name:
        return None
    return result


def _stale_result_bot(state: Mapping) -> str | None:
    """Which bot the stored WFA result belongs to (for 'no result for X' text).

    None when there is no stored result at all.
    """
    result = state.get(_WFA_KEY)
    if result is None:
        return None
    return state.get(_BOT_KEY) or getattr(result, "bot_name", None) or "unknown"


def _stored_mc_result(state: Mapping, bot_name: str) -> MCResult | None:
    """The stored MC result, or None unless it was produced for *bot_name*."""
    mc = state.get(_MC_KEY)
    if mc is None or state.get(_MC_BOT_KEY) != bot_name:
        return None
    return mc


def _run_capital(state: Mapping, fallback: float) -> float:
    """Capital the displayed WFA run was computed with — NOT the live slider."""
    cap = state.get(_CAPITAL_KEY)
    return float(cap) if cap is not None else float(fallback)


def _adapter_failure_message(bot_name: str, adapter_error: str | None) -> str | None:
    """Loud message when the adapter FAILED to load (vs merely not configured)."""
    if not adapter_error:
        return None
    return (
        f"Adapter '{bot_name}' FAILED to load — this is a broken adapter, "
        f"not a missing one: {adapter_error}"
    )


# ── sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar controls and return the current settings dict."""
    st.sidebar.title("WFA Engine")

    # Bot selector
    try:
        bots = list_bots()
    except Exception as e:
        st.sidebar.error(f"Bot registry FAILED to load: {type(e).__name__}: {e}")
        bots = []
    if not bots:
        bots = ["(no adapters configured)"]

    bot_name = st.sidebar.selectbox(
        "Bot",
        bots,
        help=HELP["bot"],
    )

    # Load adapter for defaults
    adapter = None
    adapter_error: str | None = None
    if bot_name and not bot_name.startswith("("):
        try:
            adapter = load_adapter(bot_name)
        except Exception as e:
            # A raising adapter (e.g. DataQualityError) must stay LOUD — it is
            # a broken adapter, not a missing one.
            adapter_error = f"{type(e).__name__}: {e}"
            st.sidebar.error(_adapter_failure_message(bot_name, adapter_error))

    rec_train, rec_test = (18, 6) if adapter is None else adapter.recommended_windows()
    default_obj = "sortino" if adapter is None else adapter.default_objective()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Walk-Forward Windows")

    train_months = st.sidebar.slider(
        "Train months",
        min_value=6,
        max_value=36,
        value=rec_train,
        step=1,
        help=HELP["train_months"],
    )
    test_months = st.sidebar.slider(
        "Test months",
        min_value=1,
        max_value=12,
        value=rec_test,
        step=1,
        help=HELP["test_months"],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Objective")

    obj_options = [
        ("Sortino", "sortino", HELP["objective_sortino"]),
        ("Calmar", "calmar", HELP["objective_calmar"]),
        ("Sharpe", "sharpe", HELP["objective_sharpe"]),
        ("Total Return", "total_return", HELP["objective_total_return"]),
        ("Win Rate", "win_rate", HELP["objective_win_rate"]),
        ("Profit Factor", "profit_factor", HELP["objective_profit_factor"]),
        ("PSR", "psr", HELP["objective_psr"]),
    ]
    obj_labels = [o[0] for o in obj_options]
    obj_values = [o[1] for o in obj_options]
    obj_helps = {o[1]: o[2] for o in obj_options}

    default_obj_idx = obj_values.index(default_obj) if default_obj in obj_values else 0
    obj_label = st.sidebar.selectbox(
        "Objective",
        obj_labels,
        index=default_obj_idx,
    )
    objective = obj_values[obj_labels.index(obj_label)]
    st.sidebar.caption(obj_helps[objective])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Parameter Search")

    search_method = st.sidebar.selectbox(
        "Search method",
        ["random", "grid", "optuna"],
        help=(
            HELP["search_random"] + " | " +
            HELP["search_grid"] + " | " +
            HELP["search_optuna"]
        ),
    )

    search_budget = st.sidebar.number_input(
        "Search budget (trials/fold)",
        min_value=20,
        max_value=500,
        value=100,
        step=10,
        help=HELP["search_budget"],
    )

    min_trades = st.sidebar.number_input(
        "Min trades per fold",
        min_value=5,
        max_value=100,
        value=10,
        step=5,
        help=HELP["min_trades"],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Monte Carlo")

    mc_sims = st.sidebar.select_slider(
        "MC simulations",
        options=[1_000, 5_000, 10_000, 25_000, 50_000],
        value=10_000,
        help=HELP["mc_sims"],
    )

    mc_method = st.sidebar.radio(
        "MC method",
        ["reshuffle", "bootstrap"],
        captions=[
            HELP["mc_method_reshuffle"][:80] + "...",
            HELP["mc_method_bootstrap"][:80] + "...",
        ],
    )

    ruin_threshold = st.sidebar.slider(
        "Ruin threshold %",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help=HELP["ruin_threshold"],
    ) / 100.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("Misc")

    seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=99_999,
        value=42,
        step=1,
        help=HELP["seed"],
    )

    initial_capital = st.sidebar.number_input(
        "Initial capital ($)",
        min_value=1_000,
        max_value=1_000_000,
        value=10_000,
        step=1_000,
        help=HELP["initial_capital"],
    )

    return {
        "bot_name": bot_name,
        "adapter": adapter,
        "adapter_error": adapter_error,
        "train_months": int(train_months),
        "test_months": int(test_months),
        "objective": objective,
        "search_method": search_method,
        "search_budget": int(search_budget),
        "min_trades": int(min_trades),
        "mc_sims": int(mc_sims),
        "mc_method": mc_method,
        "ruin_threshold": ruin_threshold,
        "seed": int(seed),
        "initial_capital": float(initial_capital),
    }


# ── setup tab ─────────────────────────────────────────────────────────────────

def render_setup_tab(cfg: dict) -> None:
    st.header("Bot Setup")

    adapter = cfg["adapter"]
    if adapter is None:
        err_msg = _adapter_failure_message(cfg["bot_name"], cfg.get("adapter_error"))
        if err_msg:
            st.error(err_msg, icon="🚨")
        else:
            st.warning("No adapter loaded. Check ~/.wfa/config.toml.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Bot", adapter.bot_name)
    col2.metric("Timeframe", getattr(adapter, "timeframe", "—"))
    col3.metric("Default objective", adapter.default_objective())

    try:
        data_start, data_end = adapter.data_range()
        import math
        total_months = (data_end.year - data_start.year) * 12 + (data_end.month - data_start.month)
        window = cfg["train_months"] + cfg["test_months"]
        n_folds = max(0, total_months - cfg["train_months"]) // cfg["test_months"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Data range", f"{data_start.date()} → {data_end.date()}")
        col2.metric("Total months", total_months)
        col3.metric("Expected folds", n_folds)

        if n_folds == 0:
            st.error(
                f"Window ({cfg['train_months']} + {cfg['test_months']} = {window} mo) "
                f"exceeds data range ({total_months} mo). Reduce windows or load more data."
            )
    except Exception as e:
        st.warning(f"Could not load data range: {e}")

    st.subheader("Parameter Schema")

    schema = adapter.param_schema()
    for pname, spec in schema.items():
        with st.expander(f"**{pname}** — {spec.get('type', '?')}"):
            st.markdown(spec.get("help", ""))
            col1, col2 = st.columns(2)
            col1.markdown(f"**Choices:** `{spec.get('choices', [])}`")
            col2.markdown(f"**Production default:** `{spec.get('default', '—')}`")

    total_combos = 1
    for spec in schema.values():
        total_combos *= len(spec.get("choices", [1]))
    budget = cfg['search_budget']
    coverage = "covers full grid" if budget >= total_combos else f"samples {budget}/{total_combos}"
    st.caption(f"Grid = {total_combos} combos. Random N={budget} {coverage}.")


# ── helpers ───────────────────────────────────────────────────────────────────

def _metrics_cards(metrics: OOSMetrics, label: str = "Stitched OOS") -> None:
    st.subheader(f"{label} Metrics")
    cols = st.columns(8)
    cols[0].metric("Trades", metrics.n_trades)
    cols[1].metric("Total Return", f"{metrics.total_return * 100:.1f}%")
    cols[2].metric("Sharpe", f"{metrics.sharpe:.2f}" if math.isfinite(metrics.sharpe) else "—")
    cols[3].metric("Sortino", f"{metrics.sortino:.2f}" if math.isfinite(metrics.sortino) else "—")
    cols[4].metric("Calmar", f"{metrics.calmar:.2f}" if math.isfinite(metrics.calmar) else "—")
    cols[5].metric("Max DD", f"{metrics.max_drawdown_pct * 100:.1f}%")
    cols[6].metric("Win Rate", f"{metrics.win_rate * 100:.1f}%")
    cols[7].metric("PF", f"{metrics.profit_factor:.2f}" if metrics.profit_factor is not None else "—")
    st.caption(
        "Sharpe / Sortino / Calmar here are **per-trade annualised** (√trades-per-year) — "
        "NOT comparable to the bot's own per-bar metrics (√8760). Win rate, PF, return % "
        "and max-DD ARE comparable across both."
    )


def _degradation_banner_text(result: WFAResult) -> tuple[str, str, str]:
    """Compute the (color, headline, body) text for the degradation banner.

    Pure so it can be unit tested without streamlit. MUST use the SAME basis
    result.degradation was computed from (per-month, matched folds only —
    see wfa.runner.run_wfa) so the printed numbers are internally consistent
    with the printed percentage. Do not substitute full-span figures
    (result.mean_is_return / result.stitched_metrics.total_return) here —
    those come from a different fold population and time basis.
    """
    assert result.degradation is not None
    deg_pct = result.degradation * 100
    oos_pm = result.oos_return_per_month * 100
    is_pm = result.mean_is_return_per_month * 100
    if deg_pct < 0:
        color = "green"
        headline = f"Return Degradation: {deg_pct:.1f}% (OOS outperformed IS)"
        body = (
            f"Blind-test (OOS) return **exceeded** mean in-sample — no curve-fit decay. "
            f"Per-month, matched folds: OOS {oos_pm:.2f}%/mo vs IS {is_pm:.2f}%/mo. Treat "
            f"with care: this can be a lucky OOS sample rather than genuine robustness."
        )
    else:
        color = "red" if deg_pct > 70 else "orange" if deg_pct > 40 else "green"
        headline = f"Return Degradation: {deg_pct:.1f}%"
        body = (
            f"**{deg_pct:.0f}%** of in-sample return did **not** transfer to blind test. "
            f"Per-month, matched folds: OOS {oos_pm:.2f}%/mo vs IS {is_pm:.2f}%/mo."
        )
    return color, headline, body


def _degradation_unavailable_message(result: WFAResult) -> str:
    """Explain WHY result.degradation is None — branch on the actual cause.

    degradation is None when the matched-fold set (both IS re-run and OOS run
    succeeded) is empty, or when its per-month IS baseline is <= 0 (see
    wfa.metrics.return_degradation / wfa.runner.run_wfa). A single hardcoded
    "IS return was zero" string is false when folds failed outright — this
    mirrors run_wfa.py's n_failed_folds / n_degradation_unavailable reporting
    instead of guessing.
    """
    if result.n_failed_folds > 0:
        return (
            f"Degradation not computed — {result.n_failed_folds} fold(s) FAILED "
            f"(OOS blind test raised, lost evidence). No matched folds to compare."
        )
    if result.n_degradation_unavailable > 0 and result.n_degradation_unavailable >= len(result.folds):
        return (
            f"Degradation not computed — all {result.n_degradation_unavailable} fold(s) "
            f"had an IS re-run error (OOS was valid); no matched folds to compare."
        )
    return (
        "Degradation not computed (mean IS return per month was zero, negative, "
        "or not finite)."
    )


def _degraded_banner() -> None:
    """Loud warning when results may rest on <90% real-CVD coverage.

    The degraded-CVD override (EXAMPLEBOT_ALLOW_DEGRADED_CVD) lets a run proceed on the
    OHLCV CVD approximation, which is known to INVERT conclusions. The per-fold watermark
    (trades_df.attrs['degraded_cvd']) does not survive pd.concat stitching, so key off the
    env that enabled it — the necessary condition for any degraded fold.
    """
    if os.environ.get("EXAMPLEBOT_ALLOW_DEGRADED_CVD"):
        st.error(
            "EXAMPLEBOT_ALLOW_DEGRADED_CVD is set — results may use the CVD approximation "
            "(<90% real CVD), which can INVERT conclusions. Treat every metric below as "
            "untrustworthy until re-run with full real-CVD coverage.",
            icon="🚨",
        )


# ── WFA tab ───────────────────────────────────────────────────────────────────

def render_wfa_tab(cfg: dict) -> None:
    st.header("Walk-Forward Analysis")

    adapter = cfg["adapter"]
    if adapter is None:
        err_msg = _adapter_failure_message(cfg["bot_name"], cfg.get("adapter_error"))
        if err_msg:
            st.error(err_msg, icon="🚨")
        else:
            st.warning("Load a bot in the sidebar first.")
        return

    run_btn = st.button("Run WFA", type="primary", key="run_wfa_btn")

    if run_btn:
        progress = st.progress(0.0, text="Starting WFA...")

        try:
            _fold_count: list[int] = [0]
            _total_folds: list[int] = [0]

            from wfa.folds import generate_folds
            data_start, data_end = adapter.data_range()
            preview_folds = generate_folds(
                data_start, data_end, cfg["train_months"], cfg["test_months"]
            )
            _total_folds[0] = len(preview_folds)
            progress.progress(0.05, text=f"Prepared {len(preview_folds)} folds…")

            result: WFAResult = run_wfa(
                adapter=adapter,
                train_months=cfg["train_months"],
                test_months=cfg["test_months"],
                objective_name=cfg["objective"],
                search_method=cfg["search_method"],
                search_budget=cfg["search_budget"],
                min_trades_per_fold=cfg["min_trades"],
                seed=cfg["seed"],
                initial_capital=cfg["initial_capital"],
            )
            progress.progress(1.0, text="WFA complete.")
            st.session_state[_WFA_KEY] = result
            st.session_state[_BOT_KEY] = cfg["bot_name"]
            st.session_state[_CAPITAL_KEY] = cfg["initial_capital"]

            # Store for Compare tab
            compare: dict = st.session_state.get(_COMPARE_KEY, {})
            compare[cfg["bot_name"]] = result
            st.session_state[_COMPARE_KEY] = compare

        except Exception as e:
            st.error(f"WFA failed: {e}")
            st.exception(e)
            return

    result: WFAResult | None = _stored_wfa_result(st.session_state, cfg["bot_name"])
    if result is None:
        stale_bot = _stale_result_bot(st.session_state)
        if stale_bot is not None:
            st.warning(
                f"No WFA result for **{cfg['bot_name']}** — the stored result belongs "
                f"to **{stale_bot}**. Press **Run WFA** to analyse {cfg['bot_name']}."
            )
        else:
            st.info("Press **Run WFA** to start.")
        return

    if not result.folds:
        st.warning("No folds produced. Check data range vs window settings.")
        return

    # ── degraded-CVD trust banner (before any metric is shown) ────────────────
    _degraded_banner()

    # ── degradation callout ───────────────────────────────────────────────────
    st.markdown("---")

    if result.degradation is not None:
        color, headline, body = _degradation_banner_text(result)
        st.markdown(f"<h2 style='color:{color};'>{headline}</h2>", unsafe_allow_html=True)
        st.markdown(body)
        st.caption(HELP["degradation"])
    else:
        st.info(_degradation_unavailable_message(result))

    # ── stitched OOS metrics ──────────────────────────────────────────────────
    _metrics_cards(result.stitched_metrics)

    # ── equity chart ──────────────────────────────────────────────────────────
    if not result.stitched_trades.empty and "pnl" in result.stitched_trades.columns:
        st.subheader("Stitched OOS Equity Curve")
        # Rebuild with the capital the RUN used, not the live sidebar slider.
        equity = [_run_capital(st.session_state, cfg["initial_capital"])]
        for pnl in result.stitched_trades["pnl"]:
            equity.append(equity[-1] + float(pnl))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=equity,
                mode="lines",
                name="Stitched OOS",
                line={"color": "steelblue", "width": 2},
            )
        )
        fig.update_layout(
            title="Stitched OOS equity (all blind-test folds concatenated)",
            xaxis_title="Trade #",
            yaxis_title="Equity ($)",
            height=350,
            margin={"t": 40, "b": 20},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── per-fold table ────────────────────────────────────────────────────────
    st.subheader("Per-Fold Results")
    rows = []
    for fr in result.folds:
        rows.append({
            "Fold": fr.fold.index + 1,
            "Train start": str(fr.fold.train_start.date()),
            "Train end": str(fr.fold.train_end.date()),
            "Test start": str(fr.fold.test_start.date()),
            "Test end": str(fr.fold.test_end.date()),
            "Best params": str(fr.best_params),
            "IS return": f"{fr.is_return * 100:.1f}%" if fr.is_return is not None and abs(fr.is_return) < 1e6 else "—",
            "IS obj": f"{fr.is_objective:.3f}",
            "OOS trades": fr.oos_n_trades,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── param drift heatmap ───────────────────────────────────────────────────
    if result.folds:
        schema = adapter.param_schema()
        param_names = list(schema.keys())

        if param_names:
            st.subheader("Param Drift Heatmap")
            st.caption(
                "Each row = fold, each column = parameter chosen by IS optimisation. "
                "Stable columns = robust params. Unstable = regime-sensitive / low confidence."
            )

            heat_rows = []
            for fr in result.folds:
                row = {"Fold": fr.fold.index + 1}
                for p in param_names:
                    row[p] = fr.best_params.get(p, None)
                heat_rows.append(row)

            heat_df = pd.DataFrame(heat_rows).set_index("Fold")

            # Encode param values numerically for colour
            fig_h = go.Figure()
            for col in heat_df.columns:
                vals = heat_df[col].tolist()
                choices = schema[col].get("choices", [])
                try:
                    z_col = [choices.index(v) if v in choices else 0 for v in vals]
                except Exception:
                    z_col = [0] * len(vals)
                text_col = [str(v) for v in vals]
                fig_h.add_trace(
                    go.Heatmap(
                        z=[z_col],
                        x=list(range(1, len(vals) + 1)),
                        y=[col],
                        text=[text_col],
                        texttemplate="%{text}",
                        colorscale="Blues",
                        showscale=False,
                        xgap=2,
                        ygap=2,
                    )
                )

            fig_h.update_layout(
                height=max(120, 60 * len(param_names)),
                xaxis_title="Fold",
                margin={"t": 20, "b": 30},
            )
            st.plotly_chart(fig_h, use_container_width=True)


# ── MC tab ────────────────────────────────────────────────────────────────────

def render_mc_tab(cfg: dict) -> None:
    st.header("Monte Carlo Simulation")

    wfa_result: WFAResult | None = _stored_wfa_result(st.session_state, cfg["bot_name"])

    if wfa_result is not None and not wfa_result.stitched_trades.empty:
        source_label = (
            f"Stitched OOS from last **{cfg['bot_name']}** WFA run "
            f"({len(wfa_result.stitched_trades)} trades)"
        )
        source_trades = wfa_result.stitched_trades
        mc_source = "stitched_oos"
        st.info(f"Input: {source_label}")
        _degraded_banner()
    else:
        stale_bot = _stale_result_bot(st.session_state)
        if stale_bot is not None and stale_bot != cfg["bot_name"]:
            st.warning(
                f"No WFA result for **{cfg['bot_name']}** — the stored result belongs "
                f"to **{stale_bot}**. Run WFA for {cfg['bot_name']} first."
            )
        else:
            st.warning("No WFA result available. Run WFA first to get stitched OOS trades.")
        return

    run_btn = st.button("Run MC", type="primary", key="run_mc_btn")

    if run_btn:
        with st.spinner(f"Running {cfg['mc_sims']:,} simulations…"):
            try:
                mc: MCResult = run_mc(
                    trades=source_trades,
                    initial_capital=cfg["initial_capital"],
                    n_sims=cfg["mc_sims"],
                    method=cfg["mc_method"],
                    ruin_threshold=cfg["ruin_threshold"],
                    compounding=True,
                    seed=cfg["seed"],
                    source=mc_source,
                )
                st.session_state[_MC_KEY] = mc
                st.session_state[_MC_BOT_KEY] = cfg["bot_name"]
            except Exception as e:
                st.error(f"MC failed: {e}")
                st.exception(e)
                return

    mc: MCResult | None = _stored_mc_result(st.session_state, cfg["bot_name"])
    if mc is None:
        if st.session_state.get(_MC_KEY) is not None:
            stale_mc_bot = st.session_state.get(_MC_BOT_KEY) or "unknown"
            st.warning(
                f"No MC result for **{cfg['bot_name']}** — the stored MC run belongs "
                f"to **{stale_mc_bot}**. Press **Run MC** for {cfg['bot_name']}."
            )
        else:
            st.info("Press **Run MC** to start.")
        return

    # ── headline cards ────────────────────────────────────────────────────────
    ruin_pct = mc.probability_of_ruin * 100
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "P(Ruin)",
        f"{ruin_pct:.1f}%",
        help=f"Fraction of paths that hit ≥{cfg['ruin_threshold'] * 100:.0f}% drawdown.",
    )
    col2.metric(
        "Median return",
        f"{mc.total_return_pct.p50:.1f}%",
    )
    col3.metric(
        "Max DD p95",
        f"{mc.max_drawdown_pct.p95:.1f}%",
    )
    col4.metric(
        "Return 90% CI",
        f"[{mc.total_return_pct.p5:.1f}%, {mc.total_return_pct.p95:.1f}%]",
    )

    st.markdown("---")

    # ── equity fan ────────────────────────────────────────────────────────────
    st.subheader("Equity Fan (sampled paths)")

    _N_SHOW = 200
    import numpy as np
    n_paths, n_steps = mc.equity_curves.shape
    sample_idx = np.random.RandomState(0).choice(n_paths, size=min(_N_SHOW, n_paths), replace=False)
    x = list(range(n_steps))

    # Single trace with None separators — avoids 200 separate trace objects
    fan_x: list = []
    fan_y: list = []
    for i in sample_idx:
        fan_x.extend(x)
        fan_x.append(None)
        fan_y.extend(mc.equity_curves[i].tolist())
        fan_y.append(None)

    fig_fan = go.Figure()
    fig_fan.add_trace(
        go.Scatter(
            x=fan_x,
            y=fan_y,
            mode="lines",
            line={"width": 0.4, "color": "rgba(100,149,237,0.15)"},
            showlegend=False,
        )
    )
    # Median path
    median_path = np.percentile(mc.equity_curves, 50, axis=0)
    fig_fan.add_trace(
        go.Scatter(
            x=x,
            y=median_path.tolist(),
            mode="lines",
            name="Median",
            line={"color": "steelblue", "width": 2.5},
        )
    )
    fig_fan.update_layout(
        height=350,
        xaxis_title="Trade #",
        yaxis_title="Equity ($)",
        margin={"t": 20},
    )
    st.plotly_chart(fig_fan, use_container_width=True)

    # ── 2×2 panel ─────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Max Drawdown Distribution")
        fig_dd = go.Figure(
            go.Histogram(
                x=mc.all_max_dd_pct.tolist(),
                nbinsx=50,
                marker_color="salmon",
                opacity=0.8,
            )
        )
        fig_dd.update_layout(
            xaxis_title="Max DD %",
            yaxis_title="Count",
            height=260,
            margin={"t": 10, "b": 30},
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    with col_b:
        st.subheader("Final Return Distribution")
        fig_ret = go.Figure(
            go.Histogram(
                x=mc.all_return_pct.tolist(),
                nbinsx=50,
                marker_color="mediumseagreen",
                opacity=0.8,
            )
        )
        fig_ret.update_layout(
            xaxis_title="Final Return %",
            yaxis_title="Count",
            height=260,
            margin={"t": 10, "b": 30},
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Longest Losing Streak")
        fig_ls = go.Figure(
            go.Histogram(
                x=mc.all_losing_streak.tolist(),
                nbinsx=30,
                marker_color="mediumpurple",
                opacity=0.8,
            )
        )
        fig_ls.update_layout(
            xaxis_title="Streak (# trades)",
            yaxis_title="Count",
            height=260,
            margin={"t": 10, "b": 30},
        )
        st.plotly_chart(fig_ls, use_container_width=True)

    with col_d:
        st.subheader("Sharpe Distribution")
        fig_sh = go.Figure(
            go.Histogram(
                x=mc.all_sharpe.tolist(),
                nbinsx=50,
                marker_color="goldenrod",
                opacity=0.8,
            )
        )
        fig_sh.update_layout(
            xaxis_title="Sharpe",
            yaxis_title="Count",
            height=260,
            margin={"t": 10, "b": 30},
        )
        st.plotly_chart(fig_sh, use_container_width=True)


# ── Compare tab ───────────────────────────────────────────────────────────────

def render_compare_tab() -> None:
    st.header("Bot Comparison")
    st.caption("Run WFA for each bot to populate this chart.")

    compare: dict[str, WFAResult] = st.session_state.get(_COMPARE_KEY, {})

    if not compare:
        st.info("No WFA results yet. Run WFA on at least one bot.")
        return

    rows = []
    for bot_name, result in compare.items():
        m = result.stitched_metrics
        rows.append({
            "Bot": bot_name,
            "Sortino": m.sortino,
            "Sharpe": m.sharpe,
            "Calmar": m.calmar,
            "Total Return %": round(m.total_return * 100, 2),
            "Max DD %": round(m.max_drawdown_pct * 100, 2),
            "Win Rate %": round(m.win_rate * 100, 2),
            "Profit Factor": m.profit_factor,
            "Trades": m.n_trades,
            "Degradation %": (
                round(result.degradation * 100, 1) if result.degradation is not None else None
            ),
        })

    df = pd.DataFrame(rows).set_index("Bot")
    st.dataframe(df, use_container_width=True)

    # Sortino bar chart
    bots = list(compare.keys())
    sortinos = [compare[b].stitched_metrics.sortino for b in bots]

    fig = go.Figure(
        go.Bar(
            x=bots,
            y=sortinos,
            marker_color=["steelblue" if s >= 0 else "salmon" for s in sortinos],
            text=[f"{s:.2f}" if s is not None else "—" for s in sortinos],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Stitched OOS Sortino by Bot",
        yaxis_title="Sortino",
        height=350,
        margin={"t": 50, "b": 20},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Clear button
    if st.button("Clear comparison data"):
        st.session_state[_COMPARE_KEY] = {}
        st.rerun()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Unauthenticated portal on a bot box: refuse non-loopback binding unless
    # remote exposure was explicitly accepted via env.
    address = st.get_option("server.address")
    if not _binding_allowed(address, os.environ.get(_ALLOW_REMOTE_ENV)):
        st.error(
            f"REFUSING TO SERVE: bound to '{address or 'all interfaces'}' with no "
            "authentication. Relaunch with "
            "`streamlit run portal/app.py --server.address localhost`, or set "
            f"{_ALLOW_REMOTE_ENV}=1 to accept remote exposure explicitly.",
            icon="🚨",
        )
        st.stop()

    cfg = render_sidebar()

    tab_setup, tab_wfa, tab_mc, tab_compare = st.tabs(
        ["Setup", "WFA Run", "MC Run", "Compare"]
    )

    with tab_setup:
        render_setup_tab(cfg)

    with tab_wfa:
        render_wfa_tab(cfg)

    with tab_mc:
        render_mc_tab(cfg)

    with tab_compare:
        render_compare_tab()


if __name__ == "__main__":
    main()
