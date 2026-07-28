"""Tests for portal/app.py pure state/config logic.

Covers review findings:
  #1  (HIGH) _BOT_KEY stored but never checked — a selected bot could display
      the PREVIOUS bot's WFA/MC result (`_stored_wfa_result` / `_stored_mc_result`).
  MED app.py:67  load_adapter exceptions swallowed — broken adapter was
      indistinguishable from "no adapter" (`_adapter_failure_message`).
  MED app.py:27  no server config — default binding exposes the unauthenticated
      portal on all interfaces (`_binding_allowed`).
  LOW app.py:409 equity chart rebuilt with the CURRENT sidebar capital instead
      of the capital the run used (`_run_capital`).

streamlit/plotly are stubbed when not installed — the helpers under test are
pure functions over a Mapping and never touch either library.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _stub_missing_ui_deps() -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        st = types.ModuleType("streamlit")
        st.set_page_config = lambda **kwargs: None  # module-level call in app.py
        sys.modules["streamlit"] = st
    try:
        import plotly.graph_objects  # noqa: F401
    except ImportError:
        plotly = types.ModuleType("plotly")
        go = types.ModuleType("plotly.graph_objects")
        for name in ("Figure", "Scatter", "Heatmap", "Histogram", "Bar"):
            setattr(go, name, type(name, (), {}))
        plotly.graph_objects = go  # type: ignore[attr-defined]
        sys.modules["plotly"] = plotly
        sys.modules["plotly.graph_objects"] = go


_stub_missing_ui_deps()

from portal.app import (  # noqa: E402
    _BOT_KEY,
    _CAPITAL_KEY,
    _MC_BOT_KEY,
    _MC_KEY,
    _WFA_KEY,
    _adapter_failure_message,
    _binding_allowed,
    _degradation_banner_text,
    _degradation_unavailable_message,
    _run_capital,
    _stale_result_bot,
    _stored_mc_result,
    _stored_wfa_result,
)


class _FakeResult:
    """Stand-in for WFAResult — helpers only touch .bot_name via getattr."""

    def __init__(self, bot_name: str) -> None:
        self.bot_name = bot_name


# ── finding #1: WFA result gated on the requested bot ──────────────────────────

def test_wfa_result_returned_for_matching_bot() -> None:
    result = _FakeResult("mybot")
    state = {_WFA_KEY: result, _BOT_KEY: "mybot"}
    assert _stored_wfa_result(state, "mybot") is result


def test_wfa_result_hidden_for_other_bot() -> None:
    """Switching the sidebar bot must NOT show the previous bot's result."""
    state = {_WFA_KEY: _FakeResult("mybot"), _BOT_KEY: "mybot"}
    assert _stored_wfa_result(state, "samplebot-a") is None


def test_wfa_result_hidden_when_bot_key_missing() -> None:
    """Fail-safe: an unattributed result must never render as any bot's data."""
    state = {_WFA_KEY: _FakeResult("mybot")}
    assert _stored_wfa_result(state, "mybot") is None


def test_no_stored_result_returns_none() -> None:
    assert _stored_wfa_result({}, "mybot") is None


def test_stale_result_bot_names_the_owner() -> None:
    state = {_WFA_KEY: _FakeResult("mybot"), _BOT_KEY: "mybot"}
    assert _stale_result_bot(state) == "mybot"


def test_stale_result_bot_falls_back_to_result_attr_then_unknown() -> None:
    assert _stale_result_bot({_WFA_KEY: _FakeResult("samplebot-a")}) == "samplebot-a"
    assert _stale_result_bot({_WFA_KEY: object()}) == "unknown"
    assert _stale_result_bot({}) is None


# ── finding #1 (MC leg): MC result gated on the requested bot ──────────────────

def test_mc_result_returned_for_matching_bot() -> None:
    mc = object()
    state = {_MC_KEY: mc, _MC_BOT_KEY: "mybot"}
    assert _stored_mc_result(state, "mybot") is mc


def test_mc_result_hidden_for_other_bot() -> None:
    state = {_MC_KEY: object(), _MC_BOT_KEY: "mybot"}
    assert _stored_mc_result(state, "samplebot-a") is None


def test_mc_result_hidden_when_bot_key_missing() -> None:
    assert _stored_mc_result({_MC_KEY: object()}, "mybot") is None


# ── MED app.py:27: loopback-only binding ───────────────────────────────────────

def test_binding_default_unset_is_refused() -> None:
    """Streamlit's default (address unset) binds all interfaces — refuse it."""
    assert _binding_allowed(None, None) is False
    assert _binding_allowed("", None) is False


def test_binding_all_interfaces_is_refused() -> None:
    assert _binding_allowed("0.0.0.0", None) is False
    assert _binding_allowed("192.168.1.20", None) is False


def test_binding_loopback_is_allowed() -> None:
    assert _binding_allowed("localhost", None) is True
    assert _binding_allowed("127.0.0.1", None) is True
    assert _binding_allowed("::1", None) is True


def test_binding_env_override_allows_remote() -> None:
    assert _binding_allowed("0.0.0.0", "1") is True
    assert _binding_allowed(None, "1") is True


# ── MED app.py:67: broken adapter is loud, not silent ──────────────────────────

def test_adapter_failure_message_surfaces_the_error() -> None:
    msg = _adapter_failure_message(
        "mybot", "DataQualityError: CVD coverage 62% < 90%"
    )
    assert msg is not None
    assert "mybot" in msg
    assert "DataQualityError: CVD coverage 62% < 90%" in msg
    assert "FAILED" in msg


def test_adapter_failure_message_none_when_no_error() -> None:
    assert _adapter_failure_message("mybot", None) is None
    assert _adapter_failure_message("mybot", "") is None


# ── LOW app.py:409: equity chart uses the RUN's capital, not the slider ────────

def test_run_capital_prefers_stored_run_capital() -> None:
    """Bumping the sidebar slider after a run must not rescale the chart."""
    state = {_CAPITAL_KEY: 10_000.0}
    assert _run_capital(state, 100_000.0) == 10_000.0


def test_run_capital_falls_back_when_unset() -> None:
    assert _run_capital({}, 100_000.0) == 100_000.0


# ── MED app.py:466: degradation banner must be internally consistent ───────────
#
# result.degradation is computed on a PER-MONTH, MATCHED-FOLDS-ONLY basis
# (wfa/runner.py). The banner text must quote the SAME basis, not the
# full-span stitched_metrics.total_return / mean_is_return (a different fold
# population and time basis) — otherwise the printed % contradicts the
# printed returns beside it.

class _FakeDegradedResult:
    """Stand-in for WFAResult with only the fields the banner helpers touch."""

    def __init__(
        self,
        degradation: float | None,
        mean_is_return_per_month: float = 0.0,
        oos_return_per_month: float = 0.0,
        mean_is_return: float = 0.0,
        n_failed_folds: int = 0,
        n_degradation_unavailable: int = 0,
        n_folds: int = 1,
    ) -> None:
        self.degradation = degradation
        self.mean_is_return_per_month = mean_is_return_per_month
        self.oos_return_per_month = oos_return_per_month
        self.mean_is_return = mean_is_return
        self.n_failed_folds = n_failed_folds
        self.n_degradation_unavailable = n_degradation_unavailable
        self.folds = [object()] * n_folds


def test_degradation_banner_quotes_per_month_matched_basis_not_full_span() -> None:
    """Old-code regression: full-span IS return (18%) must NOT appear in the
    banner body when the per-month matched-fold basis (1.5%) is what backs
    the printed degradation %. This is the app.py:466 finding's failure case.
    """
    result = _FakeDegradedResult(
        degradation=0.0,
        mean_is_return_per_month=0.015,   # 1.5%/mo — what degradation is computed from
        oos_return_per_month=0.015,
        mean_is_return=0.18,              # 18% full-span — must NOT leak into the banner
    )
    color, headline, body = _degradation_banner_text(result)
    assert "1.5" in body
    assert "18.0" not in body
    assert color == "green"


def test_degradation_banner_negative_deg_reports_matched_per_month_figures() -> None:
    result = _FakeDegradedResult(
        degradation=-1.0,  # OOS outperformed IS
        mean_is_return_per_month=0.01,
        oos_return_per_month=0.02,
        mean_is_return=0.36,  # full-span decoy value that must not appear
    )
    color, headline, body = _degradation_banner_text(result)
    assert color == "green"
    assert "outperformed" in headline
    assert "2.00" in body  # oos_return_per_month * 100
    assert "1.00" in body  # mean_is_return_per_month * 100
    assert "36.0" not in body


def test_degradation_banner_color_thresholds() -> None:
    mild = _FakeDegradedResult(degradation=0.30, mean_is_return_per_month=0.01, oos_return_per_month=0.007)
    med = _FakeDegradedResult(degradation=0.50, mean_is_return_per_month=0.01, oos_return_per_month=0.005)
    severe = _FakeDegradedResult(degradation=0.80, mean_is_return_per_month=0.01, oos_return_per_month=0.002)
    assert _degradation_banner_text(mild)[0] == "green"
    assert _degradation_banner_text(med)[0] == "orange"
    assert _degradation_banner_text(severe)[0] == "red"


# ── LOW app.py:495: "not computed" fallback must name the real cause ───────────

def test_degradation_unavailable_names_failed_folds_not_zero_is() -> None:
    """Old-code regression: when every fold's OOS raised (n_failed_folds > 0),
    the message must say so — NOT the generic 'IS return was zero or not
    finite', which is false and hides a broken run behind a benign label.
    """
    result = _FakeDegradedResult(degradation=None, n_failed_folds=3, n_folds=3)
    msg = _degradation_unavailable_message(result)
    assert "FAILED" in msg
    assert "3" in msg
    assert "zero or not finite" not in msg


def test_degradation_unavailable_names_all_is_reruns_errored() -> None:
    result = _FakeDegradedResult(
        degradation=None, n_failed_folds=0, n_degradation_unavailable=2, n_folds=2,
    )
    msg = _degradation_unavailable_message(result)
    assert "IS re-run error" in msg
    assert "zero or not finite" not in msg


def test_degradation_unavailable_falls_back_to_negative_is_baseline_message() -> None:
    """No failed/errored folds at all — the real cause is a non-positive
    per-month IS baseline (return_degradation's <= 0.0 guard)."""
    result = _FakeDegradedResult(degradation=None, n_failed_folds=0, n_degradation_unavailable=0, n_folds=2)
    msg = _degradation_unavailable_message(result)
    assert "FAILED" not in msg
    assert "zero, negative, or not finite" in msg
