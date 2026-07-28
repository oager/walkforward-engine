"""Tests for scripts/rolling_psr_tripwire.py (code-review findings #8, #9, #10, #16).

These import the live monitor module directly and exercise its pure functions:
status classification (incl. STALE_DATA), alert rendering (no green banner on
degraded runs), Telegram retry, exit codes, and .env parsing. All of these fail
against the pre-review script (no classify_status/build_alert/compute_exit_code,
hardcoded /home/user chdir crashes the import itself off-box).
"""
from __future__ import annotations

import io
import json
import logging
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import rolling_psr_tripwire as rpt


def R(bot: str, status: str, psr: float | None = None, n: int | None = None,
      window: str = "2025-01-01..2026-07-01", **kw) -> dict:
    d = {"bot": bot, "status": status, "psr": psr, "n": n, "window": window,
         "floor": 0.95, "warn": 0.90}
    d.update(kw)
    return d


# ── classify_status (pure) ────────────────────────────────────────────────────

def test_classify_ok() -> None:
    assert rpt.classify_status(0.97, 100, 0.95, 0.90) == "OK"


def test_classify_warn_band() -> None:
    assert rpt.classify_status(0.92, 100, 0.95, 0.90) == "WARN"


def test_classify_tripped() -> None:
    assert rpt.classify_status(0.85, 100, 0.95, 0.90) == "TRIPPED"


def test_classify_low_n() -> None:
    assert rpt.classify_status(0.99, 10, 0.95, 0.90) == "LOW_N"


def test_classify_no_psr() -> None:
    assert rpt.classify_status(None, 50, 0.95, 0.90) == "NO_PSR"


def test_classify_stale_trumps_ok_psr() -> None:
    # Finding #9: a perfect PSR off a frozen window must NOT read as OK.
    assert rpt.classify_status(0.99, 200, 0.95, 0.90,
                               staleness_days=30.0, max_stale_days=8.0) == "STALE_DATA"


def test_classify_fresh_data_not_stale() -> None:
    assert rpt.classify_status(0.99, 200, 0.95, 0.90,
                               staleness_days=2.0, max_stale_days=8.0) == "OK"


# ── build_alert (finding #8: degraded runs must never render green) ───────────

def test_alert_all_ok_is_green_and_shows_window() -> None:
    txt = rpt.build_alert([R("mybot", "OK", 0.98, 100),
                           R("otherbot", "OK", 0.97, 90)])
    assert txt.startswith("✅")
    assert "DEGRADED" not in txt
    assert "recent crypto-MR edge intact" in txt
    assert "2025-01-01..2026-07-01" in txt  # finding #9: window visible in the alert


def test_alert_primary_error_is_not_green() -> None:
    # mybot (PRIMARY) ERROR => the monitor can't judge the crypto edge => DEGRADED + red.
    txt = rpt.build_alert([R("mybot", "ERROR", detail="FileNotFoundError: cfg"),
                           R("otherbot", "ERROR", detail="ImportError: adapter")])
    assert not txt.startswith("✅")
    assert txt.startswith("\U0001f534")
    assert "DEGRADED" in txt
    assert "recent crypto-MR edge intact" not in txt
    assert "no current-regime verdict" in txt  # per-book mybot footer, not a joint verdict


def test_alert_low_n_is_not_green() -> None:
    txt = rpt.build_alert([R("mybot", "LOW_N", None, 3),
                           R("otherbot", "OK", 0.97, 90)])
    assert not txt.startswith("✅")
    assert "DEGRADED" in txt
    assert "recent crypto-MR edge intact" not in txt


def test_alert_no_psr_is_not_green() -> None:
    # NO_PSR is a PRIMARY (PSR-path) status; otherbot is guard-judged and never emits it.
    txt = rpt.build_alert([R("mybot", "NO_PSR", None, 50)])
    assert not txt.startswith("✅")
    assert "recent crypto-MR edge intact" not in txt


def test_alert_stale_data_is_red_with_detail() -> None:
    # A PRIMARY (mybot) frozen data window => monitor's own job broke => red.
    txt = rpt.build_alert([R("mybot", "STALE_DATA", 0.99, 200,
                             detail="data ends 2026-05-01 (60d old > 8d)")])
    assert txt.startswith("\U0001f534")
    assert "data ends 2026-05-01" in txt
    assert "recent crypto-MR edge intact" not in txt


def test_alert_tripped_is_red_with_regime_warning() -> None:
    txt = rpt.build_alert([R("mybot", "TRIPPED", 0.80, 100),
                           R("otherbot", "OK", 0.97, 90)])
    assert txt.startswith("\U0001f534")
    assert "below floor" in txt


def test_alert_scoring_drift_is_not_green() -> None:
    txt = rpt.build_alert([R("mybot", "OK", 0.98, 100, scoring_drift_warnings=7)])
    assert not txt.startswith("✅")
    assert "scoring_drift=7" in txt


# ── check() staleness end-to-end via a stub adapter (finding #9) ──────────────

class _StubAdapter:
    def __init__(self, de: datetime):
        self._de = de

    def param_schema(self) -> dict:
        return {"x": {"default": 1}}

    def data_range(self):
        return self._de - timedelta(days=900), self._de

    def run(self, params, start, end) -> pd.DataFrame:
        return pd.DataFrame({"pnl": [10.0, -5.0] * 30})


def test_check_frozen_window_reports_stale(monkeypatch) -> None:
    monkeypatch.setattr(
        rpt, "load_adapter",
        lambda b: _StubAdapter(datetime.now() - timedelta(days=30)))
    r = rpt.check("mybot", 18.0, 0.95, 0.90, max_stale_days=8.0)
    assert r["status"] == "STALE_DATA"
    assert r["staleness_days"] > 8
    assert "window frozen" in r["detail"]


def test_check_fresh_window_not_stale(monkeypatch) -> None:
    monkeypatch.setattr(
        rpt, "load_adapter",
        lambda b: _StubAdapter(datetime.now() - timedelta(days=1)))
    r = rpt.check("mybot", 18.0, 0.95, 0.90, max_stale_days=8.0)
    assert r["status"] != "STALE_DATA"
    assert "staleness_days" in r  # additive schema field present
    # old schema keys still present (additive-only contract)
    for key in ("bot", "window", "months", "n", "psr", "floor", "warn",
                "status", "survival_pass"):
        assert key in r


def test_check_adapter_failure_is_error(monkeypatch) -> None:
    def boom(b):
        raise FileNotFoundError("~/.wfa/config.toml missing")
    monkeypatch.setattr(rpt, "load_adapter", boom)
    r = rpt.check("otherbot", 18.0, 0.95, 0.90)
    assert r["status"] == "ERROR"
    assert "FileNotFoundError" in r["detail"]


# ── send_telegram retry (finding #10) ─────────────────────────────────────────

def test_send_retries_then_succeeds(monkeypatch) -> None:
    calls: list[int] = []

    def fake_urlopen(url, data=None, timeout=None):
        calls.append(1)
        if len(calls) < 3:
            raise OSError("transient DNS blip")
        return io.BytesIO(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ok = rpt.send_telegram("tok", "chat", "msg", attempts=3, sleep=lambda s: None)
    assert ok is True
    assert len(calls) == 3


def test_send_fails_after_bounded_attempts(monkeypatch) -> None:
    calls: list[int] = []

    def fake_urlopen(url, data=None, timeout=None):
        calls.append(1)
        raise OSError("telegram down")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ok = rpt.send_telegram("tok", "chat", "msg", attempts=3, sleep=lambda s: None)
    assert ok is False
    assert len(calls) == 3  # bounded — no infinite retry


def test_send_requires_ok_true(monkeypatch) -> None:
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda url, data=None, timeout=None:
            io.BytesIO(json.dumps({"ok": False, "description": "chat not found"}).encode()))
    ok = rpt.send_telegram("tok", "chat", "msg", attempts=2, sleep=lambda s: None)
    assert ok is False


def test_notify_missing_creds_prints_alert_and_fails(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(rpt, "ENV_PATH", str(tmp_path / "nonexistent.env"))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("EXAMPLEBOT_HEALTH_TG_CHAT", raising=False)
    ok = rpt._notify_telegram([R("mybot", "TRIPPED", 0.80, 100)])
    assert ok is False
    out = capsys.readouterr().out
    assert "UNDELIVERED ALERT" in out
    assert "TRIPPED" in out  # the full alert is visible in the journal


def test_notify_send_failure_prints_alert(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(rpt, "ENV_PATH", str(tmp_path / "nonexistent.env"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("EXAMPLEBOT_HEALTH_TG_CHAT", "chat")
    monkeypatch.setattr(rpt, "send_telegram", lambda *a, **k: False)
    ok = rpt._notify_telegram([R("mybot", "OK", 0.98, 100)])
    assert ok is False
    assert "UNDELIVERED ALERT" in capsys.readouterr().out


# ── compute_exit_code (finding #16) ───────────────────────────────────────────

def test_exit_code_all_ok_delivered() -> None:
    assert rpt.compute_exit_code([R("mybot", "OK", 0.98, 100)], True, True) == 0


def test_exit_code_error_result_nonzero() -> None:
    assert rpt.compute_exit_code(
        [R("mybot", "ERROR", detail="x")], False, True) != 0


def test_exit_code_otherbot_stale_does_not_fail_unit() -> None:
    # otherbot is a separate live book (guard-judged), NOT the crypto edge the monitor
    # exists to watch. Its STALE_DATA must not fail the weekly unit while mybot is OK.
    res = [R("mybot", "OK", 0.97, 100), R("otherbot", "STALE_DATA", None, 0),
           R("samplebot-c", "OK", 0.96, 50)]
    assert rpt.compute_exit_code(res, False, True) == 0


def test_exit_code_notify_failure_nonzero() -> None:
    assert rpt.compute_exit_code([R("mybot", "OK", 0.98, 100)], True, False) != 0


def test_exit_code_tripped_but_delivered_is_zero() -> None:
    # TRIPPED is a valid, successfully delivered verdict — the unit did its job.
    assert rpt.compute_exit_code([R("mybot", "TRIPPED", 0.80, 100)], True, True) == 0


# ── primary/secondary split (re-review 2026-07-02 regression fix) ─────────────

def test_exit_code_secondary_stale_does_not_fail_unit() -> None:
    # gold is a SECONDARY hedge; its chronic STALE_DATA/drawdown must NOT fail the
    # systemd unit every week (cry-wolf) while the crypto book is healthy.
    res = [R("mybot", "OK", 0.97, 100), R("otherbot", "OK", 0.96, 80),
           R("samplebot-c", "STALE_DATA", None, 0)]
    assert rpt.compute_exit_code(res, False, True) == 0


def test_exit_code_primary_stale_does_not_fail_unit() -> None:
    # STALE_DATA = the data window is frozen (needs a refresh), but the weekly alert STILL
    # delivered that warning (notify runs before exit). An alert-only unit must not flap to
    # `failed` on a routine data-maintenance gap — only a genuine ERROR (monitor machinery
    # broke) fails the unit. Severity fix 2026-07-25.
    res = [R("mybot", "STALE_DATA", None, 0), R("samplebot-c", "OK", 0.97, 50)]
    assert rpt.compute_exit_code(res, False, True) == 0


def test_exit_code_primary_error_still_fails_unit() -> None:
    # A genuine ERROR (exception / can't load) on the crypto primary IS a monitor failure —
    # the STALE relaxation must NOT swallow real breakage.
    res = [R("mybot", "ERROR", detail="boom"), R("samplebot-c", "OK", 0.97, 50)]
    assert rpt.compute_exit_code(res, False, True) == 1


def test_exit_code_secondary_only_run_still_fails_loud() -> None:
    # A targeted gold-only run has no crypto primary → fall back to treating it as
    # primary so a real gold ERROR still surfaces.
    assert rpt.compute_exit_code([R("samplebot-c", "ERROR", detail="x")], False, True) == 1


def test_alert_header_not_red_on_secondary_only_trip() -> None:
    # Chronic gold TRIPPED must not paint the weekly header red (habituation) while
    # the crypto book is OK — it caps the header at ⚠️, never green-washed.
    txt = rpt.build_alert([R("mybot", "OK", 0.97, 100), R("otherbot", "OK", 0.96, 80),
                           R("samplebot-c", "TRIPPED", 0.70, 60)])
    assert "\U0001f534" not in txt          # no red circle
    assert "⚠️" in txt
    assert "✅" not in txt                    # not green-washed either


def test_alert_header_red_on_primary_trip() -> None:
    txt = rpt.build_alert([R("mybot", "TRIPPED", 0.80, 100),
                           R("samplebot-c", "OK", 0.97, 50)])
    assert "\U0001f534" in txt


def test_alert_all_ok_is_green() -> None:
    txt = rpt.build_alert([R("mybot", "OK", 0.97, 100), R("samplebot-c", "OK", 0.96, 50)])
    assert "✅" in txt and "\U0001f534" not in txt


# ── .env parsing (MED/LOW row: quotes not stripped) ───────────────────────────

def test_parse_env_strips_double_quotes() -> None:
    assert rpt._parse_env_line('TELEGRAM_BOT_TOKEN="abc:123"') == ("TELEGRAM_BOT_TOKEN", "abc:123")


def test_parse_env_strips_single_quotes() -> None:
    assert rpt._parse_env_line("CHAT='42'") == ("CHAT", "42")


def test_parse_env_plain_and_embedded_equals() -> None:
    assert rpt._parse_env_line("PLAIN=x=y") == ("PLAIN", "x=y")


def test_parse_env_skips_comments_and_blanks() -> None:
    assert rpt._parse_env_line("# comment") is None
    assert rpt._parse_env_line("   ") is None
    assert rpt._parse_env_line("no_equals_here") is None


def test_load_env_applies_unquoted_values(monkeypatch, tmp_path) -> None:
    p = tmp_path / ".env"
    p.write_text('WFA_TEST_TOKEN_XYZ="secret-token"\n# c\nWFA_TEST_CHAT_XYZ=99\n')
    monkeypatch.delenv("WFA_TEST_TOKEN_XYZ", raising=False)
    monkeypatch.delenv("WFA_TEST_CHAT_XYZ", raising=False)
    rpt._load_env(str(p))
    import os
    assert os.environ["WFA_TEST_TOKEN_XYZ"] == "secret-token"
    assert os.environ["WFA_TEST_CHAT_XYZ"] == "99"
    monkeypatch.delenv("WFA_TEST_TOKEN_XYZ", raising=False)
    monkeypatch.delenv("WFA_TEST_CHAT_XYZ", raising=False)


# ── SCORING_DRIFT counter (MED row: drift suppressed invisibly) ───────────────

def test_warn_counter_counts_drift_only() -> None:
    c = rpt._WarnCounter()
    rec = logging.LogRecord("cc", logging.WARNING, "", 0,
                            "SCORING_DRIFT: CC tier=A vs quant_tier=B", None, None)
    other = logging.LogRecord("cc", logging.WARNING, "", 0, "some other warning", None, None)
    c.emit(rec)
    c.emit(rec)
    c.emit(other)
    assert c.drift == 2


def test_check_surfaces_drift_count(monkeypatch) -> None:
    class DriftAdapter(_StubAdapter):
        def run(self, params, start, end):
            rec = logging.LogRecord("cc", logging.WARNING, "", 0,
                                    "SCORING_DRIFT: parity", None, None)
            rpt.WARN_COUNTER.emit(rec)
            return super().run(params, start, end)

    monkeypatch.setattr(
        rpt, "load_adapter",
        lambda b: DriftAdapter(datetime.now() - timedelta(days=1)))
    r = rpt.check("mybot", 18.0, 0.95, 0.90)
    assert r.get("scoring_drift_warnings") == 1


# ── import-time cwd side effect (residual LOW: derived chdir still ran at import) ─

def test_import_does_not_change_cwd() -> None:
    # A fresh subprocess import from a cwd OTHER than the repo root: if the module
    # still chdirs at import time, the child process's cwd ends up at _ROOT instead
    # of staying put. In-process reload can't observe this reliably (the module is
    # already imported once at collection, and re-chdir to the same _ROOT is a
    # silent no-op), so exercise a real fresh interpreter.
    import subprocess
    import tempfile
    scripts_dir = str(Path(__file__).parent.parent / "scripts")
    engine_root = str(Path(__file__).parent.parent)
    with tempfile.TemporaryDirectory() as cwd:
        code = (
            f"import sys, os, json; "
            f"sys.path.insert(0, {engine_root!r}); "
            f"sys.path.insert(0, {scripts_dir!r}); "
            f"before = os.getcwd(); "
            f"import rolling_psr_tripwire; "
            f"print(json.dumps({{'before': before, 'after': os.getcwd()}}))"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=cwd,
                              capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        assert result["after"] == result["before"], (
            "importing rolling_psr_tripwire changed the process cwd")


# ── Telegram token scrubbing (residual LOW: token leaking into logged errors) ────

def test_send_telegram_scrubs_token_from_stderr_on_exception(monkeypatch, capsys) -> None:
    token = "123456:AAsecret-token-value"

    def fake_urlopen(url, data=None, timeout=None):
        # Simulate an exception whose str() embeds the full request URL (token included) —
        # the class of exception the finding warns can leak the token to the journal.
        raise OSError(f"Failed to reach {url}: connection reset")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ok = rpt.send_telegram(token, "chat", "msg", attempts=1, sleep=lambda s: None)
    assert ok is False
    err = capsys.readouterr().err
    assert token not in err
    assert "***" in err


# ── per-book split (2026-07-19 OtherBot↔CryptoDesk decouple, handoff Action Item) ──
# otherbot is now a SEPARATE live book, guard-judged (distribution-free), NOT part of a
# combined "crypto book" PSR verdict. mybot is the PRIMARY crypto edge (PSR-judged).

def test_classify_guard_ok() -> None:
    # enough trades, no drawdown/win-rate breach
    assert rpt.classify_guard(40, max_dd=0.10, win_rate=0.50) == "OK"


def test_classify_guard_low_n_is_a_state_not_a_failure() -> None:
    # structural LOW_N (otherbot's n<30) is its own non-alerting info state, not inconclusive
    assert rpt.classify_guard(24, max_dd=0.10, win_rate=0.50) == "LOW_N"


def test_classify_guard_dd_breach() -> None:
    assert rpt.classify_guard(40, max_dd=0.42, win_rate=0.50, dd_ceiling=0.35) == "DD_BREACH"


def test_classify_guard_dd_breach_trumps_low_n() -> None:
    # a real drawdown blowup fires even on few trades — that is the point of the guard
    assert rpt.classify_guard(8, max_dd=0.42, win_rate=0.50, dd_ceiling=0.35) == "DD_BREACH"


def test_classify_guard_wr_collapse() -> None:
    assert rpt.classify_guard(40, max_dd=0.10, win_rate=0.20,
                              wr_floor=0.30, wr_min_n=10) == "WR_COLLAPSE"


def test_classify_guard_wr_ignored_below_min_n() -> None:
    # too few trades for win-rate to mean anything => LOW_N, not a WR_COLLAPSE false-fire
    assert rpt.classify_guard(6, max_dd=0.10, win_rate=0.10,
                              wr_floor=0.30, wr_min_n=10) == "LOW_N"


def test_classify_guard_stale_trumps() -> None:
    assert rpt.classify_guard(40, max_dd=0.10, win_rate=0.50,
                              staleness_days=30.0, max_stale_days=8.0) == "STALE_DATA"


# ── check() routes guard bots to the distribution-free guard ──────────────────

class _GuardAdapter:
    """Adapter whose trades produce a >35% equity drawdown (peak 15k -> trough 9k)."""
    def __init__(self, de: datetime, pnls):
        self._de = de
        self._pnls = pnls

    def param_schema(self) -> dict:
        return {"x": {"default": 1}}

    def data_range(self):
        return self._de - timedelta(days=900), self._de

    def run(self, params, start, end) -> pd.DataFrame:
        return pd.DataFrame({"pnl": self._pnls})


def test_check_guard_bot_dd_breach(monkeypatch) -> None:
    pnls = [1000.0] * 5 + [-1500.0] * 4          # peak 15k, trough 9k => 40% DD
    monkeypatch.setattr(rpt, "load_adapter",
                        lambda b: _GuardAdapter(datetime.now() - timedelta(days=1), pnls))
    r = rpt.check("otherbot", 18.0, 0.95, 0.90)
    assert r["status"] == "DD_BREACH"
    assert r["psr"] is None                       # guard book is not PSR-judged
    assert r["max_dd"] > 0.35 and "win_rate" in r  # additive guard fields present


def test_check_guard_bot_low_n_when_quiet(monkeypatch) -> None:
    pnls = [10.0, -5.0] * 3                        # tiny, n=6, no breach
    monkeypatch.setattr(rpt, "load_adapter",
                        lambda b: _GuardAdapter(datetime.now() - timedelta(days=1), pnls))
    r = rpt.check("otherbot", 18.0, 0.95, 0.90)
    assert r["status"] == "LOW_N"
    assert r["psr"] is None


# ── the core false-alarm fix: otherbot LOW_N must not degrade the weekly header ──

def test_alert_otherbot_low_n_is_green_not_degraded() -> None:
    # THE regression this whole change fixes: mybot healthy + otherbot structurally
    # LOW_N must read GREEN, never "DEGRADED".
    txt = rpt.build_alert([R("mybot", "OK", 0.96, 119),
                           R("otherbot", "LOW_N", None, 24),
                           R("samplebot-c", "OK", 0.97, 60)])
    assert txt.startswith("✅")
    assert "DEGRADED" not in txt
    assert "recent crypto-MR edge intact" in txt


def test_alert_otherbot_dd_breach_is_red() -> None:
    # a real live-book drawdown blowup still fires
    txt = rpt.build_alert([R("mybot", "OK", 0.96, 119),
                           R("otherbot", "DD_BREACH", None, 24,
                             detail="maxDD 42% > 35% ceiling")])
    assert txt.startswith("\U0001f534")
    assert "42%" in txt


def test_alert_otherbot_wr_collapse_is_warn_not_green() -> None:
    txt = rpt.build_alert([R("mybot", "OK", 0.96, 119),
                           R("otherbot", "WR_COLLAPSE", None, 24,
                             detail="win-rate 20% < 30%")])
    assert "\U0001f534" not in txt
    assert "⚠️" in txt
    assert "✅" not in txt


def test_alert_guard_stale_detail_not_duplicated() -> None:
    # A guard bot has psr=None, so its body core IS r['detail']; the STALE_DATA re-append
    # must not print the detail twice on that same line.
    txt = rpt.build_alert([R("mybot", "OK", 0.96, 119),
                           R("otherbot", "STALE_DATA", None, 24,
                             detail="data ends 2026-07-02 (17d old > 8d)")])
    body = [ln for ln in txt.splitlines() if ln.startswith("[STALE_DATA] otherbot")][0]
    assert body.count("data ends 2026-07-02") == 1


def test_alert_footer_has_no_combined_crypto_book_verdict() -> None:
    # the "one crypto book" premise is dead: no joint verdict inheriting one bot's state
    # onto the other; each book reported separately.
    txt = rpt.build_alert([R("mybot", "OK", 0.96, 119),
                           R("otherbot", "LOW_N", None, 24)])
    assert "CRYPTO BOOK:" not in txt
    assert "OTHERBOT" in txt  # its own guard line
