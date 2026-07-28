#!/usr/bin/env python3
"""Rolling-PSR regime tripwire (wfa-engine ops tool).

For each bot, backtest its prod (adapter-default) config over the TRAILING N months — a single fixed
-config window, not full WFA, so it's fast — and run the survival PSR check. Emits OK / WARN / TRIPPED
so a cron can alert when a bot's recent-edge PSR drifts below the floor: the favorable regime has
likely turned. PER-BOOK verdicts (OtherBot↔CryptoDesk split 2026-07-19): mybot is the PRIMARY
crypto edge, PSR-judged; otherbot is a separate live book, GUARD-judged distribution-free (see below);
gold breakout is the complementary SECONDARY hedge (its edge was 2023-24, in drawdown 2025). The old
"one combined crypto book" premise is dead — the books may diverge in strategy, so neither inherits the
other's verdict.

  python rolling_psr_tripwire.py [--bots otherbot mybot samplebot-c] [--months 18]
                                 [--floor 0.95] [--warn 0.90] [--max-stale-days 8] [--json]
Status: OK (PSR>=floor) · WARN (warn<=PSR<floor, margin thinning) · TRIPPED (PSR<warn, regime turned)
        · LOW_N (<30 trades, inconclusive) · NO_PSR (PSR not computable) · ERROR (adapter failed)
        · STALE_DATA (newest cached bar older than --max-stale-days — frozen window, PSR not a verdict).
GUARD books (otherbot) are not PSR-judged — they are too selective to reach n>=30, so PSR is chronically
inconclusive. Instead a distribution-free guard: DD_BREACH (trailing max-drawdown over ceiling — a real
live-money blowup, reds the header) · WR_COLLAPSE (win-rate under floor, ⚠️) · LOW_N here is a NON-ALERTING
info state (structural, by design), never DEGRADED. Guard thresholds: GUARD_MAX_DD / GUARD_MIN_WR / GUARD_WR_MIN_N.

ROLES: header severity and the exit code key off PRIMARY (mybot) health. A GUARD DD_BREACH also reds
the header (live money), but a GUARD ERROR/STALE/LOW_N never fails the systemd unit (it is not the crypto
edge the monitor exists to watch). A non-OK SECONDARY (gold, in EXPECTED 2025 drawdown) is shown in the
body and caps the header at ⚠️ (never green-washed, never 🔴 on its own) — a chronically-red header +
chronically-failed unit habituates the operator to ignore both (re-review 2026-07-02). If no primary bot
is in --bots, all bots are treated as primary.
Exit codes: 0 = ran + delivered (incl. TRIPPED/WARN, secondary ERROR/STALE, and a PRIMARY
                STALE_DATA — the "refresh me" warning still went out; a data-maintenance gap
                must not flap this alert-only unit into `failed`);
            1 = a PRIMARY (crypto-book) bot is ERROR (the monitor's own machinery broke);
            2 = --notify requested but Telegram delivery failed (alert printed to stdout instead).
"""
import os, sys, json, time, argparse, logging
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)   # derive repo root from this file, not a hardcoded home
from datetime import datetime, timedelta, timezone
from wfa.registry import load_adapter
from wfa.survival import evaluate_survival, SurvivalThresholds
from wfa.metrics import build_equity_curve, max_drawdown

DEFAULT_BOTS = ["mybot", "samplebot-c", "otherbot"]

# Per-book roles (OtherBot↔CryptoDesk split, 2026-07-19). The two books were decoupled —
# CryptoDesk changes strategy freely; OtherBot keeps its own. So neither inherits the other's
# verdict (the old combined "crypto book" premise is dead).
#   PRIMARY  = the crypto edge the monitor EXISTS to watch; PSR-judged; drives header + exit code.
#   GUARD    = a live book too selective for PSR (otherbot: n<30 structurally). NOT PSR-judged —
#              a distribution-free drawdown / win-rate guard so it can still fire on a real blowup
#              while its structural LOW_N reads as a non-alerting state, not a weekly false alarm.
#   SECONDARY= complementary hedge (gold, expected 2025 drawdown); shown for context, caps header
#              at ⚠️, never reds it alone, never green-washed.
PRIMARY_BOTS = ("mybot",)
GUARD_BOTS = ("otherbot",)

# Bots whose data is frozen ON PURPOSE. For these, an old newest-bar is the DESIGN, not a
# maintenance gap — so staleness must never be reported as STALE_DATA ("refresh me"), because the
# correct response is the opposite: refreshing DESTROYS the artifact.
#   liquid-swing: pre-registered one-shot walk-forward study. GATE_REPORT_V2.md lists `data` in the
#   explicitly-unchanged set ("frozen daily OHLC, 13 symbols, 2018 -> 2026-07 — unchanged from v1";
#   "One shot, verdict binding"), and determinism was reproduced against those exact CSVs
#   (n_trades=736, calmar=0.596). scripts/freeze_data.py has NO end date — it pulls to today — so a
#   well-meaning "fix the stale data" silently re-dates the study and voids the pre-registration.
# NOTE these bots are not in DEFAULT_BOTS, so the scheduled run never touches them; this guard
# exists for ad-hoc `--bots liquid-swing` invocations, which is exactly how the false alarm arose
# (2026-07-25). Add a bot here ONLY with a pointer to the artifact that pins its data.
FROZEN_STUDY_BOTS = {
    "liquid-swing": "pre-registered one-shot study — see ~/liquid-swing/GATE_REPORT_V2.md; "
                    "refreshing breaks the pre-registration and voids the reproducibility check",
}

# Guard thresholds (distribution-free; tune via these module constants). Conservative on purpose so the guard
# trades one false alarm (chronic LOW_N=DEGRADED) for zero, not for another:
#   GUARD_MAX_DD — trailing max-drawdown ceiling; matches SurvivalThresholds.max_drawdown_pct.
#   GUARD_MIN_WR — win-rate floor; low for a momentum book where sub-40% WR is normal/healthy.
#   GUARD_WR_MIN_N — win-rate is meaningless below this many trades. TUNE once otherbot's live
#   baseline is characterized (n was 24 at split).
GUARD_MAX_DD = 0.35
GUARD_MIN_WR = 0.30
GUARD_WR_MIN_N = 10

BAD_STATUSES = ("TRIPPED", "WARN")                                  # real edge signal
INCONCLUSIVE_STATUSES = ("ERROR", "LOW_N", "NO_PSR", "STALE_DATA")  # broken/insufficient input
ENV_PATH = "~/cron-scripts/.env"


def _role(bot):
    if bot in GUARD_BOTS:
        return "guard"
    if bot in PRIMARY_BOTS:
        return "primary"
    return "secondary"

# Per-bot override when an adapter's param_schema default drifts from the DEPLOYED config.
# Empty now: otherbot's stale absorption_bonus default (5.0) was fixed to 0.0 in
# backtest_adapter.py (2026-06-30), so all adapter defaults now match deployed configs.
# Add {bot: {param: deployed_value}} here only if a future default drifts again.
CONFIG_OVERRIDES = {}


class _WarnCounter(logging.Handler):
    """Swallow WARNING records (per-bar SCORING_DRIFT parity spam = huge I/O = slow) but COUNT
    the SCORING_DRIFT ones so a stale scorer is surfaced in the alert instead of silently
    suppressed; ERROR+ records still go to stderr."""
    def __init__(self):
        super().__init__(logging.WARNING)
        self.drift = 0

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        if "SCORING_DRIFT" in msg:
            self.drift += 1
        elif record.levelno >= logging.ERROR:
            sys.stderr.write(msg + "\n")


WARN_COUNTER = _WarnCounter()


def _setup_logging():
    logging.disable(logging.INFO)                  # silence INFO/DEBUG entirely
    logging.getLogger().addHandler(WARN_COUNTER)   # count-don't-print WARNING+ (no stream I/O)


def prod_params(a, bot):
    out = {k: v["default"] for k, v in a.param_schema().items()
           if isinstance(v, dict) and "default" in v}
    out.update(CONFIG_OVERRIDES.get(bot, {}))
    return out


def stale_note(de):
    """Human phrasing for a pinned dataset's newest bar — used in the FROZEN_STUDY_BOTS note so the
    age is still visible without implying it is a defect."""
    return f"data ends {de.date()}, {_staleness_days(de):.0f}d ago"


def _staleness_days(de, now=None):
    """Days between the newest data bar and wall-clock now (tz-naive-safe)."""
    if now is None:
        now = datetime.now(timezone.utc)
    if getattr(de, "tzinfo", None) is None:
        now = now.replace(tzinfo=None)
    return (now - de).total_seconds() / 86400.0


def classify_status(psr, n, floor, warn, staleness_days=None, max_stale_days=8.0, frozen=False):
    """Pure status classification. STALE_DATA trumps everything: an OK PSR computed off a
    frozen data window is not a verdict about the current regime.

    `frozen=True` (see FROZEN_STUDY_BOTS) suppresses that override: the data is pinned by design,
    so its age is not a maintenance gap and must not read as "refresh me". The PSR still classifies
    normally — it remains the study's own verdict, just not a current-regime one."""
    if not frozen and staleness_days is not None and staleness_days > max_stale_days:
        return "STALE_DATA"
    if n < 30:
        return "LOW_N"
    if psr is None:
        return "NO_PSR"
    if psr < warn:
        return "TRIPPED"
    if psr < floor:
        return "WARN"
    return "OK"


def classify_guard(n, max_dd, win_rate, staleness_days=None, max_stale_days=8.0,
                   dd_ceiling=GUARD_MAX_DD, wr_floor=GUARD_MIN_WR, wr_min_n=GUARD_WR_MIN_N,
                   frozen=False):
    """Distribution-free verdict for a GUARD book (otherbot) that is structurally too selective
    to reach the PSR n>=30 floor. STALE_DATA still trumps (can't guard off a frozen window). A
    real drawdown breach fires even on few trades — that is the guard's job. Win-rate collapse
    needs a minimum n to be meaningful. Otherwise structural LOW_N is a non-alerting info state,
    NOT an inconclusive-monitor failure. `frozen=True` suppresses the staleness override for a
    deliberately pinned dataset (see FROZEN_STUDY_BOTS)."""
    if not frozen and staleness_days is not None and staleness_days > max_stale_days:
        return "STALE_DATA"
    if max_dd is not None and max_dd > dd_ceiling:
        return "DD_BREACH"
    if n >= wr_min_n and win_rate is not None and win_rate < wr_floor:
        return "WR_COLLAPSE"
    if n < 30:
        return "LOW_N"
    return "OK"


def _check_guard(bot, trades, de, months, floor, warn, start, max_stale_days, frozen=False):
    """Guard-book branch of check(): drawdown/win-rate instead of PSR."""
    n = int(len(trades))
    mdd = max_drawdown(build_equity_curve(trades)) if n else None
    win_rate = float((trades["pnl"] > 0).mean()) if n and "pnl" in trades.columns else None
    stale = _staleness_days(de)
    status = classify_guard(n, mdd, win_rate, staleness_days=stale, max_stale_days=max_stale_days,
                            frozen=frozen)
    out = {"bot": bot, "window": f"{start.date()}..{de.date()}", "months": months,
           "n": n, "psr": None, "floor": floor, "warn": warn, "status": status,
           "survival_pass": None, "staleness_days": round(stale, 1),
           "max_dd": round(mdd, 4) if mdd is not None else None,
           "win_rate": round(win_rate, 4) if win_rate is not None else None}
    if status == "STALE_DATA":
        out["detail"] = (f"data ends {de.date()} ({stale:.0f}d old > {max_stale_days:g}d) — "
                         "window frozen, guard cannot verdict")
    elif status == "DD_BREACH":
        out["detail"] = (f"maxDD {mdd:.0%} > {GUARD_MAX_DD:.0%} ceiling — live-book drawdown breach")
    elif status == "WR_COLLAPSE":
        out["detail"] = f"win-rate {win_rate:.0%} < {GUARD_MIN_WR:.0%} floor (n={n})"
    elif status == "LOW_N":
        out["detail"] = f"n={n} below PSR floor by design — no drawdown/win-rate breach"
    return out


def check(bot, months, floor, warn, max_stale_days=8.0):
    drift0 = WARN_COUNTER.drift
    try:
        a = load_adapter(bot)
        params = prod_params(a, bot)
        ds, de = a.data_range()
        start = max(de - timedelta(days=int(months * 30.44)), ds)
        trades = a.run(dict(params), start, de)
    except Exception as e:
        return {"bot": bot, "status": "ERROR", "detail": f"{type(e).__name__}: {e}"}

    frozen = bot in FROZEN_STUDY_BOTS

    if _role(bot) == "guard":
        out = _check_guard(bot, trades, de, months, floor, warn, start, max_stale_days,
                           frozen=frozen)
    else:
        v = evaluate_survival(trades, degradation=None, thresholds=SurvivalThresholds(min_psr=floor))
        psr, n = v.psr, v.n_trades
        stale = _staleness_days(de)
        status = classify_status(psr, n, floor, warn,
                                 staleness_days=stale, max_stale_days=max_stale_days,
                                 frozen=frozen)
        out = {"bot": bot, "window": f"{start.date()}..{de.date()}", "months": months,
               "n": n, "psr": round(psr, 4) if psr is not None else None,
               "floor": floor, "warn": warn, "status": status, "survival_pass": v.passed,
               "staleness_days": round(stale, 1)}
        if status == "STALE_DATA":
            out["detail"] = (f"data ends {de.date()} ({stale:.0f}d old > {max_stale_days:g}d) — "
                             "window frozen, PSR is NOT a current-regime verdict")

    # A frozen study's old data is the design. Say so loudly enough that nobody "fixes" it, and
    # keep the caveat that its verdict is historical rather than a read on the current regime.
    if frozen:
        out["frozen_study"] = True
        note = (f"DATA PINNED BY DESIGN ({stale_note(de)}) — DO NOT REFRESH: "
                f"{FROZEN_STUDY_BOTS[bot]}. Verdict is the study's own, not a current-regime read.")
        out["detail"] = f"{out['detail']} | {note}" if out.get("detail") else note

    drift = WARN_COUNTER.drift - drift0
    if drift:
        out["scoring_drift_warnings"] = drift
    return out


def _primaries(results):
    """The bots the tripwire exists to watch (the crypto edge = PRIMARY_BOTS). If a targeted
    run has none (e.g. --bots samplebot-c only), fall back to treating all as primary so a
    targeted secondary/guard run still fails loud on a real error."""
    prim = [r for r in results if _role(r["bot"]) == "primary"]
    return prim if prim else results


def _severity(r):
    """Header contribution of one result, role-aware: 'red' | 'warn' | 'ok'."""
    role, s = _role(r["bot"]), r["status"]
    drift = bool(r.get("scoring_drift_warnings"))
    if role == "primary":
        if s in ("TRIPPED", "ERROR", "STALE_DATA"):
            return "red"                                       # crypto edge turned / monitor broke
        if s in ("WARN", "LOW_N", "NO_PSR") or drift:
            return "warn"
        return "ok"
    if role == "guard":
        if s == "DD_BREACH":
            return "red"                                       # live-money drawdown blowup
        if s in ("WR_COLLAPSE", "STALE_DATA", "ERROR") or drift:
            return "warn"
        return "ok"                                            # OK / LOW_N — structural, non-alerting
    # secondary hedge (gold): never red on its own, never green-washed
    if s != "OK" or drift:
        return "warn"
    return "ok"


def book_lines(results):
    """Per-book verdict footer. The books were split (2026-07-19): each reports on its own,
    neither inherits the other's state. PRIMARY (mybot) = the crypto edge, PSR-judged;
    GUARD (otherbot) = a live book judged distribution-free. SECONDARY (gold) has no footer
    line — its body row suffices."""
    lines = []
    for r in results:
        role, s = _role(r["bot"]), r["status"]
        detail = r.get("detail", "")
        if role == "primary":
            if s == "OK":
                lines.append(f"CRYPTO EDGE ({r['bot']}): OK — recent crypto-MR edge intact")
            elif s in BAD_STATUSES:
                lines.append(f"CRYPTO EDGE ({r['bot']}): ⚠️ {s} — PSR below floor, regime may be turning")
            else:
                lines.append(f"CRYPTO EDGE ({r['bot']}): ❓ {s} — no current-regime verdict "
                             "(monitor input incomplete)")
        elif role == "guard":
            label = f"{r['bot'].upper()} (live-book guard):"
            if s == "DD_BREACH":
                lines.append(f"{label} 🔴 DD_BREACH — {detail}".rstrip(" —"))
            elif s == "WR_COLLAPSE":
                lines.append(f"{label} ⚠️ WR_COLLAPSE — {detail}".rstrip(" —"))
            elif s in ("OK", "LOW_N"):
                lines.append(f"{label} OK — {detail or 'no drawdown/win-rate breach'}")
            else:
                lines.append(f"{label} ⚠️ {s} — {detail}".rstrip(" —"))
    return lines


def build_alert(results):
    """Render the weekly Telegram message. Header severity is role-aware (per-book split
    2026-07-19): a PRIMARY (mybot) turn/break OR a GUARD (otherbot) drawdown blowup reds
    it; a structural guard LOW_N or a SECONDARY (gold) drawdown only caps at ⚠️, never reds it
    alone, never green-washes. Green banner ONLY when every book is effectively OK and no
    scoring drift fired. DEGRADED fires ONLY when the PRIMARY itself can't produce a verdict
    (its own job broke) — a structurally-LOW_N otherbot no longer degrades the header."""
    sev = [_severity(r) for r in results]
    if "red" in sev:
        head = "\U0001f534"
    elif "warn" in sev:
        head = "⚠️"
    else:
        head = "✅"
    prim = _primaries(results)
    prim_incomplete = any(
        r["status"] in ("ERROR", "STALE_DATA", "LOW_N", "NO_PSR") or r.get("scoring_drift_warnings")
        for r in prim)
    title = head + " WFA rolling-PSR tripwire (weekly)"
    if prim_incomplete:
        title += " — DEGRADED (monitor incomplete)"
    elif head != "✅":
        title += " — see below"
    lines = [title]
    for r in results:
        if r.get("psr") is not None:
            core = f"PSR={r['psr']:.3f} n={r.get('n', '?')} {r.get('window', '')}".rstrip()
        else:
            core = r.get("detail", "?")
        line = f"[{r['status']}] {r['bot']}: {core}"
        if r["status"] == "STALE_DATA" and r.get("detail") and r.get("psr") is not None:
            line += f" — {r['detail']}"   # psr bots: core is the PSR line; guard core already = detail
        if r.get("scoring_drift_warnings"):
            line += f" ⚠️ scoring_drift={r['scoring_drift_warnings']}"
        lines.append(line)
    lines.extend(book_lines(results))
    return "\n".join(lines)


def _parse_env_line(line):
    """Parse one .env line into (key, value) or None; strips matching quotes around the value."""
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    k, v = line.split("=", 1)
    k, v = k.strip(), v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        v = v[1:-1]
    return k, v


def _load_env(path):
    if not os.path.exists(path):
        return
    for line in open(path):
        kv = _parse_env_line(line)
        if kv:
            os.environ.setdefault(kv[0], kv[1])  # real env wins (systemd env is clean)


def send_telegram(token, chat, text, attempts=3, backoff_s=2.0, sleep=time.sleep):
    """POST sendMessage with bounded retry; True only when Telegram confirmed ok=true."""
    import urllib.request, urllib.parse
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for i in range(attempts):
        if i:
            sleep(backoff_s * i)
        try:
            with urllib.request.urlopen(url, data=data, timeout=15) as resp:
                body = json.loads(resp.read().decode() or "{}")
            if body.get("ok"):
                return True
            last = f"telegram responded ok=false: {body.get('description', body)}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}".replace(token, "***")
        sys.stderr.write(f"tripwire notify attempt {i + 1}/{attempts} failed: {last}\n")
    return False


def _notify_telegram(results):
    """Self-deliver the weekly status to Telegram (cron-alert channel). Mirrors
    mybot-health-check: loads ~/cron-scripts/.env, posts via sendMessage.
    Returns True only on confirmed delivery; on failure prints the full alert to stdout
    so a dropped weekly alert is visible in the journal."""
    _load_env(os.path.expanduser(ENV_PATH))
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("EXAMPLEBOT_HEALTH_TG_CHAT")
    text = build_alert(results)
    if not token or not chat:
        sys.stderr.write("tripwire: TELEGRAM_BOT_TOKEN/EXAMPLEBOT_HEALTH_TG_CHAT not set\n")
        print("UNDELIVERED ALERT:\n" + text)
        return False
    if send_telegram(token, chat, text):
        return True
    print("UNDELIVERED ALERT:\n" + text)
    return False


def compute_exit_code(results, notify_requested, notify_ok):
    """0 = ran + delivered (TRIPPED/WARN are valid verdicts; a SECONDARY ERROR/STALE is
    expected context; a PRIMARY STALE_DATA is a delivered "refresh me" warning, not a unit
    failure — the alert-only unit must not flap to `failed` on a routine data-maintenance
    gap); 1 = a PRIMARY (crypto-book) bot is ERROR (the monitor's own machinery broke —
    exception / can't load); 2 = requested Telegram delivery failed. (Severity fix
    2026-07-25: STALE_DATA dropped from the fail set — notify runs before exit, so the
    stale warning is delivered regardless; only genuine ERROR fails the unit.)"""
    if notify_requested and not notify_ok:
        return 2
    if any(r["status"] == "ERROR" for r in _primaries(results)):
        return 1
    return 0


def main():
    os.chdir(_ROOT)  # cwd side effect confined to actual execution, not import
    _setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--bots", nargs="*", default=DEFAULT_BOTS)
    ap.add_argument("--months", type=float, default=18.0)
    ap.add_argument("--floor", type=float, default=0.95)
    ap.add_argument("--warn", type=float, default=0.90)
    ap.add_argument("--max-stale-days", type=float, default=8.0,
                    help="newest data bar older than this => STALE_DATA (weekly cadence)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--notify", action="store_true", help="post weekly status to Telegram")
    args = ap.parse_args()

    results = [check(b, args.months, args.floor, args.warn, args.max_stale_days)
               for b in args.bots]
    notify_ok = True
    if args.notify:
        notify_ok = _notify_telegram(results)
    rc = compute_exit_code(results, args.notify, notify_ok)
    if args.json:
        print(json.dumps({"tripwire": results}, indent=2))
        sys.exit(rc)
    print(f"ROLLING-PSR TRIPWIRE  (trailing {args.months:.0f}mo, floor {args.floor}, warn {args.warn})")
    for r in results:
        line = f"  [{r['status']:>10}] {r['bot']:16}"
        if r.get("psr") is not None:
            line += f" PSR={r['psr']:.3f} n={r['n']}  {r.get('window','')}"
        else:
            line += f" {r.get('detail', r.get('window',''))}"
        if r["status"] == "STALE_DATA" and r.get("detail"):
            line += f"  {r['detail']}"
        if r.get("scoring_drift_warnings"):
            line += f"  ⚠️ scoring_drift={r['scoring_drift_warnings']}"
        print(line)
    footer = book_lines(results)
    if footer:
        print("\n" + "\n".join(footer))
    sys.exit(rc)


if __name__ == "__main__":
    main()
