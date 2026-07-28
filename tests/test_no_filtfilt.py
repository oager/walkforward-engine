"""CI guard: wfa package must not use scipy.signal.filtfilt (lookahead bias).

The YouTube walkthrough (12:09) explicitly warns: always use lfilter (causal),
never filtfilt (zero-phase = acausal = lookahead). This test fails the build
if filtfilt is ever introduced into the wfa package.
"""
from __future__ import annotations

import re
from pathlib import Path

_WFA_DIR = Path(__file__).parent.parent / "wfa"


def test_no_filtfilt_in_wfa_package() -> None:
    pattern = re.compile(r"\bfiltfilt\b")
    violations: list[str] = []
    for py_file in _WFA_DIR.rglob("*.py"):
        text = py_file.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                violations.append(f"{py_file.relative_to(_WFA_DIR.parent)}:{lineno}: {line.strip()}")

    assert not violations, (
        "filtfilt detected in wfa package (lookahead bias — use lfilter instead):\n"
        + "\n".join(violations)
    )
