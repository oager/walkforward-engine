"""Tests for wfa/folds.py."""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from wfa.folds import _add_months, generate_folds


def dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


# ── _add_months ────────────────────────────────────────────────────────────────

def test_add_months_basic() -> None:
    assert _add_months(dt(2020, 1), 3) == dt(2020, 4)


def test_add_months_year_rollover() -> None:
    assert _add_months(dt(2020, 11), 3) == dt(2021, 2)


def test_add_months_clamps_day_leap() -> None:
    base = datetime(2020, 1, 31, tzinfo=UTC)
    result = _add_months(base, 1)
    assert result.day == 29  # 2020 is leap year


def test_add_months_feb_non_leap() -> None:
    base = datetime(2021, 1, 31, tzinfo=UTC)
    result = _add_months(base, 1)
    assert result.day == 28


# ── generate_folds ─────────────────────────────────────────────────────────────

def test_basic_fold_count_12_3() -> None:
    # 2018-01 to 2020-01 = 24 months. 12mo train + 3mo test → walks by 3mo.
    # Folds until test_end <= 2020-01:
    # F0: train 2018-01→2019-01, test 2019-01→2019-04
    # F1: train 2018-04→2019-04, test 2019-04→2019-07
    # F2: train 2018-07→2019-07, test 2019-07→2019-10
    # F3: train 2018-10→2019-10, test 2019-10→2020-01 ← test_end == data_end ✓
    folds = generate_folds(dt(2018), dt(2020), train_months=12, test_months=3)
    assert len(folds) == 4


def test_fold_indices_sequential() -> None:
    folds = generate_folds(dt(2018), dt(2022), 12, 3)
    assert [f.index for f in folds] == list(range(len(folds)))


def test_no_test_overlap_between_folds() -> None:
    folds = generate_folds(dt(2018), dt(2022), 12, 3)
    for i in range(len(folds) - 1):
        assert folds[i].test_end <= folds[i + 1].test_start, (
            f"Fold {i} test_end {folds[i].test_end} overlaps fold {i+1} test_start {folds[i+1].test_start}"
        )


def test_train_end_equals_test_start() -> None:
    folds = generate_folds(dt(2018), dt(2022), 12, 3)
    for f in folds:
        assert f.train_end == f.test_start


def test_first_fold_dates() -> None:
    folds = generate_folds(dt(2018), dt(2020), 12, 3)
    first = folds[0]
    assert first.train_start == dt(2018)
    assert first.train_end == dt(2019)
    assert first.test_start == dt(2019)
    assert first.test_end == datetime(2019, 4, 1, tzinfo=UTC)


def test_incomplete_final_fold_dropped() -> None:
    # data_end at 2019-03-31 — test window 2019-01→2019-04 would overshoot
    folds = generate_folds(dt(2018), datetime(2019, 3, 31, tzinfo=UTC), 12, 3)
    assert len(folds) == 0


def test_exact_boundary_included() -> None:
    folds = generate_folds(dt(2018), dt(2019, 4, 1), 12, 3)
    assert len(folds) == 1


def test_range_too_short_for_one_fold() -> None:
    folds = generate_folds(dt(2020), dt(2020, 6), 12, 3)
    assert folds == []


def test_single_fold_18_6() -> None:
    folds = generate_folds(dt(2018), dt(2020), 18, 6)
    assert len(folds) == 1
    assert folds[0].train_start == dt(2018)
    assert folds[0].test_end == dt(2020)


def test_naive_datetime_treated_as_utc() -> None:
    folds_naive = generate_folds(datetime(2018, 1, 1), datetime(2020, 1, 1), 12, 3)
    folds_aware = generate_folds(dt(2018), dt(2020), 12, 3)
    assert len(folds_naive) == len(folds_aware)


def test_invalid_months_raises() -> None:
    with pytest.raises(ValueError):
        generate_folds(dt(2018), dt(2020), 0, 3)
    with pytest.raises(ValueError):
        generate_folds(dt(2018), dt(2020), 12, 0)


def test_inverted_range_raises() -> None:
    with pytest.raises(ValueError):
        generate_folds(dt(2020), dt(2018), 12, 3)


def test_walk_forward_step_equals_test_months() -> None:
    folds = generate_folds(dt(2018), dt(2022), 12, 3)
    for i in range(len(folds) - 1):
        delta = folds[i + 1].train_start - folds[i].train_start
        # each cursor step = test_months = 3 months ≈ 89–92 days
        assert 85 <= delta.days <= 95, f"Unexpected step {delta.days} days between fold {i} and {i+1}"


# ── day-clamping must not overlap consecutive test windows (review MED) ───────

def test_month_end_start_no_test_overlap_or_gap() -> None:
    # data_start on the 30th walks the cursor through February: the old iterated
    # cursor clamped to Feb-28 and stayed there, putting fold1.test_start one day
    # BEFORE fold0.test_end (2020-02-28 < 2020-02-29) — duplicated OOS trades.
    folds = generate_folds(datetime(2018, 11, 30, tzinfo=UTC), dt(2022), 12, 3)
    assert len(folds) >= 2
    for i in range(len(folds) - 1):
        assert folds[i].test_end == folds[i + 1].test_start, (
            f"fold {i} test_end {folds[i].test_end} != fold {i+1} test_start "
            f"{folds[i + 1].test_start} — stitched OOS would duplicate or drop trades"
        )


def test_month_end_start_windows_tile_for_31st() -> None:
    folds = generate_folds(datetime(2019, 1, 31, tzinfo=UTC), dt(2023), 12, 3)
    assert len(folds) >= 2
    for i in range(len(folds) - 1):
        assert folds[i].test_end == folds[i + 1].test_start


# ── purge/embargo (opt-in, default 0 = unchanged behaviour) ───────────────────

def test_purge_days_trims_train_end_only() -> None:
    base = generate_folds(dt(2018), dt(2022), 12, 3)
    purged = generate_folds(dt(2018), dt(2022), 12, 3, purge_days=3)
    assert len(base) == len(purged)
    for b, p in zip(base, purged):
        assert p.train_start == b.train_start
        assert p.test_start == b.test_start
        assert p.test_end == b.test_end
        assert (p.test_start - p.train_end).days == 3


def test_purge_days_default_zero_keeps_train_end_equal_test_start() -> None:
    for f in generate_folds(dt(2018), dt(2022), 12, 3):
        assert f.train_end == f.test_start


def test_purge_days_negative_raises() -> None:
    with pytest.raises(ValueError):
        generate_folds(dt(2018), dt(2022), 12, 3, purge_days=-1)


def test_purge_days_swallowing_train_window_raises() -> None:
    with pytest.raises(ValueError):
        generate_folds(dt(2018), dt(2022), 12, 3, purge_days=400)
