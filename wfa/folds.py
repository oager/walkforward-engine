"""Rolling walk-forward fold generator.

Produces non-overlapping (train, test) date windows. Each fold:
  - train: [fold_start, fold_start + train_months)
  - test:  [fold_start + train_months, fold_start + train_months + test_months)

Folds walk forward by test_months each step. Incomplete final fold (test window
extends past data_end) is dropped.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: datetime
    train_end: datetime   # exclusive
    test_start: datetime
    test_end: datetime    # exclusive


def generate_folds(
    data_start: datetime,
    data_end: datetime,
    train_months: int,
    test_months: int,
    purge_days: int = 0,
) -> list[Fold]:
    """Generate walk-forward folds within [data_start, data_end).

    All boundaries are month offsets from `data_start` (NOT iterated cursor steps):
    iterating `_add_months` from a day-clamped cursor (e.g. data_start on the 30th
    stepping through February) made fold k+1's test_start land BEFORE fold k's
    test_end — duplicating trades in the stitched OOS stream. Anchoring every
    boundary to data_start guarantees test_end(k) == test_start(k+1) exactly.

    Args:
        data_start: First available data timestamp (inclusive).
        data_end:   Last available data timestamp (exclusive).
        train_months: Number of months in the training window per fold.
        test_months:  Number of months in the blind-test window per fold.
        purge_days:   Optional embargo — trims this many days off the END of each
                      train window so trades straddling the train/test boundary
                      cannot leak IS information into OOS. Default 0 (no purge).

    Returns:
        List of Fold objects. Empty if data range is too short for even one fold.
    """
    if train_months <= 0 or test_months <= 0:
        raise ValueError(
            f"train_months and test_months must be > 0, got {train_months}/{test_months}"
        )
    if purge_days < 0:
        raise ValueError(f"purge_days must be >= 0, got {purge_days}")

    data_start = _ensure_utc(data_start)
    data_end = _ensure_utc(data_end)

    if data_start >= data_end:
        raise ValueError(f"data_start ({data_start}) must be before data_end ({data_end})")

    folds: list[Fold] = []
    fold_idx = 0

    while True:
        offset = fold_idx * test_months
        train_start = _add_months(data_start, offset)
        test_start = _add_months(data_start, offset + train_months)
        test_end = _add_months(data_start, offset + train_months + test_months)
        train_end = test_start - timedelta(days=purge_days)

        if test_end > data_end:
            break  # incomplete test window — drop

        if train_end <= train_start:
            raise ValueError(
                f"purge_days={purge_days} leaves no training window "
                f"({train_start} .. {train_end})"
            )

        folds.append(Fold(
            index=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ))
        fold_idx += 1

    return folds


def _add_months(dt: datetime, months: int) -> datetime:
    total_months = dt.month - 1 + months
    year = dt.year + total_months // 12
    month = total_months % 12 + 1
    max_day = calendar.monthrange(year, month)[1]
    day = min(dt.day, max_day)
    return dt.replace(year=year, month=month, day=day)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
