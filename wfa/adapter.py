"""BacktestAdapter Protocol — contract every bot adapter must implement."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import pandas as pd


ParamType = Literal["int", "float", "categorical"]


class ParamSpec(TypedDict):
    type: ParamType
    choices: list
    default: object
    help: str


class BacktestAdapter(Protocol):
    """Interface for a bot's backtest engine, used by the WFA runner."""

    bot_name: str    # e.g. "mybot"
    timeframe: str   # e.g. "4h"

    def param_schema(self) -> dict[str, ParamSpec]:
        """Return dict of param name → spec with choices, default, and help."""
        ...

    def recommended_windows(self) -> tuple[int, int]:
        """Return (train_months, test_months) appropriate for this strategy."""
        ...

    def default_objective(self) -> str:
        """Return the preferred objective key (e.g. "sortino")."""
        ...

    def data_snapshot_path(self) -> Path:
        """Return path to frozen OHLC data used for backtests.

        Must be a regular file or directory — never a live API endpoint.
        """
        ...

    def data_range(self) -> tuple[datetime, datetime]:
        """Return (first_available_bar, last_available_bar) UTC."""
        ...

    def run(
        self,
        params: dict,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Run a backtest for the given params over [start, end).

        Returns a trades DataFrame with at minimum these columns:
            trade_id, symbol, side, entry_time, entry_price,
            exit_time, exit_price, size, pnl, fees, status

        Schema mirrors results/trades.csv in the mybot repo.
        """
        ...
