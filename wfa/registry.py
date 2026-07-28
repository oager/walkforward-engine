"""Adapter registry — loads bot BacktestAdapter instances from ~/.wfa/config.toml."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from wfa.adapter import BacktestAdapter

_DEFAULT_CONFIG = Path.home() / ".wfa" / "config.toml"


def load_adapter(bot_name: str, config_path: Path = _DEFAULT_CONFIG) -> BacktestAdapter:
    """Load and return the BacktestAdapter for *bot_name*.

    Config format (config.toml):
        [adapters]
        mybot   = "/home/user/mybot/backtest_adapter.py"
        samplebot-a    = "/home/user/samplebot_a/backtest_adapter.py"
        samplebot-b   = "/home/user/Index Trading Bot/backtest_adapter.py"
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"WFA config not found: {config_path}\n"
            "Create ~/.wfa/config.toml with [adapters] section pointing to each bot's backtest_adapter.py"
        )

    with config_path.open("rb") as fh:
        config = tomllib.load(fh)

    adapters_section = config.get("adapters", {})
    if bot_name not in adapters_section:
        available = list(adapters_section.keys())
        raise KeyError(
            f"Bot '{bot_name}' not found in {config_path}. Available: {available}"
        )

    adapter_path = Path(adapters_section[bot_name]).expanduser()
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter file not found: {adapter_path}")

    return _load_adapter_from_path(adapter_path)


def _load_adapter_from_path(path: Path) -> BacktestAdapter:
    """Dynamically import a backtest_adapter.py and return its Adapter instance."""
    module_name = f"_wfa_adapter_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if not hasattr(module, "Adapter"):
        raise AttributeError(
            f"Adapter module {path} must define a class named 'Adapter' "
            "that implements the BacktestAdapter protocol."
        )
    return module.Adapter()


def list_bots(config_path: Path = _DEFAULT_CONFIG) -> list[str]:
    """Return all registered bot names from the config."""
    if not config_path.exists():
        return []
    with config_path.open("rb") as fh:
        config = tomllib.load(fh)
    return list(config.get("adapters", {}).keys())
