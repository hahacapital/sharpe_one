"""Strategy base class and auto-discovery."""

import importlib
import inspect
import os
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class Strategy:
    """Base class for all strategies.

    Subclasses must implement generate_signals().
    Define params (dict of defaults) and param_space (dict of (min, max) tuples)
    for parameter optimization.
    """

    name: str = "base"
    params: dict = {}
    param_space: dict = {}

    def __init__(self, **kwargs):
        # Override default params with any provided kwargs
        self.params = {**self.__class__.params, **kwargs}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals from OHLCV data.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume].
                May also contain funding_rate column.

        Returns:
            Series with same index as df. Values: 1 (long), -1 (short), 0 (flat).
        """
        raise NotImplementedError

    def required_data(self) -> dict:
        """Declare required data types and timeframes.

        Returns:
            Dict like {"ohlcv": ["1h"], "funding_rate": True}
        """
        return {"ohlcv": ["1h"]}


def discover_strategies() -> list[type[Strategy]]:
    """Auto-discover all Strategy subclasses in the strategy/ directory.

    Scans all .py files in strategy/ (excluding base.py, __init__.py),
    imports them, and collects classes that inherit from Strategy.
    """
    strategy_dir = os.path.dirname(__file__)
    strategies = []

    for filename in sorted(os.listdir(strategy_dir)):
        if not filename.endswith(".py"):
            continue
        if filename in ("base.py", "__init__.py"):
            continue

        module_name = f"strategy.{filename[:-3]}"
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            logger.error(f"Failed to import {module_name}: {e}")
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Strategy) and obj is not Strategy:
                strategies.append(obj)
                logger.info(f"Discovered strategy: {obj.name}")

    return strategies
