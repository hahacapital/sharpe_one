"""Portfolio combiner — equal-weight multi-strategy allocation.

Combines signals from multiple strategies into a single blended signal.
Each strategy gets equal capital allocation. The combined signal is the
average of all individual strategy signals, producing fractional positions
that naturally diversify across strategies.
"""

import pandas as pd
import numpy as np
from strategy.base import Strategy, discover_strategies


class PortfolioStrategy(Strategy):
    """Combines all discovered strategies into an equal-weight portfolio."""

    name = "portfolio"
    params = {}
    param_space = {}

    def __init__(self, strategy_classes=None, **kwargs):
        super().__init__(**kwargs)
        if strategy_classes is None:
            # Auto-discover all non-portfolio strategies
            strategy_classes = [
                s for s in discover_strategies()
                if s.name != "portfolio"
            ]
        self._strategies = [cls() for cls in strategy_classes]
        self._strategy_classes = strategy_classes

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if not self._strategies:
            return pd.Series(0, index=df.index)

        all_signals = []
        for s in self._strategies:
            try:
                sig = s.generate_signals(df)
                all_signals.append(sig)
            except Exception:
                continue

        if not all_signals:
            return pd.Series(0, index=df.index)

        # Equal-weight average of all strategy signals
        combined = pd.concat(all_signals, axis=1).mean(axis=1)

        # Discretize: > 0.2 → long, < -0.2 → short, else flat
        signals = pd.Series(0, index=df.index)
        signals[combined > 0.2] = 1
        signals[combined < -0.2] = -1

        return signals

    def required_data(self) -> dict:
        """Union of all sub-strategy data requirements."""
        req = {"ohlcv": set(), "funding_rate": False}
        for s in self._strategies:
            r = s.required_data()
            for tf in r.get("ohlcv", []):
                req["ohlcv"].add(tf)
            if r.get("funding_rate"):
                req["funding_rate"] = True
        req["ohlcv"] = sorted(req["ohlcv"]) or ["4h"]
        return req
