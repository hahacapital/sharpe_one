"""Funding rate carry strategy (4h timeframe).

Collects funding payments by going short when funding is persistently
positive (over days, not hours). Uses a very long smoothing period
so the signal barely changes bar-to-bar, making execution delay irrelevant.
Combines with slow price trend to avoid fighting strong directional moves.
"""

import pandas as pd
import numpy as np
from strategy.base import Strategy


class FundingArbStrategy(Strategy):
    name = "funding_arb"
    params = {
        "funding_slow_ma": 42,      # ~7 days on 4h (funding settles 3x/day)
        "funding_threshold": 0.0,   # Go short if MA funding > 0
        "price_ma": 120,            # ~20 days price MA for trend filter
        "trend_weight": 0.5,        # How much to weigh trend vs funding
    }
    param_space = {
        "funding_slow_ma": (18, 90),
        "funding_threshold": (-0.0002, 0.0002),
        "price_ma": (60, 240),
        "trend_weight": (0.0, 1.0),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if "funding_rate" not in df.columns:
            return pd.Series(0, index=df.index)

        p = self.params
        close = df["close"]
        fr = df["funding_rate"]

        # Smooth funding rate over many days
        fr_ma = fr.rolling(window=p["funding_slow_ma"], min_periods=6).mean()

        # Funding signal: short when funding positive, long when negative
        funding_signal = pd.Series(0.0, index=df.index)
        funding_signal[fr_ma > p["funding_threshold"]] = -1.0
        funding_signal[fr_ma < -p["funding_threshold"]] = 1.0

        # Price trend signal
        price_ma = close.ewm(span=p["price_ma"], adjust=False).mean()
        trend_signal = pd.Series(0.0, index=df.index)
        trend_signal[close > price_ma] = 1.0
        trend_signal[close < price_ma] = -1.0

        # Combined: weighted average. When they agree, full position.
        # When they disagree, reduced or no position.
        tw = p["trend_weight"]
        combined = (1 - tw) * funding_signal + tw * trend_signal

        signals = pd.Series(0, index=df.index)
        signals[combined > 0.3] = 1
        signals[combined < -0.3] = -1

        return signals

    def required_data(self) -> dict:
        return {"ohlcv": ["4h"], "funding_rate": True}
