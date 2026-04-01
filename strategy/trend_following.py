"""Trend following strategy.

Uses dual moving average crossover with ATR-based volatility filter.
Long when fast MA > slow MA and trend is strong, short when opposite.
"""

import pandas as pd
import numpy as np
from strategy.base import Strategy


class TrendFollowingStrategy(Strategy):
    name = "trend_following"
    params = {
        "fast_period": 20,
        "slow_period": 60,
        "atr_period": 14,
        "atr_threshold": 1.0,   # Minimum ATR multiplier to confirm trend
        "use_ema": True,        # Use EMA instead of SMA
    }
    param_space = {
        "fast_period": (5, 50),
        "slow_period": (20, 200),
        "atr_period": (7, 30),
        "atr_threshold": (0.5, 3.0),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Moving averages
        if p["use_ema"]:
            fast_ma = close.ewm(span=p["fast_period"], adjust=False).mean()
            slow_ma = close.ewm(span=p["slow_period"], adjust=False).mean()
        else:
            fast_ma = close.rolling(window=p["fast_period"], min_periods=1).mean()
            slow_ma = close.rolling(window=p["slow_period"], min_periods=1).mean()

        # ATR for volatility filter
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=p["atr_period"], min_periods=1).mean()

        # Trend strength: distance between MAs relative to ATR
        ma_distance = (fast_ma - slow_ma).abs()
        trend_strong = ma_distance > (atr * p["atr_threshold"])

        # Signals
        signals = pd.Series(0, index=df.index)
        signals[(fast_ma > slow_ma) & trend_strong] = 1    # Long
        signals[(fast_ma < slow_ma) & trend_strong] = -1   # Short

        return signals

    def required_data(self) -> dict:
        return {"ohlcv": ["1h"]}
