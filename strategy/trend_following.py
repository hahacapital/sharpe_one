"""Trend following strategy (4h timeframe).

Uses slow dual EMA crossover to catch multi-week trends.
With 4h bars and 50/150 EMA (~8d/25d), signals persist for weeks,
making the 1-bar execution delay negligible. ADX filter avoids whipsaws.
"""

import pandas as pd
import numpy as np
from strategy.base import Strategy


class TrendFollowingStrategy(Strategy):
    name = "trend_following"
    params = {
        "fast_period": 50,      # ~8 days on 4h
        "slow_period": 150,     # ~25 days on 4h
        "adx_period": 14,
        "adx_min": 20,          # Only trade when trend is strong enough
    }
    param_space = {
        "fast_period": (20, 100),
        "slow_period": (80, 300),
        "adx_period": (10, 28),
        "adx_min": (15, 35),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        close = df["close"]
        high = df["high"]
        low = df["low"]

        fast_ma = close.ewm(span=p["fast_period"], adjust=False).mean()
        slow_ma = close.ewm(span=p["slow_period"], adjust=False).mean()

        adx = self._compute_adx(high, low, close, p["adx_period"])

        signals = pd.Series(0, index=df.index)
        trending = adx > p["adx_min"]
        signals[(fast_ma > slow_ma) & trending] = 1
        signals[(fast_ma < slow_ma) & trending] = -1

        return signals

    @staticmethod
    def _compute_adx(high, low, close, period):
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period, min_periods=1).mean()
        plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(window=period, min_periods=1).mean()
        return adx.fillna(0)

    def required_data(self) -> dict:
        return {"ohlcv": ["4h"]}
