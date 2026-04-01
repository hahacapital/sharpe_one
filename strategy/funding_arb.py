"""Funding rate arbitrage strategy.

Collects funding payments by positioning in the direction that receives
funding, but ONLY when the price trend agrees. This avoids fighting
the trend while collecting funding income.

Positive funding → shorts receive → go short only if trend is down/neutral
Negative funding → longs receive → go long only if trend is up/neutral
"""

import pandas as pd
import numpy as np
from strategy.base import Strategy


class FundingArbStrategy(Strategy):
    name = "funding_arb"
    params = {
        "funding_ma_period": 24,            # Smooth funding over 24h (3 settlements)
        "funding_threshold": 0.0001,        # Minimum funding rate magnitude to act
        "trend_period": 48,                 # MA period for trend filter
        "trend_tolerance": 0.002,           # Allow entry if price within X% of MA
    }
    param_space = {
        "funding_ma_period": (8, 48),
        "funding_threshold": (0.00005, 0.0005),
        "trend_period": (24, 96),
        "trend_tolerance": (0.0, 0.01),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if "funding_rate" not in df.columns:
            return pd.Series(0, index=df.index)

        p = self.params
        close = df["close"]
        fr = df["funding_rate"].rolling(window=p["funding_ma_period"], min_periods=1).mean()

        # Trend filter: position of price relative to MA
        trend_ma = close.rolling(window=p["trend_period"], min_periods=1).mean()
        price_vs_ma = (close - trend_ma) / trend_ma

        # Funding signals with trend alignment
        # Short to collect positive funding — only when price not strongly above MA
        short_signal = (fr > p["funding_threshold"]) & (price_vs_ma < p["trend_tolerance"])
        # Long to collect negative funding — only when price not strongly below MA
        long_signal = (fr < -p["funding_threshold"]) & (price_vs_ma > -p["trend_tolerance"])

        signals = pd.Series(0, index=df.index)
        signals[short_signal] = -1
        signals[long_signal] = 1

        return signals

    def required_data(self) -> dict:
        return {"ohlcv": ["1h"], "funding_rate": True}
