"""Funding rate arbitrage strategy.

When funding rate is high positive, shorts are paying longs — go short perp.
When funding rate is very negative, longs are paying shorts — go long perp.
"""

import pandas as pd
import numpy as np
from strategy.base import Strategy


class FundingArbStrategy(Strategy):
    name = "funding_arb"
    params = {
        "funding_threshold_long": -0.0005,   # Go long when funding < this (shorts paying)
        "funding_threshold_short": 0.001,    # Go short when funding > this (longs paying)
        "funding_ma_period": 8,              # Smooth funding rate over N periods (8h = 1 day)
        "exit_threshold": 0.0002,            # Close position when funding reverts past this
    }
    param_space = {
        "funding_threshold_long": (-0.002, -0.0001),
        "funding_threshold_short": (0.0003, 0.003),
        "funding_ma_period": (4, 24),
        "exit_threshold": (0.0, 0.001),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if "funding_rate" not in df.columns:
            return pd.Series(0, index=df.index)

        p = self.params
        fr = df["funding_rate"].rolling(window=p["funding_ma_period"], min_periods=1).mean()

        signals = pd.Series(0, index=df.index)
        signals[fr > p["funding_threshold_short"]] = -1   # Short when funding high
        signals[fr < p["funding_threshold_long"]] = 1     # Long when funding negative

        return signals

    def required_data(self) -> dict:
        return {"ohlcv": ["1h"], "funding_rate": True}
