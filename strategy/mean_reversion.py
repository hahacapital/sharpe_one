"""Momentum strategy (4h timeframe).

Buys coins that have been going up over the past N days (momentum
continuation) and shorts coins going down. Momentum is the most
well-documented alpha factor — trends persist due to behavioral biases.
Uses rate-of-change over multiple lookback periods for robustness.
"""

import pandas as pd
import numpy as np
from strategy.base import Strategy


class MomentumStrategy(Strategy):
    name = "momentum"
    params = {
        "roc_short": 30,        # ~5 days on 4h
        "roc_long": 120,        # ~20 days on 4h
        "signal_threshold": 0.0,  # Go long if combined ROC > 0
        "vol_lookback": 30,     # Normalize by recent volatility
    }
    param_space = {
        "roc_short": (12, 72),
        "roc_long": (48, 240),
        "signal_threshold": (-0.02, 0.02),
        "vol_lookback": (12, 60),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        close = df["close"]

        # Rate of change over two horizons
        roc_short = close.pct_change(p["roc_short"])
        roc_long = close.pct_change(p["roc_long"])

        # Normalize by realized volatility for comparability
        vol = close.pct_change().rolling(window=p["vol_lookback"], min_periods=1).std()
        vol = vol.replace(0, np.nan).ffill().fillna(1)

        # Combined momentum score (vol-normalized)
        mom_score = (roc_short / vol + roc_long / vol) / 2

        signals = pd.Series(0, index=df.index)
        signals[mom_score > p["signal_threshold"]] = 1
        signals[mom_score < -p["signal_threshold"]] = -1

        return signals

    def required_data(self) -> dict:
        return {"ohlcv": ["4h"]}
