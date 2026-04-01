"""Mean reversion strategy.

Fades extreme deviations from a moving average using Bollinger Bands.
Goes long when price drops below lower band, short when above upper band.
"""

import pandas as pd
import numpy as np
from strategy.base import Strategy


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    params = {
        "bb_period": 20,
        "bb_std": 2.0,          # Bollinger Band standard deviations
        "rsi_period": 14,
        "rsi_oversold": 30,     # Confirm mean reversion with RSI
        "rsi_overbought": 70,
    }
    param_space = {
        "bb_period": (10, 50),
        "bb_std": (1.5, 3.5),
        "rsi_period": (7, 28),
        "rsi_oversold": (15, 40),
        "rsi_overbought": (60, 85),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        close = df["close"]

        # Bollinger Bands
        ma = close.rolling(window=p["bb_period"], min_periods=1).mean()
        std = close.rolling(window=p["bb_period"], min_periods=1).std()
        upper_band = ma + (std * p["bb_std"])
        lower_band = ma - (std * p["bb_std"])

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=p["rsi_period"], min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=p["rsi_period"], min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)

        # Signals: price outside bands + RSI confirmation
        signals = pd.Series(0, index=df.index)
        signals[(close < lower_band) & (rsi < p["rsi_oversold"])] = 1    # Long: oversold
        signals[(close > upper_band) & (rsi > p["rsi_overbought"])] = -1  # Short: overbought

        return signals

    def required_data(self) -> dict:
        return {"ohlcv": ["1h"]}
