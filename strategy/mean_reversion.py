"""Volatility breakout strategy.

Trades breakouts from consolidation ranges. When volatility contracts
(Bollinger Band width narrows), prepare for a breakout. Enter in the
direction of the breakout when price closes outside the bands.
"""

import pandas as pd
import numpy as np
from strategy.base import Strategy


class VolatilityBreakoutStrategy(Strategy):
    name = "volatility_breakout"
    params = {
        "bb_period": 20,
        "bb_std": 2.0,
        "squeeze_lookback": 50,   # Compare current bandwidth to recent history
        "squeeze_percentile": 20, # Bandwidth < 20th percentile = squeeze
        "atr_period": 14,
        "trail_atr_mult": 2.0,    # Trailing stop = ATR * multiplier
    }
    param_space = {
        "bb_period": (10, 40),
        "bb_std": (1.5, 3.0),
        "squeeze_lookback": (20, 100),
        "squeeze_percentile": (10, 40),
        "atr_period": (7, 28),
        "trail_atr_mult": (1.0, 4.0),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Bollinger Bands
        ma = close.rolling(window=p["bb_period"], min_periods=p["bb_period"]).mean()
        std = close.rolling(window=p["bb_period"], min_periods=p["bb_period"]).std()
        upper = ma + std * p["bb_std"]
        lower = ma - std * p["bb_std"]

        # Bandwidth (normalized)
        bandwidth = (upper - lower) / ma
        bandwidth = bandwidth.fillna(0)

        # Squeeze detection: bandwidth in bottom percentile of recent history
        bw_pct = bandwidth.rolling(window=p["squeeze_lookback"], min_periods=p["squeeze_lookback"]).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        was_squeezed = bw_pct.shift(1) < (p["squeeze_percentile"] / 100)

        # ATR for trailing stop
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=p["atr_period"], min_periods=1).mean()

        # Breakout entries
        long_entry = (close > upper) & was_squeezed
        short_entry = (close < lower) & was_squeezed

        # Build signals with ATR trailing stop
        signals = pd.Series(0, index=df.index, dtype=float)
        pos = 0
        stop_price = 0.0

        for i in range(len(df)):
            if pos == 1:
                # Update trailing stop
                new_stop = close.iloc[i] - atr.iloc[i] * p["trail_atr_mult"]
                stop_price = max(stop_price, new_stop)
                if low.iloc[i] < stop_price:
                    pos = 0
            elif pos == -1:
                new_stop = close.iloc[i] + atr.iloc[i] * p["trail_atr_mult"]
                stop_price = min(stop_price, new_stop)
                if high.iloc[i] > stop_price:
                    pos = 0
            else:
                if long_entry.iloc[i]:
                    pos = 1
                    stop_price = close.iloc[i] - atr.iloc[i] * p["trail_atr_mult"]
                elif short_entry.iloc[i]:
                    pos = -1
                    stop_price = close.iloc[i] + atr.iloc[i] * p["trail_atr_mult"]

            signals.iloc[i] = pos

        return signals

    def required_data(self) -> dict:
        return {"ohlcv": ["1h"]}
