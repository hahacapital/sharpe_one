# Sharpe One

Cryptocurrency quantitative fund — long/short strategies targeting Sharpe Ratio >= 1, outperforming BTC, QQQ, and Gold across all market regimes.

## Overview

Sharpe One is an AI-driven quant trading system with two core loops:

1. **Strategy logic iteration** — Claude Code autonomously analyzes backtest results, modifies strategy code, and re-tests until finding winning approaches
2. **Parameter optimization** — Bayesian optimization (optuna) fine-tunes strategy parameters after logic is locked

## Features

- Vectorized backtest engine (pandas/numpy) for fast iteration
- Plugin-based strategy system — each strategy is a standalone Python file
- Dynamic coin universe selection (top N by volume from Binance USDT perpetuals)
- Anti-overfitting: 70/30 train/test split, look-ahead bias prevention
- Benchmark comparison against BTC, QQQ, and Gold simultaneously
- Historical data storage in Parquet format for efficient loading

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Pull historical data from Binance
python main.py fetch

# Run backtest for all strategies
python main.py backtest

# Run backtest for a single strategy
python main.py backtest --strategy trend_following
```

## Project Structure

```
sharpe_one/
├── config/settings.yaml       # Global config
├── data/
│   ├── raw/                   # Parquet data files
│   ├── fetcher.py             # Binance data fetching (CCXT)
│   └── universe.py            # Dynamic coin selection
├── strategy/
│   ├── base.py                # Strategy base class
│   ├── funding_arb.py         # Funding rate arbitrage
│   ├── trend_following.py     # Trend following
│   └── mean_reversion.py      # Mean reversion
├── backtest/
│   ├── engine.py              # Vectorized backtest engine
│   ├── metrics.py             # Performance metrics
│   └── benchmark.py           # BTC/QQQ/Gold benchmark
├── optimize/optimizer.py      # Bayesian parameter optimization
├── results/                   # Backtest results (JSON)
└── main.py                    # Entry point
```

## Strategies

| Strategy | Mechanism | File |
|----------|-----------|------|
| Funding Rate Arbitrage | Short perp + long spot when funding is high | `strategy/funding_arb.py` |
| Trend Following | Multi-timeframe momentum, long/short with trend | `strategy/trend_following.py` |
| Mean Reversion | Fade extreme deviations from moving averages | `strategy/mean_reversion.py` |

Strategies are auto-discovered from the `strategy/` directory. New strategies can be added by creating a new `.py` file that inherits from `Strategy` base class.

## Performance Targets

- Sharpe Ratio >= 1
- Max Drawdown <= 30%
- Must outperform all three benchmarks: BTC, QQQ, Gold
