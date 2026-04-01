# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Sharpe One — a cryptocurrency quantitative fund focused on long/short strategies. Goal: Sharpe Ratio >= 1, outperforming BTC, QQQ, and Gold benchmarks across all market regimes.

## Language & Tooling

- Python 3.10+
- Dependencies: ccxt, pandas, numpy, optuna, yfinance, pyarrow
- Install: `pip install -r requirements.txt`
- No test framework configured yet

## Project Structure

```
sharpe_one/
├── config/settings.yaml       # Global config (fees, slippage, universe filters)
├── data/
│   ├── raw/{SYMBOL}/*.parquet # Historical data (OHLCV, funding rates)
│   ├── fetcher.py             # Binance data fetching via CCXT
│   └── universe.py            # Dynamic coin selection
├── strategy/
│   ├── base.py                # Strategy base class + auto-discovery
│   ├── funding_arb.py         # Funding rate arbitrage
│   ├── trend_following.py     # Trend following
│   └── mean_reversion.py      # Mean reversion
├── backtest/
│   ├── engine.py              # Vectorized backtest engine
│   ├── metrics.py             # Performance metrics (Sharpe, MDD, Calmar, etc.)
│   └── benchmark.py           # Benchmark comparison (BTC/QQQ/Gold)
├── optimize/optimizer.py      # Bayesian param optimization (optuna)
├── results/                   # Backtest & optimization results (JSON)
└── main.py                    # Entry point
```

## Architecture Decisions

- **Parquet over SQLite** for data storage — zero serialization overhead for vectorized backtest
- **Vectorized backtest** — strategies receive full DataFrame, return signal Series; no per-bar callbacks
- **Strategy as plugin** — each strategy is a standalone .py file in strategy/, auto-discovered by the engine
- **Train/test split (70/30)** — params optimized on train only, final eval on test to prevent overfitting
- **Signal delay** — signals at bar N execute at bar N+1 open to prevent look-ahead bias

## Strategy Interface

Strategies must inherit `strategy.base.Strategy` and implement:
- `generate_signals(df: pd.DataFrame) -> pd.Series` — returns 1 (long), -1 (short), 0 (flat)
- `required_data() -> dict` — declares needed data types/timeframes
- Define `params` (defaults) and `param_space` (optimization ranges)

## Key Commands

- `python main.py fetch` — pull historical data from Binance
- `python main.py backtest` — run backtest for all strategies
- `python main.py backtest --strategy trend_following` — run single strategy

## Iteration Workflow

- **Strategy logic iteration**: Claude autonomously loops (backtest → analyze → modify code → backtest) in conversation, stops on breakthrough or stagnation (3 rounds no improvement)
- **Parameter optimization**: `/optimize` skill, runs optuna-based Bayesian search on param_space
