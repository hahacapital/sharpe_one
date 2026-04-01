# Self-Iterating Strategy Optimization System Design

## Overview

An AI-driven strategy evolution system for the Sharpe One crypto quant fund. The system has two iteration loops:

1. **Strategy logic iteration** — Claude Code autonomously loops (backtest → analyze → modify strategy code → backtest) within a conversation, stopping only at key milestones
2. **Parameter optimization** — A skill (`/optimize`) runs Bayesian optimization via optuna, executed by Claude Code CLI

**Target:** Sharpe Ratio >= 1, simultaneously outperforming BTC, QQQ, and Gold benchmarks across all market regimes, with max drawdown <= 30%.

## Constraints & Decisions

- **Initial capital**: < $10K (experimental phase)
- **Exchange**: Binance (CCXT), add more later
- **Data storage**: Parquet (not SQLite) — optimized for vectorized backtest
- **Backtest engine**: Vectorized (pandas/numpy), not event-driven
- **Anti-overfitting**: 70/30 train/test split
- **User profile**: Python proficient, new to quant trading
- **No Claude API key**: parameter optimization skill runs via Claude Code CLI (setup token)

## Project Structure

```
sharpe_one/
├── config/
│   └── settings.yaml          # Global config (fees, slippage, universe filters, data paths)
├── data/
│   ├── raw/                   # Parquet data files
│   │   └── {SYMBOL}/{timeframe}.parquet
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
├── optimize/
│   └── optimizer.py           # Bayesian parameter optimization (optuna)
├── results/                   # Backtest & optimization results (JSON)
├── main.py                    # Entry point
└── requirements.txt           # Dependencies
```

## Data Layer

### Data Fetching (`data/fetcher.py`)

- Pulls historical data from Binance via CCXT
- Supported data types: OHLCV candles (1m/5m/15m/1h/4h/1d), Funding Rates
- Incremental updates: tracks last timestamp per symbol/timeframe, only fetches new data
- Stores as Parquet files organized by `data/raw/{SYMBOL}/{timeframe}.parquet`

### Dynamic Coin Selection (`data/universe.py`)

- Fetches all USDT perpetual contracts from Binance
- Filter criteria (configurable in settings.yaml):
  - 24h volume ranking Top N (default: 30)
  - Listed >= X days (default: 90, excludes new coins)
  - Excludes stablecoin pairs (e.g., BUSD/USDT)
- Output: list of symbols for backtest and live trading
- Results cached — not re-computed every backtest round

### Benchmark Data (`backtest/benchmark.py`)

- BTC: from Binance via CCXT (already available)
- QQQ and Gold: pulled via `yfinance` (free)
- Stored as Parquet alongside other data

## Strategy Layer

### Strategy Base Class (`strategy/base.py`)

```python
class Strategy:
    name: str                    # Strategy name
    params: dict                 # Optimizable params with defaults
    param_space: dict            # Search space for optimizer
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Input: OHLCV DataFrame. Output: signal Series (1=long, -1=short, 0=flat)."""
        raise NotImplementedError
    
    def required_data(self) -> dict:
        """Declare required data types and timeframes."""
        return {"ohlcv": ["1h"]}
```

Key properties:
- `generate_signals()` is a pure function — input data, output signals, no side effects
- `param_space` defines search ranges, e.g., `{"fast_period": (5, 50), "slow_period": (20, 200)}`
- Parameter optimization skill reads `param_space` to auto-scan

### Strategy Auto-Discovery

- Backtest engine scans `strategy/` directory for all `.py` files (excluding `base.py`, `__init__.py`)
- Each file exports one class inheriting `Strategy`
- Claude can add/modify/delete strategy files during iteration

### Initial Strategies

| File | Strategy | Mechanism |
|------|----------|-----------|
| `funding_arb.py` | Funding Rate Arbitrage | Short perp + long spot when funding is high |
| `trend_following.py` | Trend Following | Multi-timeframe momentum, long/short with trend |
| `mean_reversion.py` | Mean Reversion | Fade extreme deviations from moving averages |

### What Claude Can Do During Iteration

- Modify existing strategy logic
- Adjust parameter space definitions
- Create entirely new strategy files
- Delete consistently underperforming strategies

## Backtest Engine

### Core (`backtest/engine.py`)

```python
def run_backtest(strategy, data, params=None) -> BacktestResult:
    """
    1. Call strategy.generate_signals(data) to get signal Series
    2. Delay signals by one bar (prevent look-ahead bias)
    3. Vectorized calculation: positions → returns → equity curve
    4. Deduct trading costs (maker/taker fees + slippage estimate)
    5. Return BacktestResult
    """
```

### Trading Cost Simulation

- Fees: maker 0.02%, taker 0.05% (Binance perpetual defaults)
- Slippage: 0.01% of trade value (configurable)
- Funding rate: positions in perpetuals charged/paid at historical actual funding rates

### Performance Metrics (`backtest/metrics.py`)

| Metric | Purpose |
|--------|---------|
| Sharpe Ratio | Core target >= 1 |
| Max Drawdown | Must be <= 30% |
| Annualized Return | Compare against benchmarks |
| Win Rate | Strategy quality |
| Profit Factor | Win/loss ratio |
| Calmar Ratio | Return / max drawdown |
| Total Trades | Strategy activity level |

### Benchmark Comparison

- Every backtest automatically computes BTC, QQQ, and Gold buy-and-hold performance over the same period
- Strategy must outperform **all three** to be considered successful
- Outputs excess returns vs each benchmark

### Anti-Overfitting

- Data split into train (70%) / test (30%)
- Parameters optimized on train set only
- Final evaluation on test set
- Backtest reports include both train and test metrics for Claude to assess overfitting

### Look-Ahead Bias Prevention

- Signals at bar N are executed at bar N+1 open price
- Funding rate: only settled historical rates used for decisions
- Strict timeline: strategy at time T can only access data before T

### Report Output

- Saved to `results/{strategy_name}_{timestamp}.json`
- Contains: all metrics, parameters, signal statistics, train/test breakdown
- Claude reads these JSON files to analyze and decide next steps

## Iteration Control Layer

### Claude Autonomous Strategy Iteration (In-Conversation Loop)

Triggered by user saying "start iterating". Claude enters autonomous loop:

```
┌→ Run backtest for all strategies
│   ↓
│  Read backtest reports (JSON)
│   ↓
│  Analyze: which strategies perform well/poorly, why
│   ↓
│  Act: modify strategy logic / create new strategies / delete poor ones
│   ↓
│  Run backtest again, compare before/after
│   ↓
│  Evaluate: significant improvement? stagnation?
│   ├─ Improvement → continue loop ──────────────────┐
│   ├─ 3 consecutive rounds no improvement → stop, report to user
│   └─ Breakthrough (Sharpe > target, beats all 3 benchmarks) → stop, report
└────────────────────────────────────────────────────┘
```

**Stop conditions:**
- A strategy achieves Sharpe >= 1 on test set AND outperforms all three benchmarks → report success
- 3 consecutive rounds with no improvement → ask user for direction
- Overfitting detected (train great, test poor) → report and discuss

### Parameter Optimization Skill (`/optimize`)

A standalone skill executed by Claude Code CLI:

```
/optimize strategy=trend_following
```

Flow:
1. Read strategy's `param_space` definition
2. Run Bayesian optimization (optuna) on train set
3. 100-500 trials (configurable)
4. Output: optimal parameters + train/test performance comparison
5. Results saved to `results/optimize_{strategy}_{timestamp}.json`

### How The Two Loops Interact

- Claude iteration modifies strategy **logic** (code)
- `/optimize` fine-tunes **parameters** after logic is locked
- Typical flow: Claude modifies logic → `/optimize` tunes params → Claude reviews results → next iteration

## Dependencies

```
ccxt          # Exchange API
pandas        # Data manipulation
numpy         # Numerical computation
optuna        # Bayesian optimization
yfinance      # QQQ and Gold benchmark data
pyarrow       # Parquet read/write
```
