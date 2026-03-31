# Sharpe One Architecture Design

## Overview

Sharpe One is a cryptocurrency quantitative fund aiming for Sharpe Ratio >= 1, outperforming BTC, Gold, and QQQ benchmarks across all market environments through long/short strategies.

## Constraints & Decisions

- **Initial capital**: < $10K (experimental phase)
- **Exchanges**: Multi-exchange (Binance, OKX, Bybit, etc.)
- **Architecture**: Modular layered (Phase B), migrating to event-driven (Phase C) later
- **Deployment**: Local development first, then AWS (EC2/ECS)
- **Data**: Free sources first (CCXT), upgrade as needed
- **On-chain**: Alchemy/Helius APIs for supplementary signals, not primary strategies
- **Max drawdown**: 30%
- **User profile**: Python proficient, new to quant trading

## Project Structure

```
sharpe_one/
├── config/              # Configuration (exchange keys, strategy params)
│   └── settings.yaml
├── data/
│   ├── feeds/           # Market data sources (CEX via CCXT, on-chain via Alchemy/Helius)
│   ├── store.py         # Data storage (SQLite → PostgreSQL+TimescaleDB)
│   └── models.py        # Data models (OHLCV, FundingRate, OrderBook, etc.)
├── strategy/
│   ├── base.py          # Strategy base class (unified interface)
│   ├── funding_arb.py   # Funding rate arbitrage
│   └── trend.py         # Trend following
├── portfolio/
│   ├── manager.py       # Portfolio management, position aggregation
│   └── risk.py          # Risk controls
├── execution/
│   ├── router.py        # Exchange routing (select optimal exchange)
│   └── executor.py      # Order execution, order management
├── backtest/
│   ├── engine.py        # Backtesting engine
│   └── metrics.py       # Performance metrics
├── utils/
│   └── logger.py        # Logging
└── main.py              # Entry point
```

## Data Layer

### Data Models (`data/models.py`)

Core data structures using Python dataclasses:

- **OHLCV**: Open/High/Low/Close/Volume candlestick data
- **FundingRate**: Perpetual contract funding rates (exchange, symbol, rate, timestamp)
- **OrderBook**: Order book snapshots (bid/ask depth)
- **OnChainFlow**: On-chain fund flows (large transfers, exchange inflow/outflow)
- **Signal**: Strategy output (direction, strength, target position)

### Data Sources (`data/feeds/`)

| Source | Library | Data |
|--------|---------|------|
| CEX market data | CCXT | Candles, funding rates, order books, account balances |
| On-chain data | Alchemy/Helius SDK | Large wallet transfers, exchange inflow/outflow |

### Storage (`data/store.py`)

Abstract interface supporting backend swap without code changes:

- **Development**: Local SQLite (zero cost, fast iteration)
- **Production**: AWS RDS PostgreSQL + TimescaleDB extension (time-series optimized)
- **Object storage**: AWS S3 for historical backtest data, strategy logs, performance reports
- **Secrets**: AWS Secrets Manager for exchange API keys
- **In-memory cache**: Real-time data (latest tick, order book) for trading

Data sources implement `subscribe()` and `get_latest()` methods — future EventBus integration requires only adding an event publish layer.

## Strategy Layer

### Strategy Base Class (`strategy/base.py`)

```python
class Strategy:
    def on_data(self, data) -> list[Signal]:
        """Receive market data, return trading signals."""
        raise NotImplementedError

    def get_params(self) -> dict:
        """Return strategy parameters for backtest optimization."""

    def get_state(self) -> dict:
        """Return internal state for monitoring and recovery."""
```

### Initial Strategies

| Strategy | Mechanism | Expected Sharpe | Priority |
|----------|-----------|----------------|----------|
| `funding_arb.py` | Long spot + short perp, collect funding rate | 1.5-3 | First |
| `trend.py` | Multi-timeframe momentum, long/short with trend | 0.5-1.0 | Second |
| `cross_exchange_arb.py` | Same-coin cross-exchange spread | 2+ | Third |

### On-Chain Signal Enhancement

On-chain data acts as a signal overlay, not an independent strategy:
- Example: Large BTC inflow to exchanges detected on-chain → increase bearish weight in trend strategy
- Injected into strategy via `OnChainFlow` data in `on_data()`

### Migration to Event-Driven (Phase C)

`on_data()` is naturally an event handler. Future EventBus dispatches `MarketDataEvent` to each strategy's `on_data()` — strategy code stays unchanged.

## Portfolio Management & Risk Control

### Portfolio Manager (`portfolio/manager.py`)

- Aggregates Signals from all strategies, calculates target positions
- Strategy capital allocation: fixed ratios initially (e.g., funding arb 50%, trend 30%, cross-exchange arb 20%), dynamic adjustment later
- Unified cross-exchange position view (balances and positions aggregated)

### Risk Controls (`portfolio/risk.py`)

| Rule | Description |
|------|-------------|
| Max position per strategy | No more than X% of total capital |
| Max single-coin exposure | Net long/short exposure cap |
| Max drawdown stop-loss | **30% max drawdown** — reduce all positions when breached |
| Exchange diversification | No more than Y% of capital on a single exchange |
| Leverage cap | Total leverage multiplier limit |

Parameters configured via `config/settings.yaml`, not hardcoded.

Core flow:
```
Strategy Signals → Manager aggregates → Risk checks → Pass: generate Order → Fail: reduce position
```

## Execution Layer

### Exchange Router (`execution/router.py`)

- Compares same coin across exchanges, selects optimal execution: lowest fees, best liquidity, tightest spread
- Maintains available balance per exchange, prevents insufficient-funds orders

### Order Executor (`execution/executor.py`)

- Unified order placement via CCXT
- Order types: limit (default), market (urgent situations)
- Order status tracking: pending → partially filled → fully filled / cancelled
- Failure retry: auto-retry on network timeout, log exchange errors

### Execution Safety

| Mechanism | Description |
|-----------|-------------|
| Pre-order balance check | Verify sufficient funds before placing order |
| Slippage protection | Reject limit orders deviating beyond threshold from mid-price |
| Rate limiting | Respect exchange API rate limits |
| Simulation mode | Development phase: log orders only, no real execution |

**Simulation mode is first priority** — all orders are simulated until strategy is validated, preventing bugs from causing fund loss.

## Backtesting Engine

### Engine (`backtest/engine.py`)

- Reads historical data from SQLite, replays in chronological order
- Reuses strategy code — same `Strategy.on_data()` for backtest and live
- Simulates trading costs: fees (maker/taker), slippage, funding rates

### Performance Metrics (`backtest/metrics.py`)

| Metric | Purpose |
|--------|---------|
| Sharpe Ratio | Core target, >= 1 |
| Max Drawdown | Must be <= 30% |
| Annualized Return | Benchmark against BTC/Gold/QQQ |
| Win Rate & Profit Factor | Strategy quality assessment |
| Calmar Ratio | Return / max drawdown |
| Per-strategy performance | Identify top contributors |

### Benchmark Comparison

- Every backtest automatically calculates BTC, Gold, QQQ performance over the same period
- Outputs comparison charts and excess returns

### Anti-Overfitting

- Data split into train / test sets (e.g., 70/30)
- Parameters not optimized on test set
- Every backtest logs parameters and results for traceability

### Look-Ahead Bias Prevention

Three identified risk points and their mitigations:

1. **Signal vs execution timing**: Signal generated at bar N close, **executed at bar N+1 open price**. Never trade at the current bar's close.
2. **Funding rate**: Only use **settled historical rates** for decisions. Current (unsettled) rate is reference only, not a decision input.
3. **On-chain data timestamps**: Block confirmation has latency. Add **confirmation delay** (e.g., +2 minutes) to on-chain events to simulate real observable time.

**Engine-level safeguards**:
- Strict timeline management: strategy at time T can only access data before T
- Data access interface with `as_of(timestamp)` filter — physically impossible to read future data
- Execution uses next bar's open, not current bar's close
