# Self-Iterating Strategy Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete backtest system that Claude can autonomously iterate on — fetching Binance data, running vectorized backtests, computing metrics, comparing against BTC/QQQ/Gold benchmarks, and optimizing parameters via optuna.

**Architecture:** Plugin-based strategy system with vectorized backtest engine. Strategies are standalone Python files auto-discovered from `strategy/` directory. Data stored as Parquet files. Two iteration loops: Claude modifies strategy logic in-conversation; `/optimize` skill runs Bayesian parameter search via CLI.

**Tech Stack:** Python 3.10+, ccxt, pandas, numpy, optuna, yfinance, pyarrow

---

## File Map

| File | Responsibility |
|------|----------------|
| `requirements.txt` | Python dependencies |
| `config/settings.yaml` | Global config: fees, slippage, universe filters, data paths |
| `data/__init__.py` | Package init |
| `data/fetcher.py` | Pull OHLCV + funding rate from Binance via CCXT, save as Parquet |
| `data/universe.py` | Dynamic coin selection (top N by volume, filter stablecoins/new coins) |
| `strategy/__init__.py` | Package init |
| `strategy/base.py` | Strategy base class + `discover_strategies()` auto-loader |
| `strategy/funding_arb.py` | Funding rate arbitrage strategy |
| `strategy/trend_following.py` | Trend following strategy |
| `strategy/mean_reversion.py` | Mean reversion strategy |
| `backtest/__init__.py` | Package init |
| `backtest/metrics.py` | Compute Sharpe, MDD, Calmar, win rate, profit factor, etc. |
| `backtest/benchmark.py` | Fetch and compute BTC/QQQ/Gold buy-and-hold returns |
| `backtest/engine.py` | Vectorized backtest: signals → positions → equity curve → report JSON |
| `optimize/__init__.py` | Package init |
| `optimize/optimizer.py` | Optuna-based Bayesian parameter optimization |
| `main.py` | CLI entry point: fetch, backtest, optimize subcommands |

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `config/settings.yaml`
- Create: `data/__init__.py`
- Create: `strategy/__init__.py`
- Create: `backtest/__init__.py`
- Create: `optimize/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
ccxt>=4.0
pandas>=2.0
numpy>=1.24
optuna>=3.0
yfinance>=0.2
pyarrow>=14.0
pyyaml>=6.0
```

- [ ] **Step 2: Create config/settings.yaml**

```yaml
# Sharpe One global configuration

data:
  base_dir: "data/raw"
  default_timeframes: ["1h", "4h", "1d"]
  benchmark_dir: "data/benchmarks"

universe:
  exchange: "binance"
  quote: "USDT"
  market_type: "swap"  # perpetual futures
  top_n: 30
  min_listing_days: 90
  exclude_stablecoins: true
  stablecoin_keywords: ["BUSD", "USDC", "DAI", "TUSD", "FDUSD"]

backtest:
  initial_capital: 10000
  maker_fee: 0.0002    # 0.02%
  taker_fee: 0.0005    # 0.05%
  slippage: 0.0001     # 0.01%
  train_ratio: 0.7     # 70% train, 30% test
  default_timeframe: "1h"

optimize:
  n_trials: 200
  timeout: 600         # seconds
  direction: "maximize"
  metric: "sharpe_ratio"

results:
  output_dir: "results"
```

- [ ] **Step 3: Create package __init__.py files**

Create empty `__init__.py` in `data/`, `strategy/`, `backtest/`, `optimize/`.

- [ ] **Step 4: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: all packages install successfully.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt config/settings.yaml data/__init__.py strategy/__init__.py backtest/__init__.py optimize/__init__.py
git commit -m "feat: project setup with dependencies and config"
```

---

### Task 2: Data Fetcher

**Files:**
- Create: `data/fetcher.py`

- [ ] **Step 1: Create data/fetcher.py**

```python
"""Fetch historical OHLCV and funding rate data from Binance via CCXT."""

import os
import time
import logging

import ccxt
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_settings():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


def get_exchange():
    return ccxt.binance({"options": {"defaultType": "swap"}})


def fetch_ohlcv(exchange, symbol: str, timeframe: str, since_ms: int | None = None) -> pd.DataFrame:
    """Fetch all OHLCV data for a symbol/timeframe from Binance.
    
    Paginates through history using since parameter.
    Returns DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    all_candles = []
    limit = 1000  # Binance max per request

    if since_ms is None:
        # Start from 2020-01-01 to get max history
        since_ms = int(pd.Timestamp("2020-01-01").timestamp() * 1000)

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        except ccxt.RateLimitExceeded:
            logger.warning("Rate limit hit, sleeping 10s...")
            time.sleep(10)
            continue
        except ccxt.NetworkError as e:
            logger.warning(f"Network error: {e}, retrying in 5s...")
            time.sleep(5)
            continue

        if not candles:
            break

        all_candles.extend(candles)
        since_ms = candles[-1][0] + 1  # next ms after last candle

        if len(candles) < limit:
            break

        # Respect rate limits
        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_funding_rate(exchange, symbol: str, since_ms: int | None = None) -> pd.DataFrame:
    """Fetch historical funding rate data for a symbol."""
    all_rates = []
    limit = 1000

    if since_ms is None:
        since_ms = int(pd.Timestamp("2020-01-01").timestamp() * 1000)

    while True:
        try:
            rates = exchange.fetch_funding_rate_history(symbol, since=since_ms, limit=limit)
        except ccxt.RateLimitExceeded:
            logger.warning("Rate limit hit, sleeping 10s...")
            time.sleep(10)
            continue
        except ccxt.NetworkError as e:
            logger.warning(f"Network error: {e}, retrying in 5s...")
            time.sleep(5)
            continue
        except Exception as e:
            logger.warning(f"Failed to fetch funding rate for {symbol}: {e}")
            break

        if not rates:
            break

        all_rates.extend(rates)
        since_ms = rates[-1]["timestamp"] + 1

        if len(rates) < limit:
            break

        time.sleep(exchange.rateLimit / 1000)

    if not all_rates:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df = pd.DataFrame([
        {"timestamp": r["timestamp"], "funding_rate": r["fundingRate"]}
        for r in all_rates
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def save_parquet(df: pd.DataFrame, symbol: str, data_type: str, base_dir: str = "data/raw"):
    """Save DataFrame as Parquet file."""
    # Convert symbol: BTC/USDT:USDT -> BTCUSDT
    clean_symbol = symbol.replace("/", "").replace(":USDT", "")
    dir_path = os.path.join(base_dir, clean_symbol)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{data_type}.parquet")

    if os.path.exists(file_path):
        existing = pd.read_parquet(file_path)
        df = pd.concat([existing, df]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df.to_parquet(file_path, index=False)
    logger.info(f"Saved {len(df)} rows to {file_path}")
    return file_path


def get_last_timestamp(symbol: str, data_type: str, base_dir: str = "data/raw") -> int | None:
    """Get the last timestamp in an existing Parquet file for incremental updates."""
    clean_symbol = symbol.replace("/", "").replace(":USDT", "")
    file_path = os.path.join(base_dir, clean_symbol, f"{data_type}.parquet")

    if not os.path.exists(file_path):
        return None

    df = pd.read_parquet(file_path, columns=["timestamp"])
    if df.empty:
        return None

    last_ts = df["timestamp"].max()
    return int(last_ts.timestamp() * 1000) + 1


def fetch_all(symbols: list[str], timeframes: list[str] | None = None, base_dir: str = "data/raw"):
    """Fetch OHLCV and funding rate data for all symbols."""
    settings = load_settings()
    if timeframes is None:
        timeframes = settings["data"]["default_timeframes"]

    exchange = get_exchange()
    exchange.load_markets()

    total = len(symbols) * (len(timeframes) + 1)  # +1 for funding rate
    done = 0

    for symbol in symbols:
        for tf in timeframes:
            done += 1
            since_ms = get_last_timestamp(symbol, tf, base_dir)
            action = "Updating" if since_ms else "Fetching"
            logger.info(f"[{done}/{total}] {action} {symbol} {tf}...")

            df = fetch_ohlcv(exchange, symbol, tf, since_ms)
            if not df.empty:
                save_parquet(df, symbol, tf, base_dir)
            else:
                logger.info(f"  No new data for {symbol} {tf}")

        # Funding rate
        done += 1
        since_ms = get_last_timestamp(symbol, "funding_rate", base_dir)
        action = "Updating" if since_ms else "Fetching"
        logger.info(f"[{done}/{total}] {action} {symbol} funding_rate...")

        df = fetch_funding_rate(exchange, symbol, since_ms)
        if not df.empty:
            save_parquet(df, symbol, "funding_rate", base_dir)

    logger.info("Data fetch complete.")
```

- [ ] **Step 2: Verify fetcher loads without errors**

Run: `python -c "from data.fetcher import fetch_all; print('OK')"`
Expected: prints `OK`

- [ ] **Step 3: Commit**

```bash
git add data/fetcher.py
git commit -m "feat: add Binance data fetcher with OHLCV and funding rate support"
```

---

### Task 3: Universe Selection

**Files:**
- Create: `data/universe.py`

- [ ] **Step 1: Create data/universe.py**

```python
"""Dynamic coin universe selection based on volume and listing age."""

import logging
from datetime import datetime, timezone

import ccxt
import yaml

logger = logging.getLogger(__name__)


def load_settings():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


def get_universe(settings: dict | None = None) -> list[str]:
    """Select top coins by 24h volume from Binance USDT perpetuals.
    
    Returns list of CCXT symbols like ['BTC/USDT:USDT', 'ETH/USDT:USDT', ...].
    """
    if settings is None:
        settings = load_settings()

    cfg = settings["universe"]

    exchange = ccxt.binance({"options": {"defaultType": cfg["market_type"]}})
    exchange.load_markets()

    now = datetime.now(timezone.utc)
    min_age_ms = cfg["min_listing_days"] * 24 * 60 * 60 * 1000
    stablecoin_keywords = [kw.upper() for kw in cfg.get("stablecoin_keywords", [])]

    candidates = []
    for symbol, market in exchange.markets.items():
        # Only USDT-margined perpetuals
        if market.get("quote") != cfg["quote"]:
            continue
        if market.get("type") != "swap":
            continue
        if not market.get("active", True):
            continue

        # Exclude stablecoins
        base = market.get("base", "").upper()
        if any(kw in base for kw in stablecoin_keywords):
            continue

        # Check listing age if info available
        listing_ts = market.get("info", {}).get("onboardDate")
        if listing_ts:
            listing_time = datetime.fromtimestamp(int(listing_ts) / 1000, tz=timezone.utc)
            age = now - listing_time
            if age.total_seconds() * 1000 < min_age_ms:
                continue

        candidates.append(symbol)

    # Fetch 24h tickers to sort by volume
    tickers = exchange.fetch_tickers(candidates)

    volume_data = []
    for symbol in candidates:
        ticker = tickers.get(symbol)
        if ticker and ticker.get("quoteVolume"):
            volume_data.append((symbol, ticker["quoteVolume"]))

    # Sort by volume descending, take top N
    volume_data.sort(key=lambda x: x[1], reverse=True)
    selected = [s for s, _ in volume_data[:cfg["top_n"]]]

    logger.info(f"Universe selected: {len(selected)} coins (top {cfg['top_n']} by 24h volume)")
    for i, s in enumerate(selected[:10]):
        vol = next(v for sym, v in volume_data if sym == s)
        logger.info(f"  {i+1}. {s} — ${vol:,.0f}")

    return selected
```

- [ ] **Step 2: Verify universe module loads**

Run: `python -c "from data.universe import get_universe; print('OK')"`
Expected: prints `OK`

- [ ] **Step 3: Commit**

```bash
git add data/universe.py
git commit -m "feat: add dynamic coin universe selection by volume"
```

---

### Task 4: Strategy Base Class

**Files:**
- Create: `strategy/base.py`

- [ ] **Step 1: Create strategy/base.py**

```python
"""Strategy base class and auto-discovery."""

import importlib
import inspect
import os
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class Strategy:
    """Base class for all strategies.
    
    Subclasses must implement generate_signals().
    Define params (dict of defaults) and param_space (dict of (min, max) tuples)
    for parameter optimization.
    """

    name: str = "base"
    params: dict = {}
    param_space: dict = {}

    def __init__(self, **kwargs):
        # Override default params with any provided kwargs
        self.params = {**self.__class__.params, **kwargs}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals from OHLCV data.
        
        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume].
                May also contain funding_rate column.
        
        Returns:
            Series with same index as df. Values: 1 (long), -1 (short), 0 (flat).
        """
        raise NotImplementedError

    def required_data(self) -> dict:
        """Declare required data types and timeframes.
        
        Returns:
            Dict like {"ohlcv": ["1h"], "funding_rate": True}
        """
        return {"ohlcv": ["1h"]}


def discover_strategies() -> list[type[Strategy]]:
    """Auto-discover all Strategy subclasses in the strategy/ directory.
    
    Scans all .py files in strategy/ (excluding base.py, __init__.py),
    imports them, and collects classes that inherit from Strategy.
    """
    strategy_dir = os.path.dirname(__file__)
    strategies = []

    for filename in sorted(os.listdir(strategy_dir)):
        if not filename.endswith(".py"):
            continue
        if filename in ("base.py", "__init__.py"):
            continue

        module_name = f"strategy.{filename[:-3]}"
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            logger.error(f"Failed to import {module_name}: {e}")
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Strategy) and obj is not Strategy:
                strategies.append(obj)
                logger.info(f"Discovered strategy: {obj.name}")

    return strategies
```

- [ ] **Step 2: Verify base module loads**

Run: `python -c "from strategy.base import Strategy, discover_strategies; print('OK')"`
Expected: prints `OK`

- [ ] **Step 3: Commit**

```bash
git add strategy/base.py
git commit -m "feat: add strategy base class with auto-discovery"
```

---

### Task 5: Performance Metrics

**Files:**
- Create: `backtest/metrics.py`

- [ ] **Step 1: Create backtest/metrics.py**

```python
"""Performance metrics for backtest results."""

import numpy as np
import pandas as pd


def compute_metrics(equity_curve: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    """Compute all performance metrics from an equity curve.
    
    Args:
        equity_curve: Series of portfolio values indexed by timestamp.
        trades: Optional DataFrame of trades with columns [timestamp, side, pnl].
    
    Returns:
        Dict of metric name -> value.
    """
    returns = equity_curve.pct_change().dropna()

    # Handle edge cases
    if len(returns) < 2 or equity_curve.iloc[-1] == 0:
        return _empty_metrics()

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # Annualization factor: estimate bars per year from data frequency
    if len(equity_curve) > 1:
        dt = pd.Series(equity_curve.index).diff().dropna().median()
        if hasattr(dt, 'total_seconds'):
            seconds = dt.total_seconds()
        else:
            seconds = 3600  # default to 1h
        bars_per_year = 365.25 * 24 * 3600 / max(seconds, 1)
    else:
        bars_per_year = 8760  # 1h bars

    annualized_return = (1 + total_return) ** (bars_per_year / len(returns)) - 1

    # Sharpe Ratio (annualized, assuming risk-free rate = 0)
    if returns.std() > 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(bars_per_year)
    else:
        sharpe_ratio = 0.0

    # Max Drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown = abs(drawdown.min())

    # Calmar Ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

    # Trade-based metrics
    if trades is not None and not trades.empty:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        win_rate = len(wins) / len(trades) if len(trades) > 0 else 0.0
        gross_profit = wins["pnl"].sum() if not wins.empty else 0.0
        gross_loss = abs(losses["pnl"].sum()) if not losses.empty else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        total_trades = len(trades)
    else:
        # Estimate from position changes
        position_changes = (returns != 0).sum()
        win_rate = (returns[returns != 0] > 0).mean() if position_changes > 0 else 0.0
        profit_factor = 0.0
        total_trades = 0

    return {
        "total_return": round(total_return, 6),
        "annualized_return": round(annualized_return, 6),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown": round(max_drawdown, 6),
        "calmar_ratio": round(calmar_ratio, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "total_trades": total_trades,
        "bars": len(equity_curve),
    }


def _empty_metrics() -> dict:
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "calmar_ratio": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "total_trades": 0,
        "bars": 0,
    }
```

- [ ] **Step 2: Verify metrics module loads and computes**

Run:
```bash
python -c "
import pandas as pd
import numpy as np
from backtest.metrics import compute_metrics
# Simulate a simple equity curve
np.random.seed(42)
idx = pd.date_range('2023-01-01', periods=1000, freq='h')
returns = np.random.normal(0.0001, 0.01, 1000)
equity = pd.Series(10000 * np.cumprod(1 + returns), index=idx)
m = compute_metrics(equity)
print(f'Sharpe: {m[\"sharpe_ratio\"]}, MDD: {m[\"max_drawdown\"]:.2%}')
"
```
Expected: prints Sharpe and MDD values without errors.

- [ ] **Step 3: Commit**

```bash
git add backtest/metrics.py
git commit -m "feat: add performance metrics (Sharpe, MDD, Calmar, etc.)"
```

---

### Task 6: Benchmark Comparison

**Files:**
- Create: `backtest/benchmark.py`

- [ ] **Step 1: Create backtest/benchmark.py**

```python
"""Benchmark comparison: BTC, QQQ, Gold buy-and-hold returns."""

import os
import logging

import pandas as pd
import yfinance as yf

from backtest.metrics import compute_metrics

logger = logging.getLogger(__name__)


def fetch_benchmark_data(start_date: str, end_date: str, base_dir: str = "data/benchmarks") -> dict[str, pd.DataFrame]:
    """Fetch benchmark price data for BTC, QQQ, and Gold.
    
    Args:
        start_date: ISO date string like '2023-01-01'
        end_date: ISO date string like '2024-01-01'
        base_dir: Directory to cache benchmark data
    
    Returns:
        Dict of benchmark name -> DataFrame with columns [timestamp, close].
    """
    os.makedirs(base_dir, exist_ok=True)

    benchmarks = {
        "BTC": "BTC-USD",
        "QQQ": "QQQ",
        "Gold": "GC=F",
    }

    result = {}
    for name, ticker in benchmarks.items():
        cache_path = os.path.join(base_dir, f"{name}.parquet")

        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        except Exception as e:
            logger.warning(f"Failed to fetch {name} ({ticker}): {e}")
            continue

        if df.empty:
            logger.warning(f"No data for {name} ({ticker})")
            continue

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        bench = pd.DataFrame({
            "timestamp": df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC"),
            "close": df["Close"].values,
        }).reset_index(drop=True)

        bench.to_parquet(cache_path, index=False)
        result[name] = bench
        logger.info(f"Fetched {name}: {len(bench)} days from {start_date} to {end_date}")

    return result


def compute_benchmark_metrics(benchmarks: dict[str, pd.DataFrame], initial_capital: float = 10000) -> dict[str, dict]:
    """Compute buy-and-hold metrics for each benchmark.
    
    Args:
        benchmarks: Dict from fetch_benchmark_data().
        initial_capital: Starting capital for equity curve calculation.
    
    Returns:
        Dict of benchmark name -> metrics dict.
    """
    result = {}
    for name, df in benchmarks.items():
        if df.empty or len(df) < 2:
            continue

        # Compute equity curve from daily closes
        returns = df["close"].pct_change().fillna(0)
        equity = pd.Series(
            initial_capital * (1 + returns).cumprod().values,
            index=pd.DatetimeIndex(df["timestamp"]),
        )

        metrics = compute_metrics(equity)
        result[name] = metrics
        logger.info(f"{name} buy-and-hold: return={metrics['total_return']:.2%}, sharpe={metrics['sharpe_ratio']:.2f}")

    return result


def compare_with_benchmarks(strategy_metrics: dict, benchmark_metrics: dict[str, dict]) -> dict:
    """Compare strategy performance against all benchmarks.
    
    Returns:
        Dict with excess returns and whether strategy beats each benchmark.
    """
    comparison = {}
    beats_all = True

    for name, bench in benchmark_metrics.items():
        excess = strategy_metrics["total_return"] - bench["total_return"]
        beats = strategy_metrics["total_return"] > bench["total_return"]
        if not beats:
            beats_all = False

        comparison[name] = {
            "benchmark_return": bench["total_return"],
            "excess_return": round(excess, 6),
            "beats": beats,
            "benchmark_sharpe": bench["sharpe_ratio"],
        }

    comparison["beats_all"] = beats_all
    return comparison
```

- [ ] **Step 2: Verify benchmark module loads**

Run: `python -c "from backtest.benchmark import fetch_benchmark_data, compare_with_benchmarks; print('OK')"`
Expected: prints `OK`

- [ ] **Step 3: Commit**

```bash
git add backtest/benchmark.py
git commit -m "feat: add BTC/QQQ/Gold benchmark comparison"
```

---

### Task 7: Backtest Engine

**Files:**
- Create: `backtest/engine.py`

- [ ] **Step 1: Create backtest/engine.py**

```python
"""Vectorized backtest engine."""

import json
import os
import logging
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import yaml

from backtest.metrics import compute_metrics
from backtest.benchmark import fetch_benchmark_data, compute_benchmark_metrics, compare_with_benchmarks

logger = logging.getLogger(__name__)


def load_settings():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


def load_data(symbol: str, timeframe: str, base_dir: str = "data/raw") -> pd.DataFrame:
    """Load OHLCV Parquet data for a symbol."""
    clean_symbol = symbol.replace("/", "").replace(":USDT", "")
    path = os.path.join(base_dir, clean_symbol, f"{timeframe}.parquet")

    if not os.path.exists(path):
        logger.warning(f"No data file: {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_funding_rate(symbol: str, base_dir: str = "data/raw") -> pd.DataFrame:
    """Load funding rate Parquet data for a symbol."""
    clean_symbol = symbol.replace("/", "").replace(":USDT", "")
    path = os.path.join(base_dir, clean_symbol, "funding_rate.parquet")

    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets chronologically."""
    n = len(df)
    split_idx = int(n * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def run_backtest(strategy, df: pd.DataFrame, settings: dict | None = None) -> dict:
    """Run vectorized backtest for a single strategy on a single DataFrame.
    
    Args:
        strategy: Strategy instance with generate_signals() method.
        df: OHLCV DataFrame.
        settings: Config dict (loaded from settings.yaml if None).
    
    Returns:
        Dict with equity_curve, metrics, signals summary.
    """
    if settings is None:
        settings = load_settings()

    cfg = settings["backtest"]

    # Generate signals
    signals = strategy.generate_signals(df)

    # Delay signals by 1 bar to prevent look-ahead bias
    # Signal at bar N → execute at bar N+1 open
    positions = signals.shift(1).fillna(0)

    # Calculate returns: position * price change (using open-to-open)
    price_returns = df["open"].pct_change().fillna(0)
    strategy_returns = positions * price_returns

    # Deduct trading costs on position changes
    position_changes = positions.diff().abs().fillna(0)
    # Use taker fee for simplicity (market orders at signal change)
    cost_per_trade = cfg["taker_fee"] + cfg["slippage"]
    costs = position_changes * cost_per_trade
    strategy_returns = strategy_returns - costs

    # Build equity curve
    equity_curve = pd.Series(
        cfg["initial_capital"] * (1 + strategy_returns).cumprod().values,
        index=df["timestamp"],
    )

    # Compute metrics
    metrics = compute_metrics(equity_curve)

    # Signal statistics
    signal_counts = signals.value_counts().to_dict()

    return {
        "equity_curve": equity_curve,
        "metrics": metrics,
        "signals": {
            "long": int(signal_counts.get(1, 0)),
            "short": int(signal_counts.get(-1, 0)),
            "flat": int(signal_counts.get(0, 0)),
        },
        "params": strategy.params,
    }


def run_full_backtest(strategy_class, symbols: list[str], settings: dict | None = None) -> dict:
    """Run backtest for a strategy across multiple symbols with train/test split.
    
    Args:
        strategy_class: Strategy class (not instance).
        symbols: List of symbols to test on.
        settings: Config dict.
    
    Returns:
        Dict with per-symbol results, aggregate metrics, and benchmark comparison.
    """
    if settings is None:
        settings = load_settings()

    cfg = settings["backtest"]
    strategy = strategy_class()
    timeframe = cfg["default_timeframe"]
    req = strategy.required_data()
    timeframes = req.get("ohlcv", [timeframe])
    needs_funding = req.get("funding_rate", False)

    # Use the first (primary) timeframe
    primary_tf = timeframes[0] if timeframes else timeframe

    all_train_equity = []
    all_test_equity = []
    per_symbol = {}

    for symbol in symbols:
        df = load_data(symbol, primary_tf, settings["data"]["base_dir"])
        if df.empty or len(df) < 100:
            logger.info(f"Skipping {symbol}: insufficient data ({len(df)} bars)")
            continue

        # Merge funding rate if needed
        if needs_funding:
            fr = load_funding_rate(symbol, settings["data"]["base_dir"])
            if not fr.empty:
                df = pd.merge_asof(
                    df.sort_values("timestamp"),
                    fr.sort_values("timestamp"),
                    on="timestamp",
                    direction="backward",
                )
                df["funding_rate"] = df["funding_rate"].fillna(0)

        train_df, test_df = split_train_test(df, cfg["train_ratio"])

        train_result = run_backtest(strategy, train_df, settings)
        test_result = run_backtest(strategy, test_df, settings)

        per_symbol[symbol] = {
            "train": train_result["metrics"],
            "test": test_result["metrics"],
            "signals": test_result["signals"],
        }

        all_train_equity.append(train_result["equity_curve"])
        all_test_equity.append(test_result["equity_curve"])

    if not all_test_equity:
        logger.warning(f"No valid symbols for {strategy.name}")
        return {"strategy": strategy.name, "error": "no valid data"}

    # Aggregate: average metrics across symbols
    train_metrics = _average_metrics([s["train"] for s in per_symbol.values()])
    test_metrics = _average_metrics([s["test"] for s in per_symbol.values()])

    # Benchmark comparison (use test period date range)
    test_start = None
    test_end = None
    for eq in all_test_equity:
        if test_start is None or eq.index.min() < test_start:
            test_start = eq.index.min()
        if test_end is None or eq.index.max() > test_end:
            test_end = eq.index.max()

    benchmark_comparison = {}
    if test_start and test_end:
        try:
            bench_data = fetch_benchmark_data(
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
                settings["data"].get("benchmark_dir", "data/benchmarks"),
            )
            bench_metrics = compute_benchmark_metrics(bench_data, cfg["initial_capital"])
            benchmark_comparison = compare_with_benchmarks(test_metrics, bench_metrics)
        except Exception as e:
            logger.warning(f"Benchmark comparison failed: {e}")

    report = {
        "strategy": strategy.name,
        "params": strategy.params,
        "symbols_tested": len(per_symbol),
        "timeframe": primary_tf,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "benchmark_comparison": benchmark_comparison,
        "per_symbol": {k: v for k, v in per_symbol.items()},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return report


def save_report(report: dict, output_dir: str = "results"):
    """Save backtest report as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{report['strategy']}_{ts}.json"
    path = os.path.join(output_dir, filename)

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Report saved to {path}")
    return path


def _average_metrics(metrics_list: list[dict]) -> dict:
    """Average metrics across multiple symbols."""
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    result = {}
    for key in keys:
        values = [m[key] for m in metrics_list if isinstance(m.get(key), (int, float))]
        if values:
            result[key] = round(sum(values) / len(values), 6)
        else:
            result[key] = 0

    return result
```

- [ ] **Step 2: Verify engine module loads**

Run: `python -c "from backtest.engine import run_backtest, run_full_backtest; print('OK')"`
Expected: prints `OK`

- [ ] **Step 3: Commit**

```bash
git add backtest/engine.py
git commit -m "feat: add vectorized backtest engine with train/test split"
```

---

### Task 8: Initial Strategies

**Files:**
- Create: `strategy/funding_arb.py`
- Create: `strategy/trend_following.py`
- Create: `strategy/mean_reversion.py`

- [ ] **Step 1: Create strategy/funding_arb.py**

```python
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
```

- [ ] **Step 2: Create strategy/trend_following.py**

```python
"""Trend following strategy.

Uses dual moving average crossover with ATR-based volatility filter.
Long when fast MA > slow MA and trend is strong, short when opposite.
"""

import pandas as pd
import numpy as np
from strategy.base import Strategy


class TrendFollowingStrategy(Strategy):
    name = "trend_following"
    params = {
        "fast_period": 20,
        "slow_period": 60,
        "atr_period": 14,
        "atr_threshold": 1.0,   # Minimum ATR multiplier to confirm trend
        "use_ema": True,        # Use EMA instead of SMA
    }
    param_space = {
        "fast_period": (5, 50),
        "slow_period": (20, 200),
        "atr_period": (7, 30),
        "atr_threshold": (0.5, 3.0),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Moving averages
        if p["use_ema"]:
            fast_ma = close.ewm(span=p["fast_period"], adjust=False).mean()
            slow_ma = close.ewm(span=p["slow_period"], adjust=False).mean()
        else:
            fast_ma = close.rolling(window=p["fast_period"], min_periods=1).mean()
            slow_ma = close.rolling(window=p["slow_period"], min_periods=1).mean()

        # ATR for volatility filter
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=p["atr_period"], min_periods=1).mean()

        # Trend strength: distance between MAs relative to ATR
        ma_distance = (fast_ma - slow_ma).abs()
        trend_strong = ma_distance > (atr * p["atr_threshold"])

        # Signals
        signals = pd.Series(0, index=df.index)
        signals[(fast_ma > slow_ma) & trend_strong] = 1    # Long
        signals[(fast_ma < slow_ma) & trend_strong] = -1   # Short

        return signals

    def required_data(self) -> dict:
        return {"ohlcv": ["1h"]}
```

- [ ] **Step 3: Create strategy/mean_reversion.py**

```python
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
```

- [ ] **Step 4: Verify all strategies auto-discover**

Run:
```bash
python -c "
from strategy.base import discover_strategies
strategies = discover_strategies()
for s in strategies:
    print(f'{s.name}: params={list(s.params.keys())}')
"
```
Expected: prints all 3 strategies with their parameter names.

- [ ] **Step 5: Commit**

```bash
git add strategy/funding_arb.py strategy/trend_following.py strategy/mean_reversion.py
git commit -m "feat: add initial strategies (funding arb, trend following, mean reversion)"
```

---

### Task 9: Parameter Optimizer

**Files:**
- Create: `optimize/optimizer.py`

- [ ] **Step 1: Create optimize/optimizer.py**

```python
"""Bayesian parameter optimization using optuna."""

import json
import os
import logging
from datetime import datetime, timezone

import optuna
import yaml

from backtest.engine import load_data, load_funding_rate, split_train_test, run_backtest

logger = logging.getLogger(__name__)

# Suppress optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_settings():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


def optimize_strategy(strategy_class, symbols: list[str], settings: dict | None = None) -> dict:
    """Run Bayesian optimization on a strategy's parameter space.
    
    Optimizes on train set, evaluates best params on test set.
    
    Args:
        strategy_class: Strategy class with param_space defined.
        symbols: List of symbols to optimize across.
        settings: Config dict.
    
    Returns:
        Dict with best params, train/test metrics, and optimization history.
    """
    if settings is None:
        settings = load_settings()

    cfg_bt = settings["backtest"]
    cfg_opt = settings["optimize"]

    # Pre-load all data
    primary_tf = cfg_bt["default_timeframe"]
    strategy_temp = strategy_class()
    req = strategy_temp.required_data()
    needs_funding = req.get("funding_rate", False)
    timeframes = req.get("ohlcv", [primary_tf])
    primary_tf = timeframes[0] if timeframes else primary_tf

    data_cache = {}
    for symbol in symbols:
        df = load_data(symbol, primary_tf, settings["data"]["base_dir"])
        if df.empty or len(df) < 100:
            continue

        if needs_funding:
            fr = load_funding_rate(symbol, settings["data"]["base_dir"])
            if not fr.empty:
                df = df.sort_values("timestamp")
                fr = fr.sort_values("timestamp")
                df = df.merge(fr, on="timestamp", how="left")
                df["funding_rate"] = df["funding_rate"].fillna(0)

        train_df, test_df = split_train_test(df, cfg_bt["train_ratio"])
        data_cache[symbol] = {"train": train_df, "test": test_df}

    if not data_cache:
        return {"strategy": strategy_class.name, "error": "no valid data"}

    param_space = strategy_class.param_space

    def objective(trial):
        # Sample parameters from param_space
        params = {}
        for key, (low, high) in param_space.items():
            if isinstance(low, int) and isinstance(high, int):
                params[key] = trial.suggest_int(key, low, high)
            else:
                params[key] = trial.suggest_float(key, float(low), float(high))

        # Run backtest on train set for all symbols
        sharpe_values = []
        for symbol, data in data_cache.items():
            strategy = strategy_class(**params)
            result = run_backtest(strategy, data["train"], settings)
            sharpe_values.append(result["metrics"]["sharpe_ratio"])

        # Return average Sharpe across symbols
        return sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0

    # Run optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=cfg_opt["n_trials"],
        timeout=cfg_opt.get("timeout"),
    )

    best_params = study.best_params
    best_train_sharpe = study.best_value

    # Evaluate best params on test set
    test_metrics_list = []
    train_metrics_list = []
    for symbol, data in data_cache.items():
        strategy = strategy_class(**best_params)
        train_result = run_backtest(strategy, data["train"], settings)
        test_result = run_backtest(strategy, data["test"], settings)
        train_metrics_list.append(train_result["metrics"])
        test_metrics_list.append(test_result["metrics"])

    def avg_metrics(metrics_list):
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        return {
            k: round(sum(m[k] for m in metrics_list) / len(metrics_list), 6)
            for k in keys
            if isinstance(metrics_list[0].get(k), (int, float))
        }

    report = {
        "strategy": strategy_class.name,
        "best_params": best_params,
        "default_params": strategy_class.params,
        "train_metrics": avg_metrics(train_metrics_list),
        "test_metrics": avg_metrics(test_metrics_list),
        "optimization": {
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "best_train_sharpe": round(best_train_sharpe, 4),
        },
        "symbols_used": list(data_cache.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save report
    os.makedirs(settings["results"]["output_dir"], exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(settings["results"]["output_dir"], f"optimize_{strategy_class.name}_{ts}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Optimization complete for {strategy_class.name}")
    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Train Sharpe: {report['train_metrics'].get('sharpe_ratio', 'N/A')}")
    logger.info(f"  Test Sharpe: {report['test_metrics'].get('sharpe_ratio', 'N/A')}")
    logger.info(f"  Report saved to {path}")

    return report
```

- [ ] **Step 2: Verify optimizer module loads**

Run: `python -c "from optimize.optimizer import optimize_strategy; print('OK')"`
Expected: prints `OK`

- [ ] **Step 3: Commit**

```bash
git add optimize/optimizer.py
git commit -m "feat: add Bayesian parameter optimizer with optuna"
```

---

### Task 10: Main Entry Point

**Files:**
- Create: `main.py`

- [ ] **Step 1: Create main.py**

```python
"""Sharpe One entry point.

Usage:
    python main.py fetch                          # Fetch data for universe
    python main.py fetch --symbols BTCUSDT ETHUSDT # Fetch specific symbols
    python main.py backtest                        # Backtest all strategies
    python main.py backtest --strategy trend_following  # Backtest single strategy
    python main.py optimize --strategy trend_following  # Optimize single strategy
"""

import argparse
import json
import logging
import sys

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_settings():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


def cmd_fetch(args):
    """Fetch historical data from Binance."""
    from data.fetcher import fetch_all
    from data.universe import get_universe

    settings = load_settings()

    if args.symbols:
        # Convert clean symbols to CCXT format
        symbols = [f"{s.replace('USDT', '')}/USDT:USDT" for s in args.symbols]
    else:
        logger.info("Selecting coin universe...")
        symbols = get_universe(settings)

    logger.info(f"Fetching data for {len(symbols)} symbols...")
    fetch_all(symbols, args.timeframes, settings["data"]["base_dir"])


def cmd_backtest(args):
    """Run backtest for strategies."""
    from strategy.base import discover_strategies
    from backtest.engine import run_full_backtest, save_report
    from data.universe import get_universe

    settings = load_settings()

    # Get symbols
    if args.symbols:
        symbols = [f"{s.replace('USDT', '')}/USDT:USDT" for s in args.symbols]
    else:
        # Use locally available data instead of fetching universe
        import os
        base_dir = settings["data"]["base_dir"]
        if os.path.exists(base_dir):
            local_symbols = [
                f"{d}/USDT:USDT" for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            ]
            symbols = local_symbols if local_symbols else []
        else:
            symbols = []

    if not symbols:
        logger.error("No data available. Run 'python main.py fetch' first.")
        sys.exit(1)

    # Discover strategies
    all_strategies = discover_strategies()
    if args.strategy:
        all_strategies = [s for s in all_strategies if s.name == args.strategy]
        if not all_strategies:
            logger.error(f"Strategy '{args.strategy}' not found.")
            sys.exit(1)

    # Run backtests
    for strategy_class in all_strategies:
        logger.info(f"\n{'='*60}")
        logger.info(f"Backtesting: {strategy_class.name}")
        logger.info(f"{'='*60}")

        report = run_full_backtest(strategy_class, symbols, settings)

        if "error" in report:
            logger.warning(f"  Error: {report['error']}")
            continue

        # Print summary
        test = report.get("test_metrics", {})
        train = report.get("train_metrics", {})
        logger.info(f"\n  TRAIN: Sharpe={train.get('sharpe_ratio', 'N/A')}, "
                     f"Return={train.get('total_return', 0):.2%}, "
                     f"MDD={train.get('max_drawdown', 0):.2%}")
        logger.info(f"  TEST:  Sharpe={test.get('sharpe_ratio', 'N/A')}, "
                     f"Return={test.get('total_return', 0):.2%}, "
                     f"MDD={test.get('max_drawdown', 0):.2%}")

        bench = report.get("benchmark_comparison", {})
        if bench:
            beats = bench.get("beats_all", False)
            logger.info(f"  Beats all benchmarks: {'YES' if beats else 'NO'}")
            for name, b in bench.items():
                if isinstance(b, dict):
                    status = "BEAT" if b.get("beats") else "LOST"
                    logger.info(f"    {name}: {status} (excess={b.get('excess_return', 0):.2%})")

        # Save report
        path = save_report(report, settings["results"]["output_dir"])
        logger.info(f"  Report: {path}")


def cmd_optimize(args):
    """Run parameter optimization for a strategy."""
    from strategy.base import discover_strategies
    from optimize.optimizer import optimize_strategy

    settings = load_settings()

    if not args.strategy:
        logger.error("Must specify --strategy for optimization.")
        sys.exit(1)

    all_strategies = discover_strategies()
    strategy_class = next((s for s in all_strategies if s.name == args.strategy), None)
    if not strategy_class:
        logger.error(f"Strategy '{args.strategy}' not found.")
        sys.exit(1)

    # Get symbols from local data
    import os
    base_dir = settings["data"]["base_dir"]
    if args.symbols:
        symbols = [f"{s.replace('USDT', '')}/USDT:USDT" for s in args.symbols]
    elif os.path.exists(base_dir):
        symbols = [
            f"{d}/USDT:USDT" for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]
    else:
        symbols = []

    if not symbols:
        logger.error("No data available. Run 'python main.py fetch' first.")
        sys.exit(1)

    logger.info(f"Optimizing {strategy_class.name} on {len(symbols)} symbols...")
    report = optimize_strategy(strategy_class, symbols, settings)

    if "error" in report:
        logger.error(f"Optimization failed: {report['error']}")
        sys.exit(1)

    logger.info(f"\nBest params: {json.dumps(report['best_params'], indent=2)}")
    logger.info(f"Train Sharpe: {report['train_metrics'].get('sharpe_ratio', 'N/A')}")
    logger.info(f"Test Sharpe: {report['test_metrics'].get('sharpe_ratio', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="Sharpe One — Crypto Quant Fund")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # fetch
    fetch_parser = subparsers.add_parser("fetch", help="Fetch historical data from Binance")
    fetch_parser.add_argument("--symbols", nargs="+", help="Specific symbols (e.g., BTCUSDT ETHUSDT)")
    fetch_parser.add_argument("--timeframes", nargs="+", help="Timeframes to fetch")

    # backtest
    bt_parser = subparsers.add_parser("backtest", help="Run backtest")
    bt_parser.add_argument("--strategy", help="Strategy name (omit for all)")
    bt_parser.add_argument("--symbols", nargs="+", help="Specific symbols")

    # optimize
    opt_parser = subparsers.add_parser("optimize", help="Optimize strategy parameters")
    opt_parser.add_argument("--strategy", required=True, help="Strategy to optimize")
    opt_parser.add_argument("--symbols", nargs="+", help="Specific symbols")

    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify main.py loads and shows help**

Run: `python main.py --help`
Expected: shows usage with fetch/backtest/optimize subcommands.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add CLI entry point with fetch, backtest, optimize commands"
```

---

### Task 11: End-to-End Verification

- [ ] **Step 1: Fetch data for BTC and ETH**

Run: `python main.py fetch --symbols BTCUSDT ETHUSDT`
Expected: downloads historical OHLCV and funding rate data, saves to `data/raw/BTCUSDT/` and `data/raw/ETHUSDT/`.

- [ ] **Step 2: Run backtest for all strategies**

Run: `python main.py backtest --symbols BTCUSDT ETHUSDT`
Expected: runs all 3 strategies, prints train/test metrics and benchmark comparison, saves JSON reports to `results/`.

- [ ] **Step 3: Run optimization for trend_following**

Run: `python main.py optimize --strategy trend_following --symbols BTCUSDT`
Expected: runs 200 optuna trials, prints best params and train/test Sharpe, saves optimization report.

- [ ] **Step 4: Verify reports are readable**

Run:
```bash
python -c "
import json, glob
reports = sorted(glob.glob('results/*.json'))
for r in reports:
    with open(r) as f:
        data = json.load(f)
    print(f'{r}: strategy={data.get(\"strategy\")}, test_sharpe={data.get(\"test_metrics\", {}).get(\"sharpe_ratio\", \"N/A\")}')
"
```
Expected: lists all reports with strategy names and Sharpe ratios.

- [ ] **Step 5: Commit data directory gitignore and results**

Add to `.gitignore` if not present: `data/raw/`, `data/benchmarks/`. Commit any remaining changes.

```bash
git add -A
git commit -m "feat: complete sharpe_one backtest system — end-to-end verified"
```
