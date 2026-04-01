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
