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

    # Add funding rate payments if available
    # Positive funding: longs pay shorts. So adjustment = -position * rate
    # Applied at 1/8 per hour (funding settles every 8h)
    if "funding_rate" in df.columns:
        funding_adjustment = -positions * df["funding_rate"].fillna(0) / 8
        strategy_returns = strategy_returns + funding_adjustment

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
