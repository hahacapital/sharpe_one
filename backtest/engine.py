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

    # Delay signals by 2 bars to prevent look-ahead bias
    # Signal uses close[N] (decided at end of bar N)
    # → position active at bar N+1 open (execution bar)
    # → return earned is open[N+1] to open[N+2]
    # Using shift(2) on signals aligns with shift(1) on open returns:
    #   positions[N+2] = signals[N], price_returns[N+2] = (open[N+2]-open[N+1])/open[N+1]
    positions = signals.shift(2).fillna(0)

    # Calculate returns: position * price change (open-to-open for next bar)
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
    """Run backtest with walk-forward validation across multiple symbols.

    Uses walk-forward windows: train on train_years, test on test_months,
    then roll forward. Each symbol only enters when its data actually starts
    (no survivorship bias).
    """
    if settings is None:
        settings = load_settings()

    cfg = settings["backtest"]
    wf = cfg.get("walk_forward", {})
    train_bars = wf.get("train_bars", 17520)  # default 2 years of 1h bars
    test_bars = wf.get("test_bars", 4380)     # default 6 months of 1h bars
    step_bars = wf.get("step_bars", 4380)     # roll forward by 6 months

    strategy_instance = strategy_class()
    timeframe = cfg["default_timeframe"]
    req = strategy_instance.required_data()
    timeframes = req.get("ohlcv", [timeframe])
    needs_funding = req.get("funding_rate", False)
    primary_tf = timeframes[0] if timeframes else timeframe

    # Load all data per symbol
    symbol_data = {}
    for symbol in symbols:
        df = load_data(symbol, primary_tf, settings["data"]["base_dir"])
        if df.empty or len(df) < train_bars + test_bars:
            logger.info(f"Skipping {symbol}: insufficient data ({len(df)} bars, need {train_bars + test_bars})")
            continue

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

        symbol_data[symbol] = df

    if not symbol_data:
        logger.warning(f"No valid symbols for {strategy_instance.name}")
        return {"strategy": strategy_instance.name, "error": "no valid data"}

    # Find common date range across all symbols
    # Each symbol only participates from its own data start
    all_timestamps = set()
    symbol_start = {}
    for symbol, df in symbol_data.items():
        symbol_start[symbol] = df["timestamp"].iloc[0]
        all_timestamps.update(df["timestamp"].tolist())

    # Walk-forward: find max data length
    max_len = max(len(df) for df in symbol_data.values())

    # Collect all walk-forward windows
    all_train_metrics = []
    all_test_metrics = []
    per_window = []

    start = 0
    window_num = 0
    while start + train_bars + test_bars <= max_len:
        window_num += 1
        train_end = start + train_bars
        test_end = train_end + test_bars

        window_train_metrics = []
        window_test_metrics = []
        symbols_in_window = 0

        for symbol, df in symbol_data.items():
            # Skip if this symbol doesn't have enough data for this window
            if len(df) < test_end:
                continue

            train_df = df.iloc[start:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            if len(train_df) < 100 or len(test_df) < 100:
                continue

            symbols_in_window += 1
            strategy = strategy_class()
            train_result = run_backtest(strategy, train_df, settings)
            test_result = run_backtest(strategy, test_df, settings)

            window_train_metrics.append(train_result["metrics"])
            window_test_metrics.append(test_result["metrics"])

        if window_train_metrics:
            avg_train = _average_metrics(window_train_metrics)
            avg_test = _average_metrics(window_test_metrics)
            all_train_metrics.append(avg_train)
            all_test_metrics.append(avg_test)

            # Get window dates from the symbol with most data
            ref_symbol = max(symbol_data, key=lambda s: len(symbol_data[s]))
            ref_df = symbol_data[ref_symbol]
            window_info = {
                "window": window_num,
                "train_start": str(ref_df["timestamp"].iloc[start]),
                "train_end": str(ref_df["timestamp"].iloc[train_end - 1]),
                "test_start": str(ref_df["timestamp"].iloc[train_end]),
                "test_end": str(ref_df["timestamp"].iloc[min(test_end - 1, len(ref_df) - 1)]),
                "symbols": symbols_in_window,
                "train": avg_train,
                "test": avg_test,
            }
            per_window.append(window_info)

            logger.info(
                f"  Window {window_num}: "
                f"train {window_info['train_start'][:10]}→{window_info['train_end'][:10]} "
                f"test {window_info['test_start'][:10]}→{window_info['test_end'][:10]} | "
                f"train_sharpe={avg_train.get('sharpe_ratio', 'N/A')} "
                f"test_sharpe={avg_test.get('sharpe_ratio', 'N/A')} "
                f"({symbols_in_window} symbols)"
            )

        start += step_bars

    if not all_test_metrics:
        return {"strategy": strategy_instance.name, "error": "no valid walk-forward windows"}

    # Overall aggregation across all windows
    overall_train = _average_metrics(all_train_metrics)
    overall_test = _average_metrics(all_test_metrics)

    # Benchmark comparison using the full test period range
    first_test_start = per_window[0]["test_start"][:10]
    last_test_end = per_window[-1]["test_end"][:10]
    benchmark_comparison = {}
    try:
        bench_data = fetch_benchmark_data(
            first_test_start, last_test_end,
            settings["data"].get("benchmark_dir", "data/benchmarks"),
        )
        bench_metrics = compute_benchmark_metrics(bench_data, cfg["initial_capital"])
        benchmark_comparison = compare_with_benchmarks(overall_test, bench_metrics)
    except Exception as e:
        logger.warning(f"Benchmark comparison failed: {e}")

    report = {
        "strategy": strategy_instance.name,
        "params": strategy_instance.params,
        "symbols_tested": len(symbol_data),
        "timeframe": primary_tf,
        "walk_forward_windows": len(per_window),
        "train_metrics": overall_train,
        "test_metrics": overall_test,
        "benchmark_comparison": benchmark_comparison,
        "per_window": per_window,
        "symbol_data_ranges": {
            s: {"start": str(df["timestamp"].iloc[0]), "end": str(df["timestamp"].iloc[-1]), "bars": len(df)}
            for s, df in symbol_data.items()
        },
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
    """Average metrics across multiple items."""
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
