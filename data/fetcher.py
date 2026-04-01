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
