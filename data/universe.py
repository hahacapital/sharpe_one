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
