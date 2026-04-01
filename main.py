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
