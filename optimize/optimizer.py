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
