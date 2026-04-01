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
