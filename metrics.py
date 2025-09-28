import numpy as np
import pandas as pd

def sharpe_ratio(portfolio_values: pd.Series) -> float:
    returns = portfolio_values.pct_change().dropna()
    mean = returns.mean()
    std = returns.std()

    intervals = 365 * 24 * 60 / 60  # Daily intervals in 1-hour data
    annual_rets = mean * intervals
    annual_std = std * np.sqrt(intervals)

    return annual_rets / annual_std if annual_std > 0 else 0

def sortino_ratio(portfolio_values: pd.Series) -> float:
    returns = portfolio_values.pct_change().dropna()
    mean = returns.mean()
    downside = np.minimum(returns, 0).std()

    intervals = 365 * 24 * 60 / 60  # Daily intervals in 1-hour data
    annual_rets = mean * intervals
    annual_downside = downside * np.sqrt(intervals)

    return annual_rets / annual_downside if annual_downside > 0 else 0

def max_drawdown(portfolio_values: pd.Series) -> float:
    rolling_max = portfolio_values.cummax()
    drawdowns = (portfolio_values - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    return abs(max_dd)

def calmar_ratio(portfolio_values: pd.Series) -> float:
    returns = portfolio_values.pct_change().dropna()
    mean = returns.mean()

    intervals = 365 * 24 * 60 / 60  # Daily intervals in 1-hour data
    annual_rets = mean * intervals

    mdd = max_drawdown(portfolio_values)

    return annual_rets / mdd if mdd > 0 else 0

def evaluate_metrics(portfolio_values: pd.Series) -> pd.DataFrame:
    metrics = {
        'Sharpe ratio': sharpe_ratio(portfolio_values),
        'Sortino ratio': sortino_ratio(portfolio_values),
        'Max drawdown': max_drawdown(portfolio_values),
        'Calmar ratio': calmar_ratio(portfolio_values)
    }
    return pd.DataFrame([metrics])