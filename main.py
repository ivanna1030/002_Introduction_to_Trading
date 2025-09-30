# Entrypoint
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from utils import split
from backtest import backtest, params_backtest
from plots import plot_portfolio_value
from metrics import evaluate_metrics

def main():
    data = pd.read_csv('data/Binance_BTCUSDT_1h.csv').dropna()
    data = data.rename(columns={'Date': 'Datetime'})
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce', dayfirst=True)
    data = data.iloc[::-1].reset_index(drop=True)

    train, test, validation = split(data)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: backtest(data, trial), n_trials=10, n_jobs=-1)

    print("Best parameters:")
    print(study.best_params)

    print("Best value:")
    print(study.best_value)

    cash, portfolio_value = params_backtest(data, study.best_params, cash=1_000_000)

    print("Cash:")
    print(cash)

    print("Portfolio value:")
    print(portfolio_value[-1])

    print("Performance metrics:")
    print(evaluate_metrics(pd.Series(portfolio_value)))

    plot_portfolio_value(portfolio_value)

if __name__ == "__main__":
    main()