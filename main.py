# Entrypoint
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from utils import split
from backtest import backtest, params_backtest
from plots import plot_portfolio_value, plot_test_validation
from metrics import evaluate_metrics

def main():
    data = pd.read_csv('data/Binance_BTCUSDT_1h.csv').dropna()
    data = data.rename(columns={'Date': 'Datetime'})
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce', dayfirst=True)
    data = data.iloc[::-1].reset_index(drop=True)

    train, test, validation = split(data)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: backtest(train, trial), n_trials=50, n_jobs=-1)

    print("\033[1mBest parameters:\033[0m")
    print(study.best_params)

    print("\033[1mBest value:\033[0m")
    print(study.best_value)

    cash_train, portfolio_value_train, win_rate_train = params_backtest(train, study.best_params, cash=1_000_000)

    print("\033[1mTrain results:\033[0m")

    print("Cash: ", cash_train)

    print("Portfolio value: ", portfolio_value_train[-1])

    print(f"Win rate: {win_rate_train:.2%}")

    print("Performance metrics:")
    print(evaluate_metrics(pd.Series(portfolio_value_train)))

    cash_test, portfolio_value_test, win_rate_test = params_backtest(test, study.best_params, cash=1_000_000)

    print("\033[1mTest results:\033[0m")

    print("Cash: ", cash_test)

    print("Portfolio value: ", portfolio_value_test[-1])

    print(f"Win rate: {win_rate_test:.2%}")

    print("Performance metrics:")
    print(evaluate_metrics(pd.Series(portfolio_value_test)))

    cash_validation, portfolio_value_validation, win_rate_validation = params_backtest(validation, study.best_params, cash_test)

    print("\033[1mValidation results:\033[0m")

    print("Cash: ", cash_validation)

    print("Portfolio value: ", portfolio_value_validation[-1])

    print(f"Win rate: {win_rate_validation:.2%}")

    print("Performance metrics:")
    print(evaluate_metrics(pd.Series(portfolio_value_validation)))

    plot_portfolio_value(portfolio_value_train)
    
    plot_test_validation(portfolio_value_test, portfolio_value_validation, test, validation)

if __name__ == "__main__":
    main()