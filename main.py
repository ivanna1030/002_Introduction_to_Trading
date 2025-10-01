# Entrypoint
import pandas as pd
import optuna

from utils import modify_data, split, returns_table
from backtest import walk_forward, params_backtest
from metrics import evaluate_metrics
from plots import plot_portfolio_value, plot_test_validation

def main():
    data = pd.read_csv('data/Binance_BTCUSDT_1h.csv').dropna()
    data = modify_data(data)

    train, test, validation = split(data)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: walk_forward(train, trial, n_splits=5), n_trials=100, n_jobs=-1)

    print()

    print("\033[1mBest parameters:\033[0m")
    print(study.best_params)

    print()

    print("\033[1mBest value:\033[0m")
    print(study.best_value)

    print()

    cash_train, portfolio_value_train, win_rate_train = params_backtest(train, study.best_params, cash=1_000_000)

    print("\033[1mTrain results:\033[0m")

    print("Cash: ", cash_train)

    print("Portfolio value: ", portfolio_value_train[-1])

    print(f"Win rate: {win_rate_train:.2%}")

    print("Performance metrics:")
    print(evaluate_metrics(pd.Series(portfolio_value_train)))

    print()

    cash_test, portfolio_value_test, win_rate_test = params_backtest(test, study.best_params, cash=1_000_000)

    print("\033[1mTest results:\033[0m")

    print("Cash: ", cash_test)

    print("Portfolio value: ", portfolio_value_test[-1])

    print(f"Win rate: {win_rate_test:.2%}")

    print("Performance metrics:")
    print(evaluate_metrics(pd.Series(portfolio_value_test)))

    print()

    cash_validation, portfolio_value_validation, win_rate_validation = params_backtest(validation, study.best_params, cash_test)

    print("\033[1mValidation results:\033[0m")

    print("Cash: ", cash_validation)

    print("Portfolio value: ", portfolio_value_validation[-1])

    print(f"Win rate: {win_rate_validation:.2%}")

    print("Performance metrics:")
    print(evaluate_metrics(pd.Series(portfolio_value_validation)))

    print()

    total_portfolio = portfolio_value_test + portfolio_value_validation

    print("\033[1mPortfolio results:\033[0m")

    print("Cash: ", cash_validation)

    print("Portfolio value: ", portfolio_value_validation[-1])

    print("Performance metrics:")
    print(evaluate_metrics(pd.Series(total_portfolio)))

    print()

    print("\033[1mReturns table:\033[0m")
    print(returns_table(total_portfolio, test, validation))

    plot_portfolio_value(portfolio_value_train)
    
    plot_test_validation(portfolio_value_test, portfolio_value_validation, test, validation)

if __name__ == "__main__":
    main()