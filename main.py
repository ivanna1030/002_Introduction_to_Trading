# Entrypoint
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from backtest import backtest, params_backtest
from plots import plot_portfolio_value

def main():
    data = pd.read_csv('data/aapl_5m_train.csv').dropna()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: backtest(data, trial), n_trials=10, n_jobs=-1)

    print("Best parameters:")
    print(study.best_params)

    print("Best value:")
    print(study.best_value)

    cash, portfolio_value = params_backtest(data, study.best_params)

    print("Cash:")
    print(cash)

    print("Portfolio value:")
    print(portfolio_value[-1])

    historic = data.copy()
    historic['Datetime'] = pd.to_datetime(historic['Datetime'])
    historic = historic.set_index('Datetime')
    historic['Portfolio Value'] = portfolio_value

    plot_portfolio_value(historic)

if __name__ == "__main__":
    main()