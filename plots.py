import pandas as pd
import matplotlib.pyplot as plt

def plot_portfolio_value(portfolio_values):
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.title('Portfolio value over time (train)')
    plt.xlabel('Time')
    plt.ylabel('Portfolio value')
    plt.show()

def plot_test_validation(test_portfolio, validation_portfolio, test, validation):
    test_df = pd.DataFrame({
        'Date': test['Datetime'].reset_index(drop=True),
        'Portfolio Value': test_portfolio
    })
    validation_df = pd.DataFrame({
        'Date': validation['Datetime'].reset_index(drop=True),
        'Portfolio Value': validation_portfolio
    })
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['Date'], test_df['Portfolio Value'], label='Test', color='red')
    plt.plot(validation_df['Date'], validation_df['Portfolio Value'], label='Validation', color='green')
    plt.title('Portfolio value over time (test + validation)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio value')
    plt.legend()
    plt.show()