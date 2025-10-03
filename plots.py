import pandas as pd
import matplotlib.pyplot as plt

def plot_portfolio_value(portfolio_values):
    """
    Plot the portfolio value over time.
    
    Parameters:
        portfolio_values (list or pd.Series): Portfolio values over time.

    Returns:
        None
    """
    
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.title('Portfolio value over time (train)')
    plt.xlabel('Time')
    plt.ylabel('Portfolio value')
    plt.show()

def plot_test_validation(test_portfolio, validation_portfolio, test, validation):
    """
    Plot the portfolio value for test and validation sets.
    
    Parameters:
        test_portfolio (list or pd.Series): Portfolio values over time for the test set.
        validation_portfolio (list or pd.Series): Portfolio values over time for the validation set.
        test (pd.DataFrame): Test market data with datetime information.
        validation (pd.DataFrame): Validation market data with datetime information.

    Returns:
        None
    """
    
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