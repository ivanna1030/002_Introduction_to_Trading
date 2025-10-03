import pandas as pd

def modify_data(data: pd.DataFrame):
    """
    Modify and clean the input data.
    
    Parameters:
        data (pd.DataFrame): Raw market data.
    
    Returns:
        pd.DataFrame: Cleaned and modified market data.
    """

    data = data.copy()
    data = data.rename(columns={'Date': 'Datetime'})
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce', dayfirst=True)
    data = data.iloc[::-1].reset_index(drop=True)
    data = data.drop_duplicates(subset=["Open", "High", "Low", "Close", "Volume BTC", "Volume USDT", "tradecount"]).reset_index(drop=True)

    return data

def split(data: pd.DataFrame):
    """
    Split the data into training, testing, and validation sets.
    
    Parameters:
        data (pd.DataFrame): Cleaned market data.

    Returns:
        tuple: (train, test, validation) DataFrames.
    """
    
    # 60% train, 20% test, 20% validation
    train_size = int(len(data) * 0.6)
    test_size = int(len(data) * 0.2)

    # Split
    train = data[:train_size]
    test = data[train_size:train_size + test_size]
    validation = data[train_size + test_size:]

    return train, test, validation

def returns_table(portfolio, data):
    """
    Generate a returns table showing monthly, quarterly, and annual returns.
    
    Parameters:
        portfolio (list or pd.Series): Portfolio values over time.
        data (pd.DataFrame): Market data with datetime information.

    Returns:
        pd.DataFrame: DataFrame containing monthly, quarterly, and annual returns.
    """
    
    portfolio = pd.DataFrame({'Datetime': data['Datetime'].iloc[-len(portfolio):].reset_index(drop=True), 'Portfolio Value': portfolio})

    # Ensure Datetime is index and correct type
    portfolio = portfolio.copy()
    portfolio['Datetime'] = pd.to_datetime(portfolio['Datetime'])
    portfolio = portfolio.set_index('Datetime')

    # Calculate daily returns
    portfolio['Returns'] = portfolio['Portfolio Value'].pct_change()

    # Transform daily returns to monthly, quarterly, annually
    monthly = portfolio['Returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
    quarterly = portfolio['Returns'].resample('QE').apply(lambda x: (1 + x).prod() - 1)
    annually = portfolio['Returns'].resample('YE').apply(lambda x: (1 + x).prod() - 1)

    # Combine into a single DataFrame
    table = pd.DataFrame({
        "Monthly": monthly,
        "Quarterly": quarterly,
        "Annually": annually
    })

    # Fill NaN values with 0
    table = table.fillna(0)

    return table