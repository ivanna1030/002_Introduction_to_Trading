import pandas as pd

def modify_data(data: pd.DataFrame):
    data = data.copy()
    data = data.rename(columns={'Date': 'Datetime'})
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce', dayfirst=True)
    data = data.iloc[::-1].reset_index(drop=True)

    return data

def split(data: pd.DataFrame):
    train_size = int(len(data) * 0.6)
    test_size = int(len(data) * 0.2)

    train = data[:train_size]
    test = data[train_size:train_size + test_size]
    validation = data[train_size + test_size:]

    return train, test, validation

def returns_table(portfolio, test_data, validation_data):
    full_data = pd.concat([test_data, validation_data]).reset_index(drop=True)
    portfolio = pd.DataFrame({'Datetime': full_data['Datetime'], 'Portfolio Value': portfolio})

    # Ensure Datetime is index
    portfolio = portfolio.copy()
    portfolio['Datetime'] = pd.to_datetime(portfolio['Datetime'])
    portfolio = portfolio.set_index('Datetime')

    # Compute daily returns
    portfolio['Returns'] = portfolio['Portfolio Value'].pct_change()

    # Aggregate
    monthly = portfolio['Returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
    quarterly = portfolio['Returns'].resample('QE').apply(lambda x: (1 + x).prod() - 1)
    annually = portfolio['Returns'].resample('YE').apply(lambda x: (1 + x).prod() - 1)

    # Combine into a single DataFrame
    table = pd.DataFrame({
        "Monthly": monthly,
        "Quarterly": quarterly,
        "Annually": annually
    })

    table = table.fillna(0)

    return table