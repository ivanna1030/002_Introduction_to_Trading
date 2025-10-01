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