import pandas as pd

def split(data: pd.DataFrame):
    train_size = int(len(data) * 0.6)
    test_size = int(len(data) * 0.2)

    train = data[:train_size]
    test = data[train_size:train_size + test_size]
    validation = data[train_size + test_size:]

    return train, test, validation