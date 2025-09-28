import pandas as pd
import ta

def rsi_signals(data: pd.DataFrame, rsi_window: int, rsi_lower: int, rsi_upper: int):
    rsi_indicator = ta.momentum.RSIIndicator(data.Close, window=rsi_window)

    rsi = rsi_indicator.rsi()
    buy_signal = rsi < rsi_lower
    sell_signal = rsi > rsi_upper
    
    return buy_signal, sell_signal