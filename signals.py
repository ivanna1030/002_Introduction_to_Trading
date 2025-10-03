import pandas as pd
import ta

def rsi_signals(data: pd.DataFrame, rsi_window: int, rsi_lower: int, rsi_upper: int):
    """
    Generate buy and sell signals based on RSI indicator.
    
    Parameters:
        data (pd.DataFrame): Market data with 'Close' prices.
        rsi_window (int): Window size for RSI calculation.
        rsi_lower (int): Lower threshold for buy signal.
        rsi_upper (int): Upper threshold for sell signal.

    Returns:
        tuple: (buy_signal, sell_signal) as pd.Series of boolean values.
    """
    
    rsi_indicator = ta.momentum.RSIIndicator(data.Close, window=rsi_window)
    rsi = rsi_indicator.rsi()

    # Fijo
    buy_signal = rsi < rsi_lower
    sell_signal = rsi > rsi_upper

    # Cruce
    #buy_signal = (rsi < rsi_lower) & (rsi.shift(1) >= rsi_lower)
    #sell_signal = (rsi > rsi_upper) & (rsi.shift(1) <= rsi_upper)

    return buy_signal, sell_signal

def ema_signals(data: pd.DataFrame, short_window: int, long_window: int):
    """
    Generate buy and sell signals based on EMA crossover.

    Parameters:
        data (pd.DataFrame): Market data with 'Close' prices.
        short_window (int): Window size for short-term EMA.
        long_window (int): Window size for long-term EMA.

    Returns:
        tuple: (buy_signal, sell_signal) as pd.Series of boolean values.
    """
    
    short_ema = ta.trend.EMAIndicator(data.Close, window=short_window).ema_indicator()
    long_ema = ta.trend.EMAIndicator(data.Close, window=long_window).ema_indicator()

    # Fijo
    #buy_signal = short_ema > long_ema
    #sell_signal = short_ema < long_ema

    # Cruce
    buy_signal = (short_ema > long_ema) & (short_ema.shift(1) <= long_ema.shift(1))
    sell_signal = (short_ema < long_ema) & (short_ema.shift(1) >= long_ema.shift(1))

    return buy_signal, sell_signal

def macd_signals(data: pd.DataFrame, short_window: int, long_window: int, signal_window: int):
    """
    Generate buy and sell signals based on MACD indicator.
    
    Parameters:
        data (pd.DataFrame): Market data with 'Close' prices.
        short_window (int): Window size for the fast EMA.
        long_window (int): Window size for the slow EMA.
        signal_window (int): Window size for the signal line EMA.

    Returns:
        tuple: (buy_signal, sell_signal) as pd.Series of boolean values.
    """
    
    macd = ta.trend.MACD(data.Close, window_slow=long_window, window_fast=short_window, window_sign=signal_window)
    macd_line = macd.macd()
    signal_line = macd.macd_signal()

    # Fijo
    buy_signal = macd_line > signal_line
    sell_signal = macd_line < signal_line

    # Cruce
    #buy_signal = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    #sell_signal = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    return buy_signal, sell_signal

def combined_signals(rsi_buy, rsi_sell, ema_buy, ema_sell, macd_buy, macd_sell):
    """
    Combine buy and sell signals from multiple indicators.
    
    Parameters:
        rsi_buy (pd.Series): Buy signals from RSI.
        rsi_sell (pd.Series): Sell signals from RSI.
        ema_buy (pd.Series): Buy signals from EMA.
        ema_sell (pd.Series): Sell signals from EMA.
        macd_buy (pd.Series): Buy signals from MACD.
        macd_sell (pd.Series): Sell signals from MACD.

    Returns:
        tuple: (buy_signal, sell_signal) as pd.Series of boolean values.
    """
    
    # Indicators agreement
    buy_signal = rsi_buy.astype(int) + ema_buy.astype(int) + macd_buy.astype(int) >= 2
    sell_signal = rsi_sell.astype(int) + ema_sell.astype(int) + macd_sell.astype(int) >= 2

    return buy_signal, sell_signal