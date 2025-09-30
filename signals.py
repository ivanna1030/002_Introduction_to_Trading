import pandas as pd
import ta

def rsi_signals(data: pd.DataFrame, rsi_window: int, rsi_lower: int, rsi_upper: int):
    rsi_indicator = ta.momentum.RSIIndicator(data.Close, window=rsi_window)

    rsi = rsi_indicator.rsi()
    buy_signal = rsi < rsi_lower
    sell_signal = rsi > rsi_upper
    
    return buy_signal, sell_signal

def ema_signals(data: pd.DataFrame, short_window: int, long_window: int):
    short_ema = ta.trend.EMAIndicator(data.Close, window=short_window).ema_indicator()
    long_ema = ta.trend.EMAIndicator(data.Close, window=long_window).ema_indicator()

    buy_signal = short_ema > long_ema
    sell_signal = short_ema < long_ema

    return buy_signal, sell_signal

def macd_signals(data: pd.DataFrame, short_window: int, long_window: int, signal_window: int):
    macd = ta.trend.MACD(data.Close, window_slow=long_window, window_fast=short_window, window_sign=signal_window)
    macd_line = macd.macd()
    signal_line = macd.macd_signal()

    buy_signal = macd_line > signal_line
    sell_signal = macd_line < signal_line

    return buy_signal, sell_signal

def combined_signals(rsi_buy, rsi_sell, ema_buy, ema_sell, macd_buy, macd_sell):
    buy_signal = rsi_buy.astype(int) + ema_buy.astype(int) + macd_buy.astype(int) >= 2
    sell_signal = rsi_sell.astype(int) + ema_sell.astype(int) + macd_sell.astype(int) >= 2

    return buy_signal, sell_signal