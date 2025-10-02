import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from models import Operation
from signals import rsi_signals, ema_signals, macd_signals, combined_signals
from metrics import calmar_ratio

def get_portfolio_value(cash: float, long_ops: list[Operation], short_ops: list[Operation], current_price:float, COM: float) -> float:
    val = cash

    # Add long positions value
    for pos in long_ops:
        pnl = current_price * pos.n_shares * (1 - COM)
        val += pnl

    # Add short positions value
    for pos in short_ops:
        pnl = (pos.price - current_price) * pos.n_shares * (1 - COM)
        initial_sell = pos.price * pos.n_shares * (1 + COM)

        val += pnl + initial_sell

    return val

def backtest(data, trial) -> float:
    # RSI
    rsi_window = trial.suggest_int('rsi_window', 5, 50)
    rsi_lower = trial.suggest_int('rsi_lower', 5, 35)
    rsi_upper = trial.suggest_int('rsi_upper', 65, 95)

    # EMA
    ema_short_window = trial.suggest_int('ema_short_window', 5, 50)
    ema_long_window = trial.suggest_int('ema_long_window', 100, 300)

    # MACD
    macd_short_window = trial.suggest_int('macd_short_window', 5, 50)
    macd_long_window = trial.suggest_int('macd_long_window', 100, 300)
    macd_signal_window = trial.suggest_int('macd_signal_window', 5, 50)

    # Trade params
    stop_loss = trial.suggest_float('stop_loss', 0.01, 0.15)
    take_profit = trial.suggest_float('take_profit', 0.01, 0.15)
    available_cash_pct = trial.suggest_float('available_cash_pct', 0.01, 0.1)

    # Signals
    rsi_buy_signals, rsi_sell_signals = rsi_signals(data, rsi_window, rsi_lower, rsi_upper)
    ema_buy_signals, ema_sell_signals = ema_signals(data, ema_short_window, ema_long_window)
    macd_buy_signals, macd_sell_signals = macd_signals(data, macd_short_window, macd_long_window, macd_signal_window)

    buy_signals, sell_signals = combined_signals(rsi_buy_signals, rsi_sell_signals, ema_buy_signals, ema_sell_signals, macd_buy_signals, macd_sell_signals)
    
    historic = data.copy()
    historic = historic.dropna()
    historic['buy_signal'] = buy_signals
    historic['sell_signal'] = sell_signals

    # Params
    COM = 0.125 / 100
    SL = stop_loss
    TP = take_profit

    # Backtest logic
    active_long_positions: list[Operation] = []
    active_short_positions: list[Operation] = []

    cash = 1_000_000

    portfolio_value = []

    for i, row in historic.iterrows():
        # Close long positions
        for position in active_long_positions.copy():
            # Check take profit or stop loss
            if row.Close > position.take_profit or row.Close < position.stop_loss:
                cash += row.Close * position.n_shares * (1 - COM)
                active_long_positions.remove(position)

        # Close short positions
        for position in active_short_positions.copy():
            # Check take profit or stop loss
            if row.Close < position.take_profit or row.Close > position.stop_loss:
                pnl = (position.price - row.Close) * position.n_shares * (1 - COM)
                initial_sell = position.price * position.n_shares * (1 + COM)
                cash += pnl + initial_sell
                active_short_positions.remove(position)
                continue

        # --- BUY ---
        # Check signal
        if row.buy_signal:
            n_shares = cash * available_cash_pct / row.Close
            # Do we have enough cash?
            if cash > row.Close * n_shares * (1 + COM):
                # Discount the cost
                cash -= row.Close * n_shares * (1 + COM)
                # Save the operation as active position
                active_long_positions.append(
                    Operation(
                    time=row.Datetime,
                    price=row.Close,
                    take_profit=row.Close * (1 + TP),
                    stop_loss=row.Close * (1 - SL),
                    n_shares=n_shares,
                    type="LONG"
                    )
                )

        # --- SELL ---
        # Check signal
        if row.sell_signal:
            n_shares = cash * available_cash_pct / row.Close
            position_value = row.Close * n_shares * (1 + COM)
            # Do we have enough cash?
            if cash > position_value:
                cash -= position_value
                active_short_positions.append(
                    Operation(
                    time=row.Datetime,
                    price=row.Close,
                    take_profit=row.Close * (1 - TP),
                    stop_loss=row.Close * (1 + SL),
                    n_shares=n_shares,
                    type="SHORT"
                    )
                )
        
        # Add current portfolio value to the list
        portfolio_value.append(get_portfolio_value(cash, active_long_positions, active_short_positions, row.Close, COM))

    # Close long positions
    for position in active_long_positions:
        pnl = (row.Close - position.price) * position.n_shares * (1 - COM)
        cash += row.Close * position.n_shares * (1 - COM)

    # Close short positions
    for position in active_short_positions:
        pnl = (position.price - row.Close) * position.n_shares * (1 - COM)
        initial_sell = position.price * position.n_shares * (1 + COM)
        cash += pnl + initial_sell

    active_long_positions = []
    active_short_positions = []

    # Calculate and return Calmar ratio
    df = pd.DataFrame({'Portfolio Value': portfolio_value})
    calmar = calmar_ratio(df['Portfolio Value'])

    return calmar

def params_backtest(data, params, cash):
    # RSI
    rsi_window = params['rsi_window']
    rsi_lower = params['rsi_lower']
    rsi_upper = params['rsi_upper']

    # EMA
    ema_short_window = params['ema_short_window']
    ema_long_window = params['ema_long_window']

    # MACD
    macd_short_window = params['macd_short_window']
    macd_long_window = params['macd_long_window']
    macd_signal_window = params['macd_signal_window']

    # Trade params
    SL = params['stop_loss']
    TP = params['take_profit']
    available_cash_pct = params['available_cash_pct']
    COM = 0.125 / 100

    # Signals
    rsi_buy_signals, rsi_sell_signals = rsi_signals(data, rsi_window, rsi_lower, rsi_upper)
    ema_buy_signals, ema_sell_signals = ema_signals(data, ema_short_window, ema_long_window)
    macd_buy_signals, macd_sell_signals = macd_signals(data, macd_short_window, macd_long_window, macd_signal_window)

    buy_signals, sell_signals = combined_signals(rsi_buy_signals, rsi_sell_signals, ema_buy_signals, ema_sell_signals, macd_buy_signals, macd_sell_signals)
    
    historic = data.copy()
    historic = historic.dropna()
    historic['buy_signal'] = buy_signals
    historic['sell_signal'] = sell_signals

    # Backtest logic
    active_long_positions: list[Operation] = []
    active_short_positions: list[Operation] = []

    portfolio_value = []

    positive_trades = 0
    negative_trades = 0

    for i, row in historic.iterrows():
        # Close long positions
        for position in active_long_positions.copy():
            # Check take profit or stop loss
            if row.Close > position.take_profit or row.Close < position.stop_loss:
                pnl = (row.Close - position.price) * position.n_shares * (1 - COM)
                cash += row.Close * position.n_shares * (1 - COM)
                # Add to win/loss count
                if pnl >= 0:
                    positive_trades += 1
                else:
                    negative_trades += 1
                active_long_positions.remove(position)

        # Close short positions
        for position in active_short_positions.copy():
            # Check take profit or stop loss
            if row.Close < position.take_profit or row.Close > position.stop_loss:
                pnl = (position.price - row.Close) * position.n_shares * (1 - COM)
                initial_sell = position.price * position.n_shares * (1 + COM)
                cash += pnl + initial_sell
                # Add to win/loss count
                if pnl >= 0:
                    positive_trades += 1
                else:
                    negative_trades += 1
                active_short_positions.remove(position)
                continue

        # --- BUY ---
        # Check signal
        if row.buy_signal:
            n_shares = cash * available_cash_pct / row.Close
            # Do we have enough cash?
            if cash > row.Close * n_shares * (1 + COM):
                # Discount the cost
                cash -= row.Close * n_shares * (1 + COM)
                # Save the operation as active position
                active_long_positions.append(
                    Operation(
                    time=row.Datetime,
                    price=row.Close,
                    take_profit=row.Close * (1 + TP),
                    stop_loss=row.Close * (1 - SL),
                    n_shares=n_shares,
                    type="LONG"
                    )
                )

        # --- SELL ---
        # Check signal
        if row.sell_signal:
            n_shares = cash * available_cash_pct / row.Close
            position_value = row.Close * n_shares * (1 + COM)
            # Do we have enough cash?
            if cash > position_value:
                cash -= position_value
                active_short_positions.append(
                    Operation(
                    time=row.Datetime,
                    price=row.Close,
                    take_profit=row.Close * (1 - TP),
                    stop_loss=row.Close * (1 + SL),
                    n_shares=n_shares,
                    type="SHORT"
                    )
                )
        
        # Add current portfolio value to the list
        portfolio_value.append(get_portfolio_value(cash, active_long_positions, active_short_positions, row.Close, COM))

    # Close long positions        
    for position in active_long_positions:
        pnl = (row.Close - position.price) * position.n_shares * (1 - COM)
        cash += row.Close * position.n_shares * (1 - COM)
        # Add to win/loss count
        if pnl >= 0:
            positive_trades += 1
        else:
            negative_trades += 1

    # Close short positions
    for position in active_short_positions:
        pnl = (position.price - row.Close) * position.n_shares * (1 - COM)
        initial_sell = position.price * position.n_shares * (1 + COM)
        cash += pnl + initial_sell
        # Add to win/loss count
        if pnl >= 0:
            positive_trades += 1
        else:
            negative_trades += 1

    active_long_positions = []
    active_short_positions = []

    # Calculate win rate
    win_rate = positive_trades / (positive_trades + negative_trades) if (positive_trades + negative_trades) > 0 else 0

    return cash, portfolio_value, win_rate

def walk_forward(data, trial, n_splits=5):
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    # Iterate over each split
    for train_index, test_index in tscv.split(data):
        train = data.iloc[train_index]
        test = data.iloc[test_index]

        # Run backtest on the test set
        result = backtest(test, trial)
        results.append(result)

    return np.mean(results)