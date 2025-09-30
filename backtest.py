import pandas as pd
import ta

from models import Operation
from signals import rsi_signals, ema_signals, macd_signals, combined_signals
from metrics import max_drawdown, calmar_ratio

def get_portfolio_value(cash: float, long_ops: list[Operation], short_ops: list[Operation], current_price:float, n_shares: int, COM: float) -> float:
    val = cash

    # Add long positions value
    val += len(long_ops) * current_price * n_shares * (1 - COM)

    # Add short positions value
    for pos in short_ops:
        pnl = (pos.price - current_price) * pos.n_shares * (1 - COM)

        val += pnl

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
    n_shares = trial.suggest_float('n_shares', 0, 5)

    # Signals
    rsi_buy_signals, rsi_sell_signals = rsi_signals(data, rsi_window, rsi_lower, rsi_upper)
    ema_buy_signals, ema_sell_signals = ema_signals(data, ema_short_window, ema_long_window)
    macd_buy_signals, macd_sell_signals = macd_signals(data, macd_short_window, macd_long_window, macd_signal_window)

    buy_signals, sell_signals = combined_signals(rsi_buy_signals, rsi_sell_signals, ema_buy_signals, ema_sell_signals, macd_buy_signals, macd_sell_signals)
    
    historic = data.copy()
    historic = historic.dropna()
    historic['buy_signal'] = buy_signals
    historic['sell_signal'] = sell_signals

    COM = 0.125 / 100
    SL = stop_loss
    TP = take_profit
    n_shares = n_shares

    active_long_positions: list[Operation] = []
    active_short_positions: list[Operation] = []

    cash = 1_000_000

    portfolio_value = []

    for i, row in historic.iterrows():
        # Close long positions
        for position in active_long_positions.copy():
            if row.Close > position.take_profit or row.Close < position.stop_loss:
                cash += row.Close * position.n_shares * (1 - COM)
                active_long_positions.remove(position)

        # Close short positions
        for position in active_short_positions.copy():
            if row.Close < position.take_profit or row.Close > position.stop_loss:
                cover_cost = row.Close * position.n_shares * (1 + COM)
                initial_sell = position.price * position.n_shares
                pnl = initial_sell - cover_cost
                cash += pnl
                active_short_positions.remove(position)
                continue

        # --- BUY ---
        # Check signal
        if row.buy_signal:
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
            # Do we have enough cash?
            position_value = row.Close * n_shares * (1 + COM)
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
                
        portfolio_value.append(get_portfolio_value(cash, active_long_positions, active_short_positions, row.Close, n_shares, COM))

    # Close long positions        
    cash += row.Close * len(active_long_positions) * n_shares * (1 - COM)

    # Close short positions
    for position in active_short_positions:
        cover_cost = row.Close * position.n_shares * (1 + COM)
        initial_sell = position.price * position.n_shares
        pnl = initial_sell - cover_cost
        cash += pnl
    
    active_long_positions = []
    active_short_positions = []

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
    n_shares = params['n_shares']
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

    active_long_positions: list[Operation] = []
    active_short_positions: list[Operation] = []

    portfolio_value = []

    for i, row in historic.iterrows():
        # Close long positions
        for position in active_long_positions.copy():
            if row.Close > position.take_profit or row.Close < position.stop_loss:
                cash += row.Close * position.n_shares * (1 - COM)
                active_long_positions.remove(position)

        # Close short positions
        for position in active_short_positions.copy():
            if row.Close < position.take_profit or row.Close > position.stop_loss:
                cover_cost = row.Close * position.n_shares * (1 + COM)
                initial_sell = position.price * position.n_shares
                pnl = initial_sell - cover_cost
                cash += pnl
                active_short_positions.remove(position)
                continue

        # --- BUY ---
        # Check signal
        if row.buy_signal:
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
            # Do we have enough cash?
            position_value = row.Close * n_shares * (1 + COM)
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
                
        portfolio_value.append(get_portfolio_value(cash, active_long_positions, active_short_positions, row.Close, n_shares, COM))

    # Close long positions        
    cash += row.Close * len(active_long_positions) * n_shares * (1 - COM)

    # Close short positions
    for position in active_short_positions:
        cover_cost = row.Close * position.n_shares * (1 + COM)
        initial_sell = position.price * position.n_shares
        pnl = initial_sell - cover_cost
        cash += pnl

    active_long_positions = []
    active_short_positions = []

    return cash, portfolio_value