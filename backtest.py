from indicator import find_levels
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
import numpy as np 
import sys

MONTH = sys.argv[1]

@dataclass
class PendingTrade:
    id: int
    level_type: str  # 'support' or 'resistance'
    level: float
    timestamp: pd.Timestamp
    entry: float 
    sl: float
    tp: float
    size: int

@dataclass
class OpenTrade:
    level_type: str  # 'support' or 'resistance'
    entry_level: float
    tp: float
    sl: float
    opened: pd.Timestamp 
    size: int
    pending_id: int
    order_created: pd.Timestamp

@dataclass
class ClosedTrade:
    level_type: str  # 'support' or 'resistance'
    entry_level: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    tp: float
    sl: float 
    outcome: str
    pnl: float
    pending_id: int
    order_created: pd.Timestamp

# ============== HELPER FUNCTIONS ==============

def find_next_resistance(current_level, levels_df):
    higher_levels = levels_df[(levels_df['level_type'] == 'resistance') & (levels_df['level'] > current_level)]
    if not higher_levels.empty:
        index_min = higher_levels['level'].idxmin()
        return higher_levels.loc[index_min, 'level'], higher_levels.loc[index_min, 'strength']
    return None, None

def find_next_support(current_level, levels_df):
    lower_levels = levels_df[(levels_df['level_type'] == 'support') & (levels_df['level'] < current_level)]
    if not lower_levels.empty:
        index_max = lower_levels['level'].idxmax()
        return lower_levels.loc[index_max, 'level'], lower_levels.loc[index_max, 'strength']
    return None, None


# ============== DATA LOADING AND PARAMETERS ==============

""" Load 1 minute Data """
data = pd.read_csv(f'data/in_sample_monthly/CL_1m_in_sample_1_{MONTH}.csv').set_index('datetime')
data.index = pd.to_datetime(data.index)

# Strategy Paramters
lookback=4
lookforward=2
collapse_threshold=0.25
base_decay=0.0
crowd_coeff=0.25
price_window=0.5
min_strength=0.25

pending_trades = []
open_trades = []
closed_trades = []
daily_used_levels = []
levels = []
starting_pending_id = 0

SPREAD = 0.022
STARTING_CASH = 10000
RISK_PER_TRADE = 0.01

cash = STARTING_CASH
    
for candle in tqdm(data.itertuples(), desc="Crunching Backtest, Dawg", unit="candle", total=len(data)):

    open, high, low, volume, close = candle.open, candle.high, candle.low, candle.volume, candle.close
    timestamp = pd.to_datetime(candle.Index)

    # If at EOD, Create Levels 
    if timestamp.hour == 00 and timestamp.minute == 00:
        daily_used_levels = []
        data_30m = pd.read_csv(f"data/monthly_segments/CL_30m_{MONTH}.csv")
        data_30m['datetime'] = pd.to_datetime(data_30m['datetime'])
        data_30m = data_30m[data_30m['datetime'] < timestamp]
        levels = find_levels(data_30m.reset_index(), 
                            lookback, 
                            lookforward, 
                            collapse_threshold, 
                            base_decay, 
                            crowd_coeff, 
                            price_window, 
                            min_strength)
    
    if len(levels) == 0: # If no levels, return
        continue

    # Remove liquidated trades and execute pending trades
    for pending_trade in pending_trades.copy():
        if pending_trade.level_type == 'resistance':
            # Remove SHORTS that lost liquidity
            if high >= pending_trade.entry:
                # Enter SHORT pending_trade
                if pending_trade.entry <= pending_trade.tp or pending_trade.entry >= pending_trade.sl:
                    pending_trades.remove(pending_trade)
                    continue
                ## Turn pending into open trade. 
                open_trades.append(OpenTrade(
                    level_type='resistance',
                    entry_level=pending_trade.entry,
                    tp=pending_trade.tp,
                    sl=pending_trade.sl,
                    opened=timestamp,
                    size = pending_trade.size,
                    pending_id= pending_trade.id,
                    order_created= pending_trade.timestamp
                ))
                pending_trades.remove(pending_trade)

        elif pending_trade.level_type == 'support':
            # Remove LONGS that lost liquidity
            if low <= pending_trade.entry:
                # Enter LONG pending_trade
                if pending_trade.entry >= pending_trade.tp or pending_trade.entry <= pending_trade.sl:
                    pending_trades.remove(pending_trade)
                    continue
                open_trades.append(OpenTrade(
                    level_type='support',
                    entry_level=pending_trade.entry,
                    tp=pending_trade.tp,
                    sl=pending_trade.sl,
                    opened=timestamp,
                    size = pending_trade.size,
                    pending_id= pending_trade.id,
                    order_created= pending_trade.timestamp
                ))
                pending_trades.remove(pending_trade)


    # Add Pending Trade if level has been violated
    for level in levels.itertuples():
        if level.level_type == 'resistance': # Looking for short. 
            if high > level.level and low <= level.level:
                entry_resistance, entry_resistance_strength = find_next_resistance(level.level, levels)
                if entry_resistance is None:
                    continue
                take_profit_distance = abs(round((entry_resistance - level.level) / 2, 2))
                if take_profit_distance > 1.0:
                    continue # Take profit way too big
                if ((timestamp.date(), level.level) in daily_used_levels):
                    continue
                daily_used_levels.append((timestamp.date(), level.level)) # Don't trade this level again today. 
                take_profit = entry_resistance - take_profit_distance
                stop_loss = entry_resistance + take_profit_distance
                trade_size = round(((cash * RISK_PER_TRADE) / ((stop_loss + SPREAD) - (entry_resistance - SPREAD))) * entry_resistance_strength * take_profit_distance)
                pending_trade = PendingTrade(
                    id=starting_pending_id,
                    level_type='resistance',
                    level=level.level,
                    timestamp=timestamp,
                    entry=entry_resistance,
                    tp= take_profit,
                    sl= stop_loss,
                    size= trade_size
                )
                starting_pending_id += 1
                for trade in pending_trades.copy():
                    if trade.level_type == pending_trade.level_type and trade.entry == pending_trade.entry: 
                        pending_trades.remove(trade)
                pending_trades.append(pending_trade)

        elif level.level_type == 'support': # Looking for long.
            if low < level.level and high >= level.level:
                entry_support, entry_support_strength = find_next_support(level.level, levels)
                if entry_support is None:
                    continue
                take_profit_distance = abs(round((entry_support - level.level) / 2, 2))
                if take_profit_distance > 1.0:
                    continue
                if ((timestamp.date(), level.level) in daily_used_levels):
                    continue
                daily_used_levels.append((timestamp.date(), level.level)) # Don't trade this level again today. 
                take_profit = entry_support + take_profit_distance
                stop_loss = entry_support - take_profit_distance
                trade_size = round(((cash * RISK_PER_TRADE) / ((entry_support + SPREAD) - (stop_loss - SPREAD))) * entry_support_strength * take_profit_distance)
                pending_trade = PendingTrade(
                    id=starting_pending_id,
                    level_type='support',
                    level=level.level,
                    timestamp=timestamp, 
                    entry=entry_support,
                    tp= take_profit,
                    sl= stop_loss,
                    size= trade_size
                )
                starting_pending_id += 1
                for trade in pending_trades.copy():
                    if trade.level_type == pending_trade.level_type and trade.entry == pending_trade.entry: 
                        pending_trades.remove(trade)
                pending_trades.append(pending_trade)


    # Manage Open Trades
    for trade in open_trades.copy():
        ### CLOSING SHORT TRADES
        if trade.level_type == 'resistance':
            if low <= trade.tp:
                pnl = ((trade.entry_level - SPREAD) - (trade.tp + SPREAD)) * trade.size
                closed_trades.append(ClosedTrade(
                    level_type='resistance',
                    entry_level=trade.entry_level,
                    entry_time=trade.opened,
                    exit_time=timestamp,
                    tp=trade.tp,
                    sl=trade.sl,
                    outcome='tp',
                    pnl= pnl,
                    pending_id = trade.pending_id,
                    order_created= trade.order_created
                ))
                cash += pnl
                open_trades.remove(trade)
            elif high >= trade.sl:
                pnl = -((trade.sl + SPREAD) - (trade.entry_level - SPREAD)) * trade.size
                closed_trades.append(ClosedTrade(
                    level_type='resistance',
                    entry_level=trade.entry_level,
                    entry_time=trade.opened,
                    exit_time=timestamp,
                    tp=trade.tp,
                    sl=trade.sl,
                    outcome='sl',
                    pnl= pnl,
                    pending_id= trade.pending_id,
                    order_created= trade.order_created
                ))
                cash += pnl
                open_trades.remove(trade)
        ### CLOSING LONG TRADES
        elif trade.level_type == 'support':
            if high >= trade.tp:
                pnl = ((trade.tp - SPREAD) - (trade.entry_level + SPREAD)) * trade.size
                closed_trades.append(ClosedTrade(
                    level_type='support',
                    entry_level=trade.entry_level,
                    entry_time=trade.opened,
                    exit_time=timestamp,
                    tp=trade.tp,
                    sl=trade.sl,
                    outcome='tp',
                    pnl= pnl,
                    pending_id= trade.pending_id,
                    order_created= trade.order_created
                ))
                cash += pnl
                open_trades.remove(trade)
            elif low <= trade.sl:
                pnl = -((trade.entry_level + SPREAD)  - (trade.sl - SPREAD)) * trade.size
                closed_trades.append(ClosedTrade(
                    level_type='support',
                    entry_level=trade.entry_level,
                    entry_time=trade.opened,
                    exit_time=timestamp,
                    tp=trade.tp,
                    sl=trade.sl,
                    outcome='sl',
                    pnl=pnl,
                    pending_id= trade.pending_id,
                    order_created= trade.order_created
                ))
                cash += pnl
                open_trades.remove(trade)


closed_trades = pd.DataFrame([trade.__dict__ for trade in closed_trades])
closed_trades.to_csv(f'trades_{MONTH}.csv', index=False)
print(f"Saved trades to trades_{MONTH}.csv")

print(closed_trades['outcome'].value_counts())
print(f"PnL = {closed_trades['pnl'].sum()}")
