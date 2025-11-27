from backtesting import Strategy, Backtest
from indicator import find_levels
import pandas as pd
import numpy as np 

class Levels(Strategy):
   
    # Strategy Paramters
    lookback=4
    lookforward=2
    collapse_threshold=0.25
    base_decay=0.0
    crowd_coeff=0.25
    price_window=0.5
    min_strength=0.25

    def init(self):
        return
    
    def next(self):

        levels = find_levels(self.data.df.reset_index(), 
                             self.lookback, 
                             self.lookforward, 
                             self.collapse_threshold, 
                             self.base_decay, 
                             self.crowd_coeff, 
                             self.price_window, 
                             self.min_strength)

        # First, check open orders. 




        # Then, start searching for new signals. 
        if len(self.data.df) < self.lookback + self.lookforward + 2:
            return
        if len(levels) == 0:
            return
        
        previous_candle = self.data.df.iloc[-2]
        current_candle = self.data.df.iloc[-1]

        for level in levels.itertuples():
            level_price = level.level
            # Check for buy signal
            if (previous_candle.Low > level_price) and (current_candle.Low <= level_price):
                self.buy()


            # Check for sell signal
            elif (previous_candle.High < level_price) and (current_candle.High >= level_price):
                self.sell()





data = pd.read_csv('data/in_sample/CL_30m_in_sample_1.csv').set_index('datetime')
data.index = pd.to_datetime(data.index)
data = data.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume', 'datetime':'DateTime'})

bt = Backtest(data, Levels, cash=10000, commission=0.0,spread = 0.022)
stats = bt.run()
print(stats)