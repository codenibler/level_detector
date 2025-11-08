import pandas as pd 
import numpy as np
import pandas_ta as ta

# - Find Leg Patterns: 
#   () LL -> HL -> LH -> HH -> Up Leg
#   () HH -> HL -> LH -> LL -> Down Leg

def generate_data(dataset):
    # =============== Indicators ================
    df = pd.read_csv(dataset).set_index('datetime')
    df.index = pd.to_datetime(df.index)
    
    # =============== Swing Logic ===============
    df['PrevHigh'] = df['high'].shift(1)
    df['PrevPrevHigh'] = df['high'].shift(2)
    df['PrevLow'] = df['low'].shift(1)
    df['PrevPrevLow'] = df['low'].shift(2)
    swing_lows = (
        (df['PrevLow'] <= df['low']) &
        (df['PrevLow'] <= df['PrevPrevLow'])
    )
    swing_highs= (
        (df['PrevHigh'] >= df['high']) &
        (df['PrevHigh'] >= df['PrevPrevHigh'])
    )
    df['swing_high'] = swing_highs.shift(-1) # CONSIDER SHIFT FOR VISUALIZATION
    df['swing_low'] = swing_lows.shift(-1) # CONSIDER SHIFT FOR VISUALIZATION

    df = df.drop(columns=['PrevHigh', 'PrevLow', 'PrevPrevLow', 'PrevPrevHigh'])
    df = df.dropna()
    
    # =============== Cleaning ===============
    return df

if __name__ == "__main__":
    
    df = generate_data('data/monthly_segments/CL_30m_2024-08.csv')
    df['swing_type'] = pd.NA

    previous_high = None
    previous_high_idx = None
    previous_low = None
    previous_low_idx = None

    for candle in df.itertuples():
        if candle.swing_low and previous_low is not None:
            if candle.low < previous_low: 
                df.loc[candle.Index, 'swing_type'] = 'LL'
                previous_low = candle.low
            else:
                df.loc[candle.Index, 'swing_type'] = 'HL'
                previous_low = candle.low

        if candle.swing_high and previous_high is not None:
            if candle.high > previous_high: 
                df.loc[candle.Index, 'swing_type'] = 'HH'
                previous_high = candle.high
            else:
                df.loc[candle.Index, 'swing_type'] = 'LH'
                previous_high = candle.high

        if candle.swing_low and previous_low is None:
            df.loc[candle.Index, 'swing_type'] = 'HL'
            previous_low = candle.low
        if candle.swing_high and previous_high is None:
            df.loc[candle.Index, 'swing_type'] = 'LH'
            previous_high = candle.high
            
    df.to_csv('swings.csv')

