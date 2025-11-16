import pandas as pd 

df = pd.read_csv("swings.csv")
df['datetime'] = pd.to_datetime(df['datetime']) 
df = df.set_index('datetime')

recent_hl = None
recent_hl_index = None
recent_lh = None
recent_lh_index = None
recent_ll = None
recent_ll_index = None
recent_hh = None
recent_hh_index = None

""" Takes in df with swings, converts them into legs """

for candle in df.itertuples():
    if candle.Index.hour < 3 or candle.Index.hour > 17:
        continue
    if candle.swing_type == 'HH':
        recent_hh = candle.high
        recent_hh_index = candle.Index
        if recent_hl is not None and recent_hl_index.day == candle.Index.day:
            if (candle.Index.hour) - (recent_hl_index.hour) < 3: # MIN 4 candle leg.
                continue 
            df.loc[candle.Index, 'leg_start_time'] = recent_hl_index
            df.loc[candle.Index, 'leg_start_price'] = recent_hl 
            df.loc[candle.Index, 'leg_end_time'] = candle.Index
            df.loc[candle.Index, 'leg_end_price'] = candle.high 
            df.loc[candle.Index, 'leg_bars'] = (pd.to_datetime(candle.Index) - pd.to_datetime(recent_hl_index)) / pd.Timedelta(minutes=30) 
            df.loc[candle.Index, 'leg_direction'] = 1    
            recent_hl = None
            recent_hl_index = None
        elif recent_hl is None:
            if recent_ll is not None and recent_ll_index.day == candle.Index.day:
                if (candle.Index.hour) - (recent_ll_index.hour) < 3: # MIN 4 candle leg.
                    continue 
                df.loc[candle.Index, 'leg_start_time'] = recent_ll_index
                df.loc[candle.Index, 'leg_start_price'] = recent_ll 
                df.loc[candle.Index, 'leg_end_time'] = candle.Index
                df.loc[candle.Index, 'leg_end_price'] = candle.high 
                df.loc[candle.Index, 'leg_bars'] = (pd.to_datetime(candle.Index) - pd.to_datetime(recent_ll_index)) / pd.Timedelta(minutes=30) 
                df.loc[candle.Index, 'leg_direction'] = 1    
                recent_ll = None 
                recent_ll_index = None 

    if candle.swing_type == 'LL':
        recent_ll = candle.low
        recent_ll_index = candle.Index
        if recent_lh is not None and recent_lh_index.day == candle.Index.day:
            if (candle.Index.hour) - (recent_lh_index.hour) < 3: # MIN 4 candle leg.
                continue 
            if df.loc[recent_lh_index:candle.Index]['low'].max() > recent_lh:
                continue
            df.loc[candle.Index, 'leg_start_time'] = recent_lh_index
            df.loc[candle.Index, 'leg_start_price'] = recent_lh 
            df.loc[candle.Index, 'leg_end_time'] = candle.Index
            df.loc[candle.Index, 'leg_end_price'] = candle.low 
            df.loc[candle.Index, 'leg_bars'] = (pd.to_datetime(candle.Index) - pd.to_datetime(recent_lh_index)) / pd.Timedelta(minutes=30) 
            df.loc[candle.Index, 'leg_direction'] = 0    
            recent_lh = None
            recent_lh_index = None

        elif recent_lh is None:
            if recent_hh is not None and recent_hh_index.day == candle.Index.day:
                if (candle.Index.hour) - (recent_hh_index.hour) < 3: # MIN 4 candle leg.
                    continue 
                df.loc[candle.Index, 'leg_start_time'] = recent_hh_index
                df.loc[candle.Index, 'leg_start_price'] = recent_hh 
                df.loc[candle.Index, 'leg_end_time'] = candle.Index
                df.loc[candle.Index, 'leg_end_price'] = candle.low 
                df.loc[candle.Index, 'leg_bars'] = (pd.to_datetime(candle.Index) - pd.to_datetime(recent_hh_index)) / pd.Timedelta(minutes=30) 
                df.loc[candle.Index, 'leg_direction'] = 0    
                recent_hh = None
                recent_hh_index = None

    if candle.swing_type == 'LH':
        recent_lh = candle.high
        recent_lh_index = candle.Index
        recent_hl = None
        recent_hl_index = None
        if recent_ll is not None and recent_ll_index.day == candle.Index.day:
            if (candle.Index.hour) - (recent_ll_index.hour) < 3: # MIN 4 candle leg.
                continue 
            df.loc[candle.Index, 'leg_start_time'] = recent_ll_index
            df.loc[candle.Index, 'leg_start_price'] = recent_ll 
            df.loc[candle.Index, 'leg_end_time'] = candle.Index
            df.loc[candle.Index, 'leg_end_price'] = candle.high
            df.loc[candle.Index, 'leg_bars'] = (pd.to_datetime(candle.Index) - pd.to_datetime(recent_ll_index)) / pd.Timedelta(minutes=30) 
            df.loc[candle.Index, 'leg_direction'] = 1
            recent_ll = None
            recent_ll_index = None
    
    if candle.swing_type == 'HL':
        recent_hl = candle.low
        recent_hl_index = candle.Index
        recent_lh = None
        recent_lh_index = None
        if recent_hh is not None and recent_hh_index.day == candle.Index.day:
            if (candle.Index.hour) - (recent_hh_index.hour) < 3: # MIN 4 candle leg.
                continue 
            if df.loc[recent_hh_index:candle.Index]['high'].max() > recent_hh:
                continue
            df.loc[candle.Index, 'leg_start_time'] = recent_hh_index
            df.loc[candle.Index, 'leg_start_price'] = recent_hh 
            df.loc[candle.Index, 'leg_end_time'] = candle.Index
            df.loc[candle.Index, 'leg_end_price'] = candle.low
            df.loc[candle.Index, 'leg_bars'] = (pd.to_datetime(candle.Index) - pd.to_datetime(recent_hh_index)) / pd.Timedelta(minutes=30) 
            df.loc[candle.Index, 'leg_direction'] = 0
            recent_hh = None
            recent_hh_index = None
    


print(f"Legs Created: {len(df[(~pd.isna(df['leg_direction']))])}")
df.to_csv("legs_test.csv")
