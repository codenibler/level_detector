import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

""" lookback = n1 """ 
""" lookforward = n2 """ 
""" base decay = starting decay constant"""
""" crowd_coeff = how much nearness to other levels accelerates decay """
""" price_window = price window to consider another level 'nearby' """
""" min_strength = minimum strength to keep level """

""" If using as a trading indicator, consider that every time this is called, it should be done with a spliced 
    dataframe that only includes data up to the current time. Otherwise, future data will leak in and affect level detection. 
"""

def find_levels(data, lookback, lookforward, collapse_threshold, base_decay, crowd_coeff, price_window, min_strength):
    
    if data.empty: # No days available
        return pd.DataFrame(columns=['row', 'level', 'strength', 'time', 'level_type'])
        
    data = data.reset_index(drop=True)
    window_end = data['datetime'].max()
    window_start = window_end - pd.Timedelta(days=30)
    data = data[(data['datetime'] >= window_start) & (data['datetime'] <= window_end)].reset_index(drop=True)

    """ Support and Resistance Functions """
    def support(df1, l, peek_back, peek_forward):         
        for i in range(l-peek_back+1, l+1):
            if(df1.low[i]>df1.low[i-1]):
                return 0
        for i in range(l+1,l+peek_forward+1):
            if(df1.low[i]<df1.low[i-1]):
                return 0
        return 1
    def resistance(df1, l, peek_back, peek_forward): 
        for i in range(l-peek_back+1, l+1):
            if(df1.high[i]<df1.high[i-1]):
                return 0
        for i in range(l+1,l+peek_forward+1):
            if(df1.high[i]>df1.high[i-1]):
                return 0
        return 1
    

    """ Find all levels """
    supports = []
    resistances = []
    for row in range(lookback, data.index.max() - lookforward): 
        if support(data, row, lookback, lookforward):
            supports.append((row,data.low[row], data['datetime'][row], 1))
        if resistance(data, row, lookback, lookforward):
            resistances.append((row,data.high[row], data['datetime'][row],2))
    
    """ Collapse nearby levels """ 
    supports.sort()
    resistances.sort()
    pre_collapse_num  = len(supports) + len(resistances)    

    for i in range(1,len(supports)):
        if(i>=len(supports)):
            break
        if abs(supports[i][1]-supports[i-1][1]) <= collapse_threshold:
            # Keep more conservative level.
            if supports[i][1] < supports[i-1][1]:
                supports.pop(i-1)
            else:
                supports.pop(i)
    for i in range(1,len(resistances)):
        if(i>=len(resistances)):
            break
        if abs(resistances[i][1]-resistances[i-1][1]) <= collapse_threshold:
            if resistances[i][1] > resistances[i-1][1]:
                resistances.pop(i-1)
            else:
                resistances.pop(i)

    post_collapse_num = len(supports) + len(resistances)

    """ Time Decay Strength Calculation """
    month_seconds = 2592000 # Approximate number of seconds in a month
    def time_decay_levels(levels):
        out = []
        for row, level, creation_time, level_type in levels:
            level_time = creation_time
            age_sec = max((data['datetime'].max() - level_time).total_seconds(), 0)
            age_norm = age_sec / month_seconds
            near_count = sum(1 for v in all_levels if abs(v - level) <= price_window) 
            effective_decay = base_decay + crowd_coeff * max(0, near_count - 1)
            strength = float(np.exp(-effective_decay * age_norm))
            out.append((row, level, strength, level_time))
        return out

    all_levels = [level for _, level, _, _ in supports + resistances]
    supports = time_decay_levels(supports)
    resistances = time_decay_levels(resistances)

    """ Prune Weak Levels """
    supports = [lvl for lvl in supports if lvl[2] >= min_strength]
    resistances = [lvl for lvl in resistances if lvl[2] >= min_strength]

    """ Prepare DataFrames for Output """
    supports = pd.DataFrame(supports, columns=['row', 'level', 'strength', 'time'])
    supports['level_type'] = 'support'

    resistances = pd.DataFrame(resistances, columns=['row', 'level', 'strength', 'time'])
    resistances['level_type'] = 'resistance'

    """ Avoid concatenating empty dataframes """
    """ Return combined levels DataFrame """
    if resistances.empty:
        if supports.empty:
            return pd.DataFrame(columns=['row', 'level', 'strength', 'time', 'level_type'])
        else:
            supports.set_index('row', inplace=True)
            supports = supports.sort_values(by='level', ascending=False)
            return supports
    else:
        if supports.empty:
            resistances.set_index('row', inplace=True)
            resistances = resistances.sort_values(by='level', ascending=False)
            return resistances
        else:    
            levels = pd.concat([supports, resistances], ignore_index=True)
            levels.set_index('row', inplace=True)
            levels = levels.sort_values(by='level', ascending=False)
            return levels



if __name__ == "__main__":

    """ DF Cleanup """
    data1 = pd.read_csv("data/monthly_segments/CL_30m_2024-11.csv")
    data2 = pd.read_csv("data/monthly_segments/CL_30m_2024-12.csv")
    data = pd.concat([data1, data2], ignore_index=True)
    data['datetime'] = pd.to_datetime(data['datetime'])

    """ Run Level Detection """
    levels_df = find_levels(data=data, lookback=4, lookforward=2, collapse_threshold=0.0,
                             base_decay=0.0, crowd_coeff=0.0, price_window=0.5, min_strength=0.25)

    """ Output Levels """
    print(levels_df.head(10))
    fig = go.Figure(data=[go.Candlestick(
        x=data['datetime'],
        open=data['open'], high=data['high'], low=data['low'], close=data['close'],
        increasing_line_color='green', decreasing_line_color='red'
    )])

    for _, level, strength, creation_time, level_type in levels_df.itertuples():
        fig.add_shape(
            type='line', xref='x', yref='y',
            x0=creation_time, x1=data['datetime'].max(), y0=level, y1=level,
            line=dict(color='rgb(220,20,60)' if level_type == 'resistance' else 'rgb(30,144,255)', width=2),
            opacity=strength,
        )

    fig.update_layout(
        showlegend=False,
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        paper_bgcolor='black',
        plot_bgcolor='black',
        title=dict(text=f'Levels, Time and Crowding Decayed. 30 minutes. {data["datetime"].min().date()} to {data["datetime"].max().date()}',
        font=dict(color='white', size=25, family='Cascadia Black'))
    )

    fig.show()



    """
    For you, codex.
    This code snippet defines a function called find_levels that takes in several parameters including data, lookback, lookforward, 
    collapse_threshold, base_decay, crowd_coeff, price_window, and min_strength.

    The function first defines two helper functions: support and resistance. These functions check if a given row in the data DataFrame has a 
    low value that is lower than the previous low value (for support) or a high value that is higher than the previous high value (for resistance). If the condition is met, the function returns 1, otherwise it returns 0.

    The find_levels function then finds all the levels (support and resistance) in the data DataFrame by iterating over the rows and calling the 
    support and resistance functions. The levels are stored in supports and resistances lists.

    Next, the function collapses nearby levels by sorting the supports and resistances lists and removing levels that are within a certain distance of another level (based on the collapse_threshold). The number of levels before and after the collapse is printed.

    The function then calculates the time decay strength of each level by calling the time_decay_levels
    function. This function takes in a list of levels and calculates the age of each level, the number of nearby levels, 
    and the effective decay based on the base_decay and crowd_coeff parameters. The strength of each level is calculated using an 
    exponential decay formula and stored in the strength column of the level.

    Weak levels (with strength below min_strength) are then pruned from the supports and resistances lists.

    Finally, the function prepares the supports and resistances data for output by converting them into DataFrames and concatenating them into a single DataFrame called 
    levels. The levels DataFrame is indexed by the row column and returned by the function.
    """
