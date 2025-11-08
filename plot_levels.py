import pandas as pd 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
from neuro_trader_sp import support_resistance_levels

df = pd.read_csv('swings.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
levels = support_resistance_levels(df, 365, first_w=1.0, atr_mult=3.0)
sig_levels = set()
for level_list in levels:
    if level_list is not None:
        for level in level_list:
            sig_levels.add(round(level, 2))

fig = go.Figure(
    go.Candlestick(
        x = df['datetime'],
        high=df['high'],
        low=df['low'],
        open=df['open'],
        close=df['close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )
)
""" HL """
fig.add_trace(
    go.Scatter(
        x = df['datetime'][df['swing_type'] == 'HL'],
        y = df['low'][df['swing_type'] == 'HL'] - 0.05,
        mode='markers+text',
        marker=dict(color='green', symbol='triangle-up',size=12),
        text='HL',
        textposition='bottom center',
        name='HL'
    )
)
""" LL """
fig.add_trace(
    go.Scatter(
        x = df['datetime'][df['swing_type'] == 'LL'],
        y = df['low'][df['swing_type'] == 'LL'] - 0.05,
        mode='markers+text',
        marker=dict(color='green', symbol='triangle-up',size=15),
        text='LL',
        textposition='bottom center',
        name='LL'
    )
)
""" LH """
fig.add_trace(
    go.Scatter(
        x = df['datetime'][df['swing_type'] == 'LH'],
        y = df['high'][df['swing_type'] == 'LH'] + 0.05,
        mode='markers+text',
        marker=dict(color='red', symbol='triangle-down',size=12),
        text='LH',
        textposition='top center',
        name='LH'
    )
)
""" HH """
fig.add_trace(
    go.Scatter(
        x = df['datetime'][df['swing_type'] == 'HH'],
        y = df['high'][df['swing_type'] == 'HH'] + 0.05,
        mode='markers+text',
        marker=dict(color='red', symbol='triangle-down',size=15),
        text='HH',
        textposition='top center',
        name='HH'
    )
)

shapes = [
        dict(type='line', xref='paper', x0=0, x1=1, y0=y, y1=y,
             line=dict(color='royalblue', width=1, dash='dot'))
        for y in sig_levels
    ]

fig.update_layout(
    dragmode='zoom',
    selectdirection='any', 
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    shapes=shapes
)

fig.show()