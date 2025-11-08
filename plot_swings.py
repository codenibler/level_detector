import pandas as pd 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots

df = pd.read_csv('swings.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

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


fig.update_layout(
    dragmode='zoom',
    selectdirection='any', 
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

fig.show()