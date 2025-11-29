from plotly.subplots import make_subplots
from indicator import find_levels
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import sys

MONTH = sys.argv[1]

def plot_closed_trades(
    price_path: str = f"data/in_sample_monthly/CL_1m_in_sample_1_{MONTH}.csv",
    trades_path: str = f"trades_{MONTH}.csv",
    output_path: str = f"trades_&_equity_{MONTH}.html",
) -> None:
    """Plot price action with closed trades overlaid."""
    price_path = Path(price_path)
    trades_path = Path(trades_path)

    price_df = pd.read_csv(price_path)
    price_df["datetime"] = pd.to_datetime(price_df["datetime"])

    trades_df = pd.read_csv(trades_path)
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

    equity_df = trades_df.sort_values("exit_time", ascending=True)
    equity_df["equity"] = equity_df["pnl"].cumsum()

    equity_timeline = price_df[["datetime"]].merge(
      equity_df[["exit_time", "equity"]],
      left_on="datetime",
      right_on="exit_time",
      how="left",
    )
    equity_timeline["equity"] = equity_timeline["equity"].ffill()
    print(equity_timeline)

    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.2, 0.8], 
        subplot_titles=("Price", "Trades")
    )

    fig.add_trace(
        go.Scatter(
            x=equity_timeline["datetime"],
            y=equity_timeline['equity'],
            line=dict(color="blue", width=2),
            name="Equity",
        ),
        row=1,
        col=1,
    )

    fig.add_trace( 
            go.Candlestick(
                x=price_df["datetime"],
                open=price_df["open"],
                high=price_df["high"],
                low=price_df["low"],
                close=price_df["close"],
                increasing_line_color="green",
                decreasing_line_color="red",
                name="Price",
            ),
    row=2, 
    col=1
    )


    type_color = {"support": "blue", "resistance": "red"}
    tp_color = "forestgreen"
    sl_color = "red"

    shapes = []
    entry_markers_x = []
    entry_markers_y = []
    entry_markers_text = []
    exit_markers_x = []
    exit_markers_y = []
    exit_markers_text = []

    for trade in trades_df.itertuples():

        entry_color = type_color[trade.level_type]
        x0 = trade.entry_time
        x1 = trade.exit_time
        exit_price = trade.tp if trade.outcome == "tp" else trade.sl if trade.outcome == "sl" else trade.entry_level


        # Take profit band (ensure y0<y1 so Plotly renders the rect)
        y0_tp, y1_tp = sorted([trade.entry_level, trade.tp])
        shapes.append(
            dict(
                type="rect",
                xref="x2",
                yref="y2",
                x0=trade.entry_time,
                x1=trade.exit_time,
                y0=y0_tp,
                y1=y1_tp,
                line=dict(color=tp_color, width=1),
                fillcolor=tp_color,
                opacity=0.25,
            )
        )

        # Stop loss band (ensure y0<y1 so Plotly renders the rect)
        y0_sl, y1_sl = sorted([trade.entry_level, trade.sl])
        shapes.append(
            dict(
                type="rect",
                xref="x2",
                yref="y2",
                x0=trade.entry_time,
                x1=trade.exit_time,
                y0=y0_sl,
                y1=y1_sl,
                line=dict(color=sl_color, width=1),
                fillcolor=sl_color,
                opacity=0.25,
            )
        )

        entry_markers_x.append(x0)
        entry_markers_y.append(trade.entry_level)
        entry_markers_text.append(f"{trade.level_type} entry")

        exit_markers_x.append(x1)
        exit_markers_y.append(exit_price)
        exit_markers_text.append(f"exit ({trade.outcome})")


    price_df_30m = pd.read_csv(f'data/monthly_segments/CL_30m_{MONTH}.csv')
    price_df_30m['datetime'] = pd.to_datetime(price_df_30m['datetime'])
    first_date = pd.Timestamp(f"{MONTH}-01")
    epsilon = 1e-4

    for date in range(0, 30):
        current_date = first_date + pd.Timedelta(days=date)
        chunk = price_df_30m.loc[price_df_30m['datetime'] <= current_date]
        if chunk.empty:
            continue
        levels = find_levels(data=chunk, lookback=4, lookforward=2, collapse_threshold=0.25,
                                base_decay=0.0, crowd_coeff=0.25, price_window=0.5, min_strength=0.25)
        
        if levels.empty:
            continue
        
        # Remove levels that fully decayed
        for shape in shapes:
            if shape['line']['color'] in ["blue", "crimson"]:
                for price in levels['level'].values:
                    if abs(price - shape['y0']) < epsilon:
                        shape['x1'] = current_date

        for level in levels.itertuples():
            shapes.append(
                dict(
                    type="line",
                    xref="x2",
                    yref="y2",
                    x0=level.time,
                    x1=price_df['datetime'].max(),
                    y0=level.level,
                    y1=level.level,
                    line=dict(color="blue" if level.level_type == "support" else "crimson", width=2),
                    opacity=level.strength
                )
            )

        print(f"Appended levels for {first_date + pd.Timedelta(days=date)}")


    fig.update_layout(
        title="Price with Closed Trades",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        shapes=shapes,
        showlegend=False,
        dragmode='zoom'
    )
    fig.update_xaxes(rangeslider_visible=False)

    fig.add_trace(
        go.Scatter(
            x=entry_markers_x,
            y=entry_markers_y,
            mode="markers",
            marker=dict(color="black", size=8, symbol="triangle-up"),
            name="Entry",
            hovertext=entry_markers_text,
            hoverinfo="text+x+y",
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=exit_markers_x,
            y=exit_markers_y,
            mode="markers",
            marker=dict(color="orange", size=8, symbol="x"),
            name="Exit",
            hovertext=exit_markers_text,
            hoverinfo="text+x+y",
        ),
        row=2, col=1
    )

    fig.write_html(output_path)


if __name__ == "__main__":
    plot_closed_trades()
