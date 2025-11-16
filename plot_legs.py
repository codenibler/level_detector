import pandas as pd
import plotly.graph_objects as go


def _parse_datetime_columns(df: pd.DataFrame, columns):
    """Safely parse datetime columns that may contain NaNs."""
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


# def _plot_swings(fig: go.Figure, data: pd.DataFrame) -> None:
#     # """Add swing annotations (HH/HL/LH/LL) to the figure."""
#     # swing_markers = [
#     #     ("HH", "high", dict(symbol="triangle-down", size=12, color="firebrick", line=dict(width=1, color="white")), "top center"),
#     #     ("LH", "high", dict(symbol="triangle-down", size=12, color="#e377c2", line=dict(width=1, color="white")), "top center"),
#     #     ("LL", "low", dict(symbol="triangle-up", size=12, color="green", line=dict(width=1, color="white")), "bottom center"),
#     #     ("HL", "low", dict(symbol="triangle-up", size=12, color="seagreen", line=dict(width=1, color="white")), "bottom center"),
#     # ]

#     for label, price_col, marker, text_position in swing_markers:
#         mask = data["swing_type"] == label
#         if mask.any():
#             fig.add_trace(
#                 go.Scatter(
#                     x=data.loc[mask, "datetime"],
#                     y=data.loc[mask, price_col],
#                     mode="markers+text",
#                     marker=marker,
#                     text=[label] * int(mask.sum()),
#                     textposition=text_position,
#                     name=label,
#                 )
#             )


def _plot_legs(fig: go.Figure, data: pd.DataFrame) -> None:
    """Overlay price legs, with colour by direction."""
    legs = data.dropna(subset=["leg_start_time", "leg_end_time", "leg_start_price", "leg_end_price"])
    if legs.empty:
        return

    def _leg_color(row):
        return "#2ca02c" if row.leg_end_price >= row.leg_start_price else "#d62728"

    for row in legs.itertuples():
        color = _leg_color(row)
        fig.add_trace(
            go.Scatter(
                x=[row.leg_start_time, row.leg_end_time],
                y=[row.leg_start_price, row.leg_end_price],
                mode="lines+markers+text",
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                text=[f"{'LH' if row.swing_type == 'LL' else 'HL'}", f"{row.swing_type}"],
                textposition="top center",
                name="Leg",
                showlegend=False,
            )
        )


def main():
    data = pd.read_csv("legs_test.csv")
    data = _parse_datetime_columns(data, ["datetime", "leg_start_time", "leg_end_time"])

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data["datetime"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            increasing_line_color="mediumseagreen",
            decreasing_line_color="indianred",
            name="Price",
        )
    )



    # _plot_swings(fig, data)
    _plot_legs(fig, data)

    fig.update_layout(
        title="Pivot Points with Legs",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.show()


if __name__ == "__main__":
    main()
