import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


# Global containers for the strategy
LEVEL_CACHE: Dict[pd.Timestamp, Dict[str, List[float]]] = {}
TRADING_START: Optional[pd.Timestamp] = None
RIGHT_PEEK = 10  # Pivot lookahead


def pivotid(df1: pd.DataFrame, index: int, left_peek: int, right_peek: int) -> int:
    if index - left_peek < 0 or index + right_peek >= len(df1):
        return 0

    pivot_low = True
    pivot_high = True
    current_low = df1.iloc[index]["Low"]
    current_high = df1.iloc[index]["High"]

    for i in range(index - left_peek, index + right_peek + 1):
        if current_low > df1.iloc[i]["Low"]:
            pivot_low = False
        if current_high < df1.iloc[i]["High"]:
            pivot_high = False

    if pivot_low and pivot_high:
        return 3
    if pivot_low:
        return 1
    if pivot_high:
        return 2
    return 0


def ema_indicator(series: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(series).ewm(span=period, adjust=False).mean().to_numpy()


def compute_pivots(df: pd.DataFrame, left_peek: int, right_peek: int) -> pd.Series:
    pivots = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        pivots[i] = pivotid(df, i, left_peek, right_peek)
    return pd.Series(pivots, index=df.index, name="pivot")


def merge_close_levels(level_df: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    if level_df.empty:
        return level_df
    merged_rows = []
    sorted_levels = level_df.sort_values("level").reset_index(drop=True)
    idx = 0
    while idx < len(sorted_levels):
        current = sorted_levels.iloc[idx].copy()
        if idx + 1 < len(sorted_levels):
            nxt = sorted_levels.iloc[idx + 1]
            if abs(current["level"] - nxt["level"]) <= bin_width:
                if current["total_hits"] >= nxt["total_hits"]:
                    merged = current
                    merged["total_hits"] += nxt["total_hits"]
                    merged["support_hits"] += nxt["support_hits"]
                    merged["resistance_hits"] += nxt["resistance_hits"]
                else:
                    merged = nxt.copy()
                    merged["total_hits"] += current["total_hits"]
                    merged["support_hits"] += current["support_hits"]
                    merged["resistance_hits"] += current["resistance_hits"]
                merged["type"] = (
                    "support" if merged["support_hits"] >= merged["resistance_hits"] else "resistance"
                )
                merged["created"] = min(current["created"], nxt["created"])
                merged_rows.append(merged)
                idx += 2
                continue
        merged_rows.append(current)
        idx += 1
    return pd.DataFrame(merged_rows).sort_values("total_hits", ascending=False).reset_index(drop=True)


def compute_levels_for_index(
    df: pd.DataFrame,
    idx: int,
    bin_width: float = 0.1,
    min_total_touches: int = 4,
    recent_window_days: int = 60,
    min_recent_touches: int = 3,
) -> pd.DataFrame:
    cutoff = idx - RIGHT_PEEK
    if cutoff <= 0:
        return pd.DataFrame(columns=["level", "type"])

    subset = df.iloc[: cutoff + 1].copy()
    pivot_subset = subset["pivot"]
    subset["level_support"] = (
        (subset["Low"] / bin_width).round() * bin_width
    )
    subset["level_resistance"] = (
        (subset["High"] / bin_width).round() * bin_width
    )

    supports = subset[pivot_subset == 1][["Datetime", "level_support"]].rename(
        columns={"level_support": "level"}
    )
    resistances = subset[pivot_subset == 2][["Datetime", "level_resistance"]].rename(
        columns={"level_resistance": "level"}
    )

    support_counts = (
        supports.groupby("level")
        .agg(support_hits=("level", "count"), created_support=("Datetime", "min"))
        .reset_index()
    )
    resistance_counts = (
        resistances.groupby("level")
        .agg(resistance_hits=("level", "count"), created_resistance=("Datetime", "min"))
        .reset_index()
    )

    levels = pd.merge(support_counts, resistance_counts, how="outer", on="level").fillna(0)
    if levels.empty:
        return levels

    levels["support_hits"] = levels["support_hits"].astype(int)
    levels["resistance_hits"] = levels["resistance_hits"].astype(int)
    levels["created_support"] = pd.to_datetime(levels["created_support"], errors="coerce")
    levels["created_resistance"] = pd.to_datetime(levels["created_resistance"], errors="coerce")
    levels["created"] = levels[["created_support", "created_resistance"]].min(axis=1)
    levels.drop(columns=["created_support", "created_resistance"], inplace=True)
    levels["total_hits"] = levels["support_hits"] + levels["resistance_hits"]
    levels = levels[levels["total_hits"] >= min_total_touches]
    if levels.empty:
        return levels

    levels["type"] = np.where(levels["support_hits"] >= levels["resistance_hits"], "support", "resistance")
    levels = merge_close_levels(levels, bin_width)
    if levels.empty:
        return levels

    current_time = subset["Datetime"].iloc[-1]
    recent_start = current_time - pd.Timedelta(days=recent_window_days)
    recent_slice = subset[subset["Datetime"] >= recent_start]

    level_tolerance = bin_width / 2
    kept_rows = []
    for _, row in levels.iterrows():
        level_price = row["level"]
        support_recent = recent_slice[
            (recent_slice["pivot"] == 1)
            & (recent_slice["Low"].sub(level_price).abs() <= level_tolerance)
        ]
        resistance_recent = recent_slice[
            (recent_slice["pivot"] == 2)
            & (recent_slice["High"].sub(level_price).abs() <= level_tolerance)
        ]
        if len(support_recent) + len(resistance_recent) >= min_recent_touches:
            kept_rows.append(row)

    return pd.DataFrame(kept_rows).reset_index(drop=True)


def prepare_level_cache(
    df: pd.DataFrame,
    start_time: pd.Timestamp,
    bin_width: float = 0.1,
    min_total_touches: int = 4,
    recent_days: int = 60,
    min_recent_touches: int = 3,
) -> Dict[pd.Timestamp, Dict[str, List[float]]]:
    cache: Dict[pd.Timestamp, Dict[str, List[float]]] = {}
    for idx in tqdm(range(len(df)), desc="Building level cache"):
        current_time = df["Datetime"].iloc[idx]
        if current_time < start_time:
            continue
        level_frame = compute_levels_for_index(
            df,
            idx,
            bin_width=bin_width,
            min_total_touches=min_total_touches,
            recent_window_days=recent_days,
            min_recent_touches=min_recent_touches,
        )
        if level_frame.empty:
            continue
        cache[current_time] = {
            "supports": sorted(level_frame[level_frame["type"] == "support"]["level"].tolist()),
            "resistances": sorted(level_frame[level_frame["type"] == "resistance"]["level"].tolist()),
        }
    return cache


class LevelBounceStrategy(Strategy):
    params = dict(bin_width=0.1, risk_per_trade=0.005)

    def init(self):
        self.ema_fast = None
        self.ema_slow = None

    def _position_size(self, entry: float, stop: float) -> float:
        risk = abs(entry - stop)
        if risk == 0:
            return 0.0
        capital_risk = self.equity * self.params["risk_per_trade"]
        size = capital_risk / risk
        if size <= 0:
            return 0.0
        if size >= 1:
            return float(int(size))
        return size

    def _current_levels(self) -> Optional[Dict[str, List[float]]]:
        current_time = self.data.index[-1]
        return LEVEL_CACHE.get(current_time)

    def next(self):
        current_time = self.data.index[-1]
        if TRADING_START and current_time < TRADING_START:
            return
        # Allow multiple simultaneous trades; skip only if no levels or no hit.
        levels = self._current_levels()
        if not levels:
            return
        close_price = self.data.Close[-1]
        high_price = self.data.High[-1]
        low_price = self.data.Low[-1]
        tolerance = self.params["bin_width"] / 2

        supports = levels.get("supports", [])
        resistances = levels.get("resistances", [])

        self._try_long(close_price, high_price, low_price, supports, resistances, tolerance)
        self._try_short(close_price, high_price, low_price, supports, resistances, tolerance)

    def _touch(self, price_low: float, price_high: float, level: float, tolerance: float) -> bool:
        return (price_low - tolerance) <= level <= (price_high + tolerance)

    def _try_long(
        self,
        close_price: float,
        high_price: float,
        low_price: float,
        supports: List[float],
        resistances: List[float],
        tolerance: float,
    ):
        if len(supports) < 2 or not resistances:
            return
        for idx in reversed(range(len(supports))):
            level = supports[idx]
            if not self._touch(low_price, high_price, level, tolerance):
                continue
            lower_support = supports[idx - 1] if idx > 0 else None
            upper_resistance = next((r for r in resistances if r > level), None)
            if lower_support is None or upper_resistance is None:
                continue
            entry = level
            sl = lower_support
            tp = upper_resistance
            size = self._position_size(entry, sl)
            size = float(size)
            if size <= 0:
                continue
            if not (sl < entry < tp):
                continue
            self.buy(size=size, sl=sl, tp=tp)
            break

    def _try_short(
        self,
        close_price: float,
        high_price: float,
        low_price: float,
        supports: List[float],
        resistances: List[float],
        tolerance: float,
    ):
        if len(resistances) < 2 or not supports:
            return
        for idx in range(len(resistances)):
            level = resistances[idx]
            if not self._touch(low_price, high_price, level, tolerance):
                continue
            support_below = max((s for s in supports if s < level), default=None)
            resistance_above = resistances[idx + 1] if idx + 1 < len(resistances) else None
            if support_below is None or resistance_above is None:
                continue
            entry = level
            sl = resistance_above
            tp = support_below
            size = self._position_size(entry, sl)
            size = float(size)
            if size <= 0:
                continue
            if not (tp < entry < sl):
                continue
            self.sell(size=size, sl=sl, tp=tp)
            break


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.rename(columns=str.title)
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df


def main():
    global LEVEL_CACHE, TRADING_START
    data_path = "data/CL_30m.csv"
    df = load_data(data_path)
    df["pivot"] = compute_pivots(df, left_peek=10, right_peek=RIGHT_PEEK)
    first_datetime = df["Datetime"].min()
    TRADING_START = first_datetime + pd.Timedelta(days=90)
    LEVEL_CACHE = prepare_level_cache(
        df,
        start_time=TRADING_START,
        bin_width=0.1,
        min_total_touches=4,
        recent_days=60,
        min_recent_touches=3,
    )

    price_df = df.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]]
    bt = Backtest(
        price_df,
        LevelBounceStrategy,
        cash=10000,
        commission=0.0,
        trade_on_close=True,
    )
    stats = bt.run()
    print(stats)
    bt.plot(filename="sr_backtest.html")


if __name__ == "__main__":
    main()
