"""把 OHLCV 转成「波动惯性」特征：多周期动量 + 波动率 + 量价。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """给定单标的 OHLCV，输出特征矩阵（含 close 列方便策略层使用）。"""
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    # 多周期收益（惯性核心）：短中长三档。
    for n in (1, 3, 5, 10, 20):
        out[f"ret_{n}"] = close.pct_change(n)

    # 滚动波动率（惯性强度的归一化分母）。
    out["vol_10"] = close.pct_change().rolling(10).std()
    out["vol_30"] = close.pct_change().rolling(30).std()

    # 移动均线斜率与距离（趋势）。
    sma_fast = close.rolling(10).mean()
    sma_slow = close.rolling(30).mean()
    out["sma_diff"] = (sma_fast - sma_slow) / close
    out["sma_slope"] = sma_fast.diff(3) / close

    # MACD 简化版。
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = (macd - signal) / close

    # RSI（动量超买/超卖）。
    out["rsi_14"] = _rsi(close, 14) / 100.0

    # ATR 归一化（波动幅度）。
    out["atr_14"] = _atr(df, 14) / close

    # 量能动量。
    vol = df["volume"].astype(float)
    out["vol_ratio"] = vol / vol.rolling(20).mean()
    out["vol_ret"] = vol.pct_change(5)

    # 高低位通道位置（0 到 1，越靠上越接近 20 期高点）。
    high20 = df["high"].rolling(20).max()
    low20 = df["low"].rolling(20).min()
    out["channel_pos"] = (close - low20) / (high20 - low20)

    # 把价格也带上，下游策略和回测要用。
    out["close"] = close

    return out.replace([np.inf, -np.inf], np.nan).dropna()


def make_label(features: pd.DataFrame, horizon: int) -> pd.Series:
    """未来 horizon 根 K 线收益为正 → 1，否则 0。"""
    fwd_ret = features["close"].shift(-horizon) / features["close"] - 1
    return (fwd_ret > 0).astype(int)


FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "vol_10", "vol_30",
    "sma_diff", "sma_slope",
    "macd", "rsi_14", "atr_14",
    "vol_ratio", "vol_ret",
    "channel_pos",
]
