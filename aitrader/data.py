"""美股行情拉取，全部走 yfinance（无需 API key）。"""

from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import yfinance as yf


def fetch_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """拉历史 K 线，返回带有 Open/High/Low/Close/Volume 的 DataFrame。"""
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance 返回空数据: {ticker} {period} {interval}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    return df[["open", "high", "low", "close", "volume"]].dropna()


def fetch_many(tickers: Iterable[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            out[t] = fetch_history(t, period, interval)
        except Exception as exc:  # noqa: BLE001
            print(f"[data] 跳过 {t}: {exc}")
    return out


def latest_quote(ticker: str) -> tuple[float, pd.Timestamp]:
    """取最近一次成交价（实时刷新）。"""
    df = yf.download(
        ticker,
        period="1d",
        interval="1m",
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df.empty:
        # 回退到日线收盘。
        df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    last = df.iloc[-1]
    return float(last["Close"]), df.index[-1]


def stream_bars(tickers: Iterable[str], interval: str, poll_seconds: int):
    """无限循环产生最新一根 K 线。yfinance 没有 push，只能轮询。"""
    while True:
        snapshot: dict[str, pd.DataFrame] = {}
        for t in tickers:
            try:
                snapshot[t] = fetch_history(t, period="5d", interval=interval).tail(200)
            except Exception as exc:  # noqa: BLE001
                print(f"[stream] {t} 拉取失败: {exc}")
        yield snapshot
        time.sleep(poll_seconds)
