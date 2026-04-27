"""日内震荡策略：跌了就买，涨了就卖。无需训练，持续 run。

核心思路：
1. 拉每只票的 1 分钟 K 线。
2. 算最近 N 分钟的均值与标准差。
3. 当前价比均值低 buy_z 个标准差 → 买入。
4. 持仓后，价格回到均值之上 sell_z 个标准差 / 达到止盈 / 触及止损 → 卖出。

参数都可以通过 CLI 调，默认偏激进，盘中可以做几十笔。
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from . import data as data_mod
from .config import LOG_DIR
from .strategy import Portfolio


def zscore(closes: pd.Series, window: int) -> float:
    if len(closes) < window:
        return 0.0
    recent = closes.tail(window)
    std = recent.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float((closes.iloc[-1] - recent.mean()) / std)


def run_oscillator(
    tickers: list[str],
    window: int = 20,
    buy_z: float = -1.0,
    sell_z: float = 0.5,
    take_profit: float = 0.005,
    stop_loss: float = 0.01,
    poll_seconds: int = 30,
    starting_cash: float = 10_000.0,
) -> None:
    pf = Portfolio(cash=starting_cash)
    log_file = LOG_DIR / f"osc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"[osc] 启动 标的={tickers} 起始资金={starting_cash:,.0f}")
    print(f"[osc] window={window}m buy_z={buy_z} sell_z={sell_z} tp={take_profit:.1%} sl={stop_loss:.1%} 刷新={poll_seconds}s")
    print(f"[osc] 日志 → {log_file}")

    for snapshot in data_mod.stream_bars(tickers, "1m", poll_seconds):
        prices_now: dict[str, float] = {}
        events: list[dict] = []

        for t, df in snapshot.items():
            if len(df) < window:
                continue
            price = float(df["close"].iloc[-1])
            prices_now[t] = price
            z = zscore(df["close"], window)
            pos = pf._pos(t)

            action = "HOLD"
            if pos.qty > 0:
                pnl = pos.unrealized_pct(price)
                if pnl >= take_profit or pnl <= -stop_loss or z >= sell_z:
                    pf.sell(t, price, df.index[-1])
                    action = f"SELL(pnl={pnl:+.2%})"
            elif z <= buy_z:
                equity = pf.equity(prices_now)
                if pf.buy(t, price, df.index[-1], equity):
                    action = "BUY"

            events.append({
                "ts": str(df.index[-1]),
                "ticker": t,
                "price": price,
                "z": round(z, 2),
                "action": action,
                "qty": pf._pos(t).qty,
            })

        equity = pf.equity(prices_now)
        record = {
            "ts": datetime.utcnow().isoformat(timespec="seconds"),
            "equity": equity,
            "cash": pf.cash,
            "events": events,
        }
        with log_file.open("a") as f:
            f.write(json.dumps(record) + "\n")

        moves = [e for e in events if e["action"] != "HOLD"]
        if moves:
            tag = " ".join(f"{e['ticker']}:{e['action']}@{e['price']:.2f}" for e in moves)
            print(f"[{record['ts']}] equity={equity:,.2f} ★ {tag}")
        else:
            tag = " ".join(f"{e['ticker']}:z={e['z']:+.2f}" for e in events)
            print(f"[{record['ts']}] equity={equity:,.2f}  {tag}")
