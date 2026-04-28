"""日内震荡策略：跌了买、涨了卖。每只票可以使用各自训练好的参数。

每只票独立维护一组 (window, buy_z, sell_z, take_profit, stop_loss)。
- 没训练过 → 用全局默认值。
- 跑过 `osc-train` → 自动加载 models/osc_{TICKER}.json 的最优参数。
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from . import data as data_mod
from .config import LOG_DIR
from .osc_train import load_params
from .safety import SafetyConfig, SafetyState, check_buy
from .strategy import Portfolio


DEFAULT_PARAMS = {
    "window": 20,
    "buy_z": -1.0,
    "sell_z": 0.5,
    "take_profit": 0.005,
    "stop_loss": 0.01,
}


def zscore(closes: pd.Series, window: int) -> float:
    if len(closes) < window:
        return 0.0
    recent = closes.tail(window)
    std = recent.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float((closes.iloc[-1] - recent.mean()) / std)


def resolve_params(ticker: str, override: dict | None = None) -> dict:
    """合并优先级：调用方 override > 训练后保存的参数 > 全局默认。"""
    params = dict(DEFAULT_PARAMS)
    saved = load_params(ticker)
    if saved:
        params.update(saved)
    if override:
        params.update(override)
    return params


def run_oscillator(
    tickers: list[str],
    override: dict | None = None,
    poll_seconds: int = 30,
    starting_cash: float = 10_000.0,
    safety_cfg: SafetyConfig | None = None,
) -> None:
    pf = Portfolio(cash=starting_cash)
    log_file = LOG_DIR / f"osc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
    params_per_ticker = {t: resolve_params(t, override) for t in tickers}
    # 全局一个刹车状态：连续亏损在所有票之间共享冷静期。
    safety = SafetyState(cfg=safety_cfg or SafetyConfig())

    print(f"[osc] 启动 标的={tickers} 起始资金={starting_cash:,.0f} 刷新={poll_seconds}s")
    print(f"[osc] 日志 → {log_file}")
    for t, p in params_per_ticker.items():
        tag = "trained" if load_params(t) else "default"
        print(f"[osc] {t} ({tag}): {p}")
    print(
        f"[osc] 刹车 连亏冷静={safety.cfg.loss_streak}笔/{safety.cfg.cooldown_minutes}分钟 "
        f"不追>{safety.cfg.chase_pct:.0%} 不接刀(连跌3根) "
        f"波动>{safety.cfg.vol_outlier_mult}倍休市"
    )

    for snapshot in data_mod.stream_bars(tickers, "1m", poll_seconds):
        prices_now: dict[str, float] = {}
        events: list[dict] = []
        now = datetime.utcnow()

        for t, df in snapshot.items():
            p = params_per_ticker[t]
            if len(df) < p["window"]:
                continue
            price = float(df["close"].iloc[-1])
            prices_now[t] = price
            z = zscore(df["close"], p["window"])
            pos = pf._pos(t)

            action = "HOLD"
            if pos.qty > 0:
                pnl = pos.unrealized_pct(price)
                if pnl >= p["take_profit"] or pnl <= -p["stop_loss"] or z >= p["sell_z"]:
                    if pf.sell(t, price, df.index[-1]):
                        msg = safety.record_trade(pnl, now)
                        if msg:
                            print(f"[safety] {msg}")
                        action = f"SELL(pnl={pnl:+.2%})"
            elif z <= p["buy_z"]:
                ok, reason = check_buy(safety, df, pos, now)
                if not ok:
                    action = f"SKIP({reason})"
                else:
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

        moves = [e for e in events if e["action"] not in ("HOLD",)]
        if moves:
            tag = " ".join(f"{e['ticker']}:{e['action']}@{e['price']:.2f}" for e in moves)
            print(f"[{record['ts']}] equity={equity:,.2f} ★ {tag}")
        else:
            tag = " ".join(f"{e['ticker']}:z={e['z']:+.2f}" for e in events)
            print(f"[{record['ts']}] equity={equity:,.2f}  {tag}")
