"""纸面交易（实时）：每隔 poll_seconds 拉新数据 → 模型推断 → 执行决策。"""

from __future__ import annotations

import json
from datetime import datetime

from . import data as data_mod
from . import features as feat
from . import model as ai_model
from .config import CONFIG, LOG_DIR
from .strategy import Action, Portfolio, decide


def _load_models(tickers):
    pipes = {}
    for t in tickers:
        try:
            pipes[t] = ai_model.load(t)
        except FileNotFoundError:
            print(f"[trader] {t} 没有训练好的模型，跳过。先跑 train。")
    return pipes


def run_paper_trading(tickers=None) -> None:
    tickers = tickers or CONFIG.tickers
    pipes = _load_models(tickers)
    if not pipes:
        raise SystemExit("没有可用模型，先执行：python -m aitrader.main train")

    pf = Portfolio()
    log_file = LOG_DIR / f"paper_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"[trader] 纸面交易启动，标的={list(pipes)}，日志={log_file}")

    for snapshot in data_mod.stream_bars(pipes.keys(), CONFIG.live_interval, CONFIG.poll_seconds):
        prices_now: dict[str, float] = {}
        events: list[dict] = []
        for t, df in snapshot.items():
            if t not in pipes or len(df) < 60:
                continue
            feats = feat.build_features(df)
            if feats.empty:
                continue
            price = float(feats["close"].iloc[-1])
            prices_now[t] = price
            proba = ai_model.predict_proba(pipes[t], feats)
            action = decide(proba, pf._pos(t), price)
            equity = pf.equity(prices_now)
            if action is Action.BUY:
                pf.buy(t, price, feats.index[-1], equity)
            elif action is Action.SELL:
                pf.sell(t, price, feats.index[-1])
            events.append({
                "ts": str(feats.index[-1]),
                "ticker": t,
                "price": price,
                "proba_up": proba,
                "action": action.value,
                "qty": pf._pos(t).qty,
            })

        equity = pf.equity(prices_now)
        record = {
            "ts": datetime.utcnow().isoformat(),
            "equity": equity,
            "cash": pf.cash,
            "events": events,
        }
        with log_file.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{record['ts']}] equity={equity:,.2f} cash={pf.cash:,.2f} events={len(events)}")
