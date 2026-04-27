"""逐 bar 回测，沿用 strategy 模块的同一套买卖逻辑。"""

from __future__ import annotations

import pandas as pd

from . import features as feat
from . import model as ai_model
from .config import CONFIG
from .strategy import Action, Portfolio, decide


def backtest(prices: dict[str, pd.DataFrame], horizon: int = CONFIG.horizon) -> dict:
    """每只票训练一个模型，再用 walk-forward 的概率走一遍买卖。"""
    pf = Portfolio()
    per_ticker_features: dict[str, pd.DataFrame] = {}
    per_ticker_model = {}

    for t, df in prices.items():
        try:
            pipe, metrics = ai_model.train_one(df, horizon=horizon)
        except Exception as exc:  # noqa: BLE001
            print(f"[backtest] {t} 训练失败: {exc}")
            continue
        per_ticker_model[t] = pipe
        per_ticker_features[t] = feat.build_features(df)
        print(f"[backtest] {t} 模型 acc={metrics['accuracy']:.3f} auc={metrics['auc']:.3f}")

    # 使用所有标的的特征索引并集并按时间对齐。
    all_idx = sorted(set().union(*(f.index for f in per_ticker_features.values())))
    # 跳过最前面热身段。
    warm = max(int(len(all_idx) * 0.2), 50)

    for ts in all_idx[warm:]:
        prices_now: dict[str, float] = {}
        for t, feats in per_ticker_features.items():
            if ts not in feats.index:
                continue
            row = feats.loc[:ts].iloc[[-1]]
            price = float(row["close"].iloc[-1])
            prices_now[t] = price
            proba = ai_model.predict_proba(per_ticker_model[t], feats.loc[:ts])
            action = decide(proba, pf._pos(t), price)
            equity = pf.equity(prices_now | {t: price})
            if action is Action.BUY:
                pf.buy(t, price, ts, equity)
            elif action is Action.SELL:
                pf.sell(t, price, ts)

    final_prices = {t: float(f["close"].iloc[-1]) for t, f in per_ticker_features.items()}
    final_equity = pf.equity(final_prices)
    return {
        "starting_cash": CONFIG.starting_cash,
        "final_equity": final_equity,
        "return_pct": final_equity / CONFIG.starting_cash - 1.0,
        "trades": len(pf.trade_log),
        "trade_log": pf.trade_log,
        "positions": {t: vars(p) for t, p in pf.positions.items()},
    }
