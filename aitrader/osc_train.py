"""为震荡策略做参数搜索：在历史 1m 数据上回测，每只票挑出最优 (window, buy_z, sell_z, tp, sl)。"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as data_mod
from .config import MODEL_DIR


DEFAULT_GRID: dict[str, list] = {
    "window":      [15, 20, 30, 45],
    "buy_z":       [-1.5, -1.2, -1.0, -0.8],
    "sell_z":      [0.2, 0.4, 0.6, 0.9],
    "take_profit": [0.002, 0.004, 0.006, 0.01],
    "stop_loss":   [0.006, 0.01, 0.015, 0.02],
}


def simulate(
    closes: np.ndarray,
    window: int,
    buy_z: float,
    sell_z: float,
    take_profit: float,
    stop_loss: float,
    starting_cash: float = 10_000.0,
) -> dict:
    """单标的纯函数回测，与 run_oscillator 的买卖规则一致。"""
    n = len(closes)
    if n < window + 5:
        return {"return": -1.0, "trades": 0, "win_rate": 0.0, "max_dd": 0.0}

    cash = starting_cash
    qty = 0.0
    avg_cost = 0.0
    trades = 0
    wins = 0
    equity_curve = []
    peak = starting_cash
    max_dd = 0.0

    # 预计算滚动均值/标准差。
    series = pd.Series(closes)
    means = series.rolling(window).mean().values
    stds = series.rolling(window).std().values

    for i in range(window, n):
        std = stds[i - 1]
        if std == 0 or np.isnan(std):
            equity_curve.append(cash + qty * closes[i])
            continue
        price = closes[i]
        z = (price - means[i - 1]) / std

        if qty > 0:
            pnl = price / avg_cost - 1.0
            if pnl >= take_profit or pnl <= -stop_loss or z >= sell_z:
                cash += qty * price
                if pnl > 0:
                    wins += 1
                qty = 0.0
                avg_cost = 0.0
                trades += 1
        elif z <= buy_z:
            spend = cash * 0.95
            if spend > 1.0:
                qty = spend / price
                avg_cost = price
                cash -= qty * price

        equity = cash + qty * price
        equity_curve.append(equity)
        peak = max(peak, equity)
        dd = equity / peak - 1.0
        if dd < max_dd:
            max_dd = dd

    final_value = cash + qty * closes[-1]
    return {
        "return": final_value / starting_cash - 1.0,
        "trades": int(trades),
        "win_rate": wins / trades if trades else 0.0,
        "max_dd": float(max_dd),
    }


def grid_search(closes: np.ndarray, grid: dict[str, list] | None = None,
                min_trades: int = 5) -> tuple[dict | None, dict | None]:
    grid = grid or DEFAULT_GRID
    keys = list(grid)
    best_params = None
    best_metrics = None
    best_score = -np.inf

    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        m = simulate(closes, **params)
        if m["trades"] < min_trades:
            continue
        # 评分：收益减去回撤惩罚，鼓励多交易但稳健。
        score = m["return"] + 0.5 * m["max_dd"]
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = m

    return best_params, best_metrics


def osc_param_path(ticker: str) -> Path:
    return MODEL_DIR / f"osc_{ticker.upper()}.json"


def save_params(ticker: str, params: dict, metrics: dict) -> Path:
    payload = {"ticker": ticker.upper(), "params": params, "metrics": metrics}
    path = osc_param_path(ticker)
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


def load_params(ticker: str) -> dict | None:
    path = osc_param_path(ticker)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload.get("params")


def train_tickers(tickers: list[str], days: int = 7,
                  grid: dict[str, list] | None = None) -> None:
    """对每只票拉 1m 历史 → 网格搜索 → 保存到 models/osc_{TICKER}.json。"""
    period = f"{min(days, 30)}d"
    for t in tickers:
        try:
            df = data_mod.fetch_history(t, period=period, interval="1m")
        except Exception as exc:  # noqa: BLE001
            print(f"[osc-train] {t} 拉数据失败: {exc}")
            continue
        params, metrics = grid_search(df["close"].values, grid)
        if params is None:
            print(f"[osc-train] {t} 网格内没有满足最低交易数的组合，跳过")
            continue
        path = save_params(t, params, metrics)
        print(
            f"[osc-train] {t} → {path.name} "
            f"return={metrics['return']:+.2%} "
            f"trades={metrics['trades']} "
            f"win={metrics['win_rate']:.0%} "
            f"maxDD={metrics['max_dd']:.2%} "
            f"params={params}"
        )
