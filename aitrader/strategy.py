"""把模型概率翻译成买/卖/持有，并维护一个简易投资组合。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import CONFIG


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Position:
    qty: float = 0.0
    avg_cost: float = 0.0

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pct(self, price: float) -> float:
        if self.qty == 0 or self.avg_cost == 0:
            return 0.0
        return price / self.avg_cost - 1.0


@dataclass
class Portfolio:
    cash: float = CONFIG.starting_cash
    positions: dict[str, Position] = field(default_factory=dict)
    trade_log: list[dict] = field(default_factory=list)

    def equity(self, prices: dict[str, float]) -> float:
        total = self.cash
        for t, pos in self.positions.items():
            total += pos.market_value(prices.get(t, pos.avg_cost))
        return total

    def _pos(self, ticker: str) -> Position:
        return self.positions.setdefault(ticker, Position())

    def buy(self, ticker: str, price: float, ts, equity: float) -> bool:
        pos = self._pos(ticker)
        budget = min(self.cash, equity * CONFIG.max_position_per_ticker - pos.market_value(price))
        if budget <= 1.0 or price <= 0:
            return False
        qty = budget / price
        new_qty = pos.qty + qty
        pos.avg_cost = (pos.avg_cost * pos.qty + price * qty) / new_qty if new_qty > 0 else price
        pos.qty = new_qty
        self.cash -= qty * price
        self.trade_log.append({"ts": ts, "ticker": ticker, "side": "BUY", "price": price, "qty": qty})
        return True

    def sell(self, ticker: str, price: float, ts, qty: float | None = None) -> bool:
        pos = self._pos(ticker)
        if pos.qty <= 0:
            return False
        sell_qty = pos.qty if qty is None else min(qty, pos.qty)
        self.cash += sell_qty * price
        pos.qty -= sell_qty
        if pos.qty == 0:
            pos.avg_cost = 0.0
        self.trade_log.append({"ts": ts, "ticker": ticker, "side": "SELL", "price": price, "qty": sell_qty})
        return True


def decide(proba_up: float, position: Position, price: float) -> Action:
    """核心买卖规则：先看止损止盈，再看模型方向。"""
    if position.qty > 0:
        pnl = position.unrealized_pct(price)
        if pnl <= -CONFIG.stop_loss or pnl >= CONFIG.take_profit:
            return Action.SELL
        if proba_up < CONFIG.sell_threshold:
            return Action.SELL
        return Action.HOLD
    if proba_up > CONFIG.buy_threshold:
        return Action.BUY
    return Action.HOLD
