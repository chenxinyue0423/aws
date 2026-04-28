"""5 条「会犹豫」的刹车规则。

bot 最大的问题不是不会算，是算完就执行、不会停。
这个模块给它装一个停的开关：
1. 连续亏损就缓一缓（亏 N 笔后冷静一段时间，不开新仓）
2. 已经亏着的票不加仓
3. 不追刚涨完的（短时间内已经涨太多，跳过）
4. 不接还在掉的（价格还在阶梯式下跌，先等横盘）
5. 市场抽风时少动手（当下波动比平日大很多，仓位砍半或休市）
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .strategy import Position


@dataclass
class SafetyConfig:
    # 规则 1
    loss_streak: int = 3              # 连续多少笔亏损触发冷静
    cooldown_minutes: int = 120       # 冷静多久（分钟）
    # 规则 3
    chase_lookback: int = 5           # 看过去几分钟
    chase_pct: float = 0.01           # 已涨过 1% 就不追
    # 规则 4
    falling_lookback: int = 5         # 看过去几分钟是否还在跌
    # 规则 5
    vol_lookback_recent: int = 30     # 当下波动用最近 30 根
    vol_lookback_baseline: int = 300  # 基线用 30~300 根
    vol_outlier_mult: float = 2.0     # 超过基线 2 倍算抽风


@dataclass
class SafetyState:
    cfg: SafetyConfig = field(default_factory=SafetyConfig)
    recent_pnls: deque = field(default_factory=lambda: deque(maxlen=10))
    cooldown_until: datetime | None = None

    def in_cooldown(self, now: datetime) -> bool:
        return self.cooldown_until is not None and now < self.cooldown_until

    def record_trade(self, pnl_pct: float, now: datetime) -> str | None:
        """每次平仓后调用，返回触发的提示（无则 None）。"""
        self.recent_pnls.append(pnl_pct)
        n = self.cfg.loss_streak
        if len(self.recent_pnls) >= n and all(p < 0 for p in list(self.recent_pnls)[-n:]):
            self.cooldown_until = now + timedelta(minutes=self.cfg.cooldown_minutes)
            return f"连续 {n} 笔亏损，冷静 {self.cfg.cooldown_minutes} 分钟至 {self.cooldown_until:%H:%M}"
        return None


def check_buy(
    state: SafetyState,
    df: pd.DataFrame,
    position: Position,
    now: datetime,
) -> tuple[bool, str]:
    """决定要不要让这一笔买入通过。返回 (放行?, 理由)。"""
    cfg = state.cfg
    closes = df["close"]
    price = float(closes.iloc[-1])

    # 规则 1：连续亏损冷静期
    if state.in_cooldown(now):
        return False, f"冷静期至 {state.cooldown_until:%H:%M}"

    # 规则 2：已持仓且亏损，不加仓
    if position.qty > 0 and position.unrealized_pct(price) < 0:
        return False, f"持仓亏 {position.unrealized_pct(price):.2%}，不加仓"

    # 规则 3：过去 N 分钟已涨太多，不追
    if len(closes) > cfg.chase_lookback:
        ret = price / closes.iloc[-cfg.chase_lookback - 1] - 1
        if ret >= cfg.chase_pct:
            return False, f"过去{cfg.chase_lookback}分钟已涨 {ret:.2%}，不追"

    # 规则 4：还在阶梯式下跌（最近 5 根有连续低点），等横盘
    n = cfg.falling_lookback
    if len(closes) >= n:
        last = closes.iloc[-n:].values
        # 至少 3 根连续创新低就算"还在掉"
        consecutive_lows = 0
        for i in range(1, len(last)):
            if last[i] < last[i - 1]:
                consecutive_lows += 1
            else:
                consecutive_lows = 0
            if consecutive_lows >= 3:
                return False, f"还在阶梯下跌中，等横盘"

    # 规则 5：当下波动远高于基线 → 抽风日，不动手
    if len(closes) >= cfg.vol_lookback_baseline:
        rets = closes.pct_change().dropna()
        recent_vol = rets.iloc[-cfg.vol_lookback_recent:].std()
        baseline_vol = rets.iloc[-cfg.vol_lookback_baseline:-cfg.vol_lookback_recent].std()
        if baseline_vol and not np.isnan(baseline_vol) and recent_vol > cfg.vol_outlier_mult * baseline_vol:
            return False, f"波动是平日 {recent_vol/baseline_vol:.1f} 倍，市场抽风"

    return True, "ok"
