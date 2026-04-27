"""集中放置所有可调参数。"""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


@dataclass
class TradeConfig:
    tickers: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"])

    # 训练时使用的历史长度与 K 线周期。
    train_period: str = "2y"
    train_interval: str = "1d"

    # 实盘/纸面交易使用的更细粒度。yfinance 对分钟级最多只给最近 ~60 天。
    live_period: str = "30d"
    live_interval: str = "15m"

    # 模型在多远之后判定方向：用未来 horizon 根 K 线的收益符号作为 label。
    horizon: int = 3

    # 概率阈值：>buy_threshold 视为强惯性向上，<sell_threshold 视为向下。
    buy_threshold: float = 0.58
    sell_threshold: float = 0.42

    # 风控。
    starting_cash: float = 10_000.0
    max_position_per_ticker: float = 0.25  # 单票最高占总资金比例
    stop_loss: float = 0.05                # -5% 强制止损
    take_profit: float = 0.10              # +10% 强制止盈

    # 实盘循环刷新秒数。
    poll_seconds: int = 60


CONFIG = TradeConfig()
