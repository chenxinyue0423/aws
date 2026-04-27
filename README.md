# AI 美股交易系统

一个聚焦的美股 AI 交易项目：实时拉数据 → 学习「波动惯性」 → 在合适的位置买入/卖出。

## 核心机制

1. **数据**：`yfinance` 拉美股 OHLCV，分钟到日线粒度都支持，无需 API key。
2. **波动惯性特征**（`aitrader/features.py`）：多周期收益率（1/3/5/10/20）、滚动波动率、SMA 斜率与差距、MACD、RSI、ATR、量比、量能动量、20 日通道位置。这些是「惯性」的量化表达——价格走势倾向于延续，特征用来描述这种延续的强度与方向。
3. **AI 模型**（`aitrader/model.py`）：`GradientBoostingClassifier`，对每只票单独训练，学习「当前的惯性特征 → 未来 N 根 K 线方向」。输出的是上涨概率。
4. **买卖逻辑**（`aitrader/strategy.py`）：
   - 概率 > `buy_threshold`（默认 0.58）→ 在仓位上限内买入
   - 概率 < `sell_threshold`（默认 0.42）→ 平仓
   - 单票止损 `-5%` / 止盈 `+10%` 强制平仓
   - 单票最高仓位占总资产 25%
5. **回测**（`aitrader/backtest.py`）：walk-forward 把模型每根 bar 输出的概率喂给同一套买卖逻辑，得出最终收益。
6. **实时纸面交易**（`aitrader/trader.py`）：每隔 60s 轮询新 K 线，用已训练好的模型推断并自动买卖（不接券商，纯模拟，所有动作写到 `logs/` jsonl）。

## 安装

```bash
pip install -r requirements.txt
```

## 用法

```bash
# 1. 训练模型（默认 AAPL/MSFT/NVDA/TSLA/SPY，2 年日线）
python -m aitrader.main train

# 2. 历史回测，看胜率与收益
python -m aitrader.main backtest

# 3. 纸面实时交易（盘中跑），按 Ctrl+C 退出
python -m aitrader.main live

# 4. 查看实时报价
python -m aitrader.main quote --tickers AAPL TSLA

# 自定义标的
python -m aitrader.main train --tickers AAPL META AMZN
```

## 调参

修改 `aitrader/config.py`：

| 字段 | 含义 |
| --- | --- |
| `tickers` | 默认追踪标的 |
| `train_period` / `train_interval` | 训练用历史长度与周期 |
| `live_interval` | 实时交易使用的 K 线粒度（默认 15m） |
| `horizon` | 模型预测向前几根 bar 的方向 |
| `buy_threshold` / `sell_threshold` | 概率到多少才动手 |
| `stop_loss` / `take_profit` | 强制止损止盈 |
| `max_position_per_ticker` | 单票最高仓位占比 |
| `poll_seconds` | 实时循环间隔 |

## 文件结构

```
aitrader/
  config.py     # 集中参数
  data.py       # yfinance 拉数据 / 实时流
  features.py   # 波动惯性特征工程
  model.py      # 训练 / 保存 / 加载 / 预测
  strategy.py   # 买卖决策 + 投资组合
  backtest.py   # 回测
  trader.py     # 纸面实时交易主循环
  main.py       # CLI 入口
models/         # 训练好的 .joblib
logs/           # 纸面交易 jsonl 流水
```

## 局限与提醒

- 纯纸面：不会真的下单，需要接券商（如 Alpaca）才能实盘。
- yfinance 分钟级最多回看 ~60 天，再细粒度需要付费数据源。
- 模型只学过去模式，遇到结构性变化（财报、突发事件）会失效——`stop_loss` 是兜底。
- 美股盘后/盘前数据稀疏，建议主时段运行。
