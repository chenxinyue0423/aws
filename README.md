# AI 美股交易系统

一个聚焦的美股自动交易项目，提供两种模式：

- **`osc` 日内震荡（推荐，简单）**：跌了买、涨了卖，盘中持续 run，一天可以来回做几十笔。**无需训练**。
- **`live` AI 惯性版（进阶）**：先用历史数据训一个动量分类器，再按它的概率做买卖。

## 安装

```bash
pip install -r requirements.txt
```

---

## 模式 A：日内震荡（推荐）

> 思路：盯每只票最近 20 分钟的均线。当前价比均线低 1 个标准差就买，回到均线上方一点就卖。配 0.5% 止盈、1% 止损做兜底。盘中持续 run。

```bash
# 默认 5 只票，10000 起步资金，每 30 秒刷新一次
python -m aitrader.main osc

# 自定义标的
python -m aitrader.main osc --tickers AAPL TSLA NVDA

# 调更激进或更保守
python -m aitrader.main osc --window 15 --buy-z -0.8 --sell-z 0.3 --tp 0.003 --sl 0.008
```

**参数表**

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--tickers` | AAPL/MSFT/NVDA/TSLA/SPY | 监控标的 |
| `--window` | 20 | 滚动均线窗口（分钟） |
| `--buy-z` | -1.0 | 价格跌到均线下几个标准差时买入 |
| `--sell-z` | 0.5 | 价格涨到均线上几个标准差时卖出 |
| `--tp` | 0.005 | 止盈，0.5% |
| `--sl` | 0.01 | 止损，1% |
| `--poll` | 30 | 刷新秒数 |
| `--cash` | 10000 | 起始资金 |

终端会持续打印每只票的 z 分数与触发的买卖，所有事件同时写到 `logs/osc_*.jsonl`。

---

## 模式 B：AI 惯性版（进阶）

每只票训一个 GradientBoosting 分类器，用多周期动量+波动率+RSI+MACD 等 16 个特征预测「未来 N 根 K 线方向概率」。概率高于阈值买、低于阈值卖。

```bash
python -m aitrader.main train       # 训练（基于 2 年日线历史）
python -m aitrader.main backtest    # 回测
python -m aitrader.main live        # 用训练好的模型做纸面交易
```

调参在 `aitrader/config.py`（horizon、buy_threshold、sell_threshold、stop_loss、take_profit、max_position_per_ticker）。

---

## 其他

```bash
python -m aitrader.main quote --tickers AAPL TSLA   # 看实时报价
```

## 文件结构

```
aitrader/
  config.py       # 集中参数
  data.py         # yfinance 拉数据 / 实时流
  oscillator.py   # 日内震荡策略（osc 模式）
  features.py     # 波动惯性特征（AI 模式用）
  model.py        # 训练 / 保存 / 预测（AI 模式用）
  strategy.py     # 通用投资组合 + AI 模式买卖逻辑
  trader.py       # AI 模式纸面交易循环
  backtest.py     # AI 模式回测
  main.py         # CLI 入口
models/           # AI 模型 .joblib
logs/             # 交易事件 jsonl
```

## 提醒

- 全部纸面，不会真的下单。要实盘需要接券商（Alpaca/IBKR 等）。
- yfinance 的 1 分钟数据最多回看 ~7 天，无需 API key 但偶尔会限流。
- 美股盘前/盘后行情稀疏，建议主时段 21:30–04:00（北京时间）跑。
- 模型 / 策略只是基线，参数务必先在小资金或回测里验证。
