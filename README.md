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

> 思路：盯每只票最近 N 分钟的均线。当前价比均线低 X 个标准差就买，回到均线上方一点就卖，配止盈/止损兜底。盘中持续 run。
>
> **参数会被训练**：`osc-train` 用最近 7 天 1m 历史做网格搜索，对每只票挑出最优 (window, buy_z, sell_z, take_profit, stop_loss)，存到 `models/osc_{TICKER}.json`。`osc` 启动时会自动加载。

```bash
# 1. 训练每只票的最优参数（一次性，几十秒到几分钟）
python -m aitrader.main osc-train
python -m aitrader.main osc-train --tickers AAPL TSLA --days 14

# 2. 跑实盘纸面震荡，自动加载训练参数
python -m aitrader.main osc
python -m aitrader.main osc --tickers AAPL TSLA NVDA

# 3. 想手动覆盖单个参数也行（其余仍用训练值）
python -m aitrader.main osc --buy-z -0.8 --tp 0.003
```

**参数表**

| 参数 | 默认（未训练时） | 含义 |
| --- | --- | --- |
| `--tickers` | AAPL/MSFT/NVDA/TSLA/SPY | 监控标的 |
| `--window` | 20 | 滚动均线窗口（分钟） |
| `--buy-z` | -1.0 | 价格跌到均线下几个标准差时买入 |
| `--sell-z` | 0.5 | 价格涨到均线上几个标准差时卖出 |
| `--tp` | 0.005 | 止盈，0.5% |
| `--sl` | 0.01 | 止损，1% |
| `--poll` | 30 | 刷新秒数 |
| `--cash` | 10000 | 起始资金 |

参数加载优先级：**命令行覆盖 > `models/osc_{TICKER}.json` 训练值 > 上表默认值**。

**训练评分**：网格内对每个组合在最近 7 天 1m 数据上做回测，评分 = 总收益 + 0.5 × 最大回撤（即收益高、回撤小的组合得分高），并要求最少 5 笔交易避免过拟合到不动手的组合。

### 自动刹车（5 条规则，永远开着）

实盘里 bot 最大的问题是**算完就执行不会停**。`osc` 跑起来自带这套自我刹车（写在 `aitrader/safety.py`）：

| # | 规则 | 触发条件 | 动作 |
|---|------|----------|------|
| 1 | **连亏冷静期** | 最近 3 笔都亏 | 停止开新仓 120 分钟 |
| 2 | **持仓亏损不加仓** | 当前票浮亏 | 跳过这次买入信号 |
| 3 | **不追涨** | 过去 5 分钟已涨 ≥1% | 跳过 |
| 4 | **不接刀** | 最近 5 根 K 线连续 3 根创新低 | 跳过，等横盘 |
| 5 | **抽风日休市** | 最近 30 分钟波动 ≥ 平日 2 倍 | 跳过 |

被刹车跳过的会显示 `SKIP(原因)`，方便复盘。回测里看不出这套规则的差别，但**实盘活命主要靠它**。

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
  osc_train.py    # 震荡策略参数搜索 + 保存/加载（osc-train）
  features.py     # 波动惯性特征（AI 模式用）
  model.py        # 训练 / 保存 / 预测（AI 模式用）
  strategy.py     # 通用投资组合 + AI 模式买卖逻辑
  trader.py       # AI 模式纸面交易循环
  backtest.py     # AI 模式回测
  main.py         # CLI 入口
models/           # AI 模型 .joblib + osc_*.json 最优参数
logs/             # 交易事件 jsonl
```

## 提醒

- 全部纸面，不会真的下单。要实盘需要接券商（Alpaca/IBKR 等）。
- yfinance 的 1 分钟数据最多回看 ~7 天，无需 API key 但偶尔会限流。
- 美股盘前/盘后行情稀疏，建议主时段 21:30–04:00（北京时间）跑。
- 模型 / 策略只是基线，参数务必先在小资金或回测里验证。
