"""命令行入口：train / backtest / live / quote。"""

from __future__ import annotations

import argparse
import json

from . import data as data_mod
from . import model as ai_model
from . import osc_train
from .backtest import backtest
from .config import CONFIG
from .oscillator import run_oscillator
from .trader import run_paper_trading


def cmd_train(args: argparse.Namespace) -> None:
    tickers = args.tickers or CONFIG.tickers
    prices = data_mod.fetch_many(tickers, CONFIG.train_period, CONFIG.train_interval)
    for t, df in prices.items():
        try:
            pipe, metrics = ai_model.train_one(df)
            ai_model.save(t, pipe)
            print(f"[train] {t} 已保存 acc={metrics['accuracy']:.3f} auc={metrics['auc']:.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"[train] {t} 失败: {exc}")


def cmd_backtest(args: argparse.Namespace) -> None:
    tickers = args.tickers or CONFIG.tickers
    prices = data_mod.fetch_many(tickers, CONFIG.train_period, CONFIG.train_interval)
    result = backtest(prices)
    summary = {k: v for k, v in result.items() if k != "trade_log"}
    print(json.dumps(summary, indent=2, default=str))
    print(f"产生 {result['trades']} 笔交易，最终收益率 {result['return_pct']*100:.2f}%")


def cmd_live(args: argparse.Namespace) -> None:
    run_paper_trading(args.tickers or CONFIG.tickers)


def cmd_osc(args: argparse.Namespace) -> None:
    # 只把用户显式传过的参数当 override，其他用训练好的或默认值。
    override: dict = {}
    if args.window is not None:
        override["window"] = args.window
    if args.buy_z is not None:
        override["buy_z"] = args.buy_z
    if args.sell_z is not None:
        override["sell_z"] = args.sell_z
    if args.tp is not None:
        override["take_profit"] = args.tp
    if args.sl is not None:
        override["stop_loss"] = args.sl
    run_oscillator(
        tickers=args.tickers or CONFIG.tickers,
        override=override or None,
        poll_seconds=args.poll,
        starting_cash=args.cash,
    )


def cmd_osc_train(args: argparse.Namespace) -> None:
    osc_train.train_tickers(args.tickers or CONFIG.tickers, days=args.days)


def cmd_quote(args: argparse.Namespace) -> None:
    for t in (args.tickers or CONFIG.tickers):
        try:
            price, ts = data_mod.latest_quote(t)
            print(f"{t}: {price:.2f} @ {ts}")
        except Exception as exc:  # noqa: BLE001
            print(f"{t}: 拉取失败 {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 美股交易系统：波动惯性学习 + 自动买卖")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="基于历史数据训练每只标的的惯性模型")
    p_train.add_argument("--tickers", nargs="+")
    p_train.set_defaults(func=cmd_train)

    p_back = sub.add_parser("backtest", help="使用 walk-forward 概率回测")
    p_back.add_argument("--tickers", nargs="+")
    p_back.set_defaults(func=cmd_backtest)

    p_live = sub.add_parser("live", help="纸面实时交易（无真实下单）")
    p_live.add_argument("--tickers", nargs="+")
    p_live.set_defaults(func=cmd_live)

    p_osc = sub.add_parser("osc", help="日内震荡：跌了买、涨了卖。自动加载训练好的参数（先跑 osc-train）")
    p_osc.add_argument("--tickers", nargs="+")
    p_osc.add_argument("--window", type=int, help="均线窗口（分钟），覆盖训练值")
    p_osc.add_argument("--buy-z", dest="buy_z", type=float, help="买入 z 阈值，覆盖训练值")
    p_osc.add_argument("--sell-z", dest="sell_z", type=float, help="卖出 z 阈值，覆盖训练值")
    p_osc.add_argument("--tp", type=float, help="止盈比例，覆盖训练值，例如 0.005=0.5%%")
    p_osc.add_argument("--sl", type=float, help="止损比例，覆盖训练值，例如 0.01=1%%")
    p_osc.add_argument("--poll", type=int, default=30, help="刷新秒数")
    p_osc.add_argument("--cash", type=float, default=CONFIG.starting_cash, help="起始资金")
    p_osc.set_defaults(func=cmd_osc)

    p_osct = sub.add_parser("osc-train", help="网格搜索每只票最优震荡参数，存到 models/osc_*.json")
    p_osct.add_argument("--tickers", nargs="+")
    p_osct.add_argument("--days", type=int, default=7, help="使用最近多少天 1m 数据回测（最多 30）")
    p_osct.set_defaults(func=cmd_osc_train)

    p_quote = sub.add_parser("quote", help="打印实时价格")
    p_quote.add_argument("--tickers", nargs="+")
    p_quote.set_defaults(func=cmd_quote)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
