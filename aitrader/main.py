"""命令行入口：train / backtest / live / quote。"""

from __future__ import annotations

import argparse
import json

from . import data as data_mod
from . import model as ai_model
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
    run_oscillator(
        tickers=args.tickers or CONFIG.tickers,
        window=args.window,
        buy_z=args.buy_z,
        sell_z=args.sell_z,
        take_profit=args.tp,
        stop_loss=args.sl,
        poll_seconds=args.poll,
        starting_cash=args.cash,
    )


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

    p_osc = sub.add_parser("osc", help="日内震荡策略：跌了买、涨了卖，持续 run（推荐）")
    p_osc.add_argument("--tickers", nargs="+")
    p_osc.add_argument("--window", type=int, default=20, help="均线窗口（分钟）")
    p_osc.add_argument("--buy-z", dest="buy_z", type=float, default=-1.0, help="跌到几个标准差以下买入")
    p_osc.add_argument("--sell-z", dest="sell_z", type=float, default=0.5, help="涨到几个标准差以上卖出")
    p_osc.add_argument("--tp", type=float, default=0.005, help="止盈比例，例如 0.005=0.5%%")
    p_osc.add_argument("--sl", type=float, default=0.01, help="止损比例，例如 0.01=1%%")
    p_osc.add_argument("--poll", type=int, default=30, help="刷新秒数")
    p_osc.add_argument("--cash", type=float, default=CONFIG.starting_cash, help="起始资金")
    p_osc.set_defaults(func=cmd_osc)

    p_quote = sub.add_parser("quote", help="打印实时价格")
    p_quote.add_argument("--tickers", nargs="+")
    p_quote.set_defaults(func=cmd_quote)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
