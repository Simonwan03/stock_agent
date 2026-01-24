from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_PORTFOLIO_PATH = Path("data/portfolio.json")


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float  # 平均成本（按你输入的币种）
    currency: str = "USD"
    updated_at: str = ""  # ISO8601

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

# 加载投资组合从文件
def load_portfolio(path: Path) -> Dict[str, Position]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    positions: Dict[str, Position] = {}
    for sym, p in raw.get("positions", {}).items():
        positions[sym] = Position(**p)
    return positions

# 保存投资组合到文件
def save_portfolio(path: Path, positions: Dict[str, Position]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "positions": {sym: asdict(pos) for sym, pos in sorted(positions.items())},
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# 添加/加仓一个持仓
def add_position(
    positions: Dict[str, Position],
    symbol: str,
    quantity: float,
    price: float,
    currency: str = "USD",
) -> None:
    symbol = symbol.upper().strip()
    if quantity <= 0:
        raise ValueError("quantity must be > 0")
    if price <= 0:
        raise ValueError("price must be > 0")

    if symbol not in positions:
        pos = Position(symbol=symbol, quantity=quantity, avg_cost=price, currency=currency)
        pos.touch()
        positions[symbol] = pos
        return

    pos = positions[symbol]
    if pos.currency != currency:
        raise ValueError(f"currency mismatch for {symbol}: existing={pos.currency} new={currency}")

    # 加仓后的加权平均成本
    total_cost = pos.quantity * pos.avg_cost + quantity * price
    total_qty = pos.quantity + quantity
    pos.quantity = total_qty
    pos.avg_cost = total_cost / total_qty
    pos.touch()

# 卖出/减仓一个持仓 
def sell_position(
    positions: Dict[str, Position],
    symbol: str,
    quantity: float,
) -> None:
    symbol = symbol.upper().strip()
    if quantity <= 0:
        raise ValueError("quantity must be > 0")
    if symbol not in positions:
        raise ValueError(f"no position for {symbol}")

    pos = positions[symbol]
    if quantity > pos.quantity:
        raise ValueError(f"sell quantity exceeds holding: sell={quantity}, holding={pos.quantity}")

    pos.quantity -= quantity
    pos.touch()
    if pos.quantity == 0:
        del positions[symbol]

# 列出当前持仓
def print_positions(positions: Dict[str, Position]) -> None:
    if not positions:
        print("No positions.")
        return

    # 简单表格输出
    print(f"{'SYMBOL':<10} {'QTY':>12} {'AVG_COST':>12} {'CCY':>6} {'UPDATED_AT'}")
    for sym, pos in sorted(positions.items()):
        print(f"{sym:<10} {pos.quantity:>12.4f} {pos.avg_cost:>12.4f} {pos.currency:>6} {pos.updated_at}")

#
def fetch_last_price_openbb(symbol: str, provider: Optional[str] = None) -> Optional[float]:
    """
    返回最新收盘/最新价（尽力而为）。
    需要安装 openbb + 对应 equity provider。
    """
    try:
        from openbb import obb  # type: ignore
    except Exception:
        return None

    kwargs = {}
    if provider:
        kwargs["provider"] = provider

    # 取最近几条日线，拿最后一条 close
    out = obb.equity.price.historical(symbol, **kwargs)
    df = out.to_dataframe()
    if df.empty:
        return None

    # 尝试兼容不同列名
    for col in ("close", "Close", "adj_close", "Adj Close"):
        if col in df.columns:
            return float(df[col].iloc[-1])
    return None


def command_value(positions: Dict[str, Position], provider: Optional[str] = None) -> int:
    if not positions:
        print("No positions.")
        return 0

    total_value = 0.0
    total_cost = 0.0
    print(f"{'SYMBOL':<10} {'QTY':>12} {'AVG_COST':>12} {'LAST':>12} {'MV':>14} {'PnL':>14}")
    for sym, pos in sorted(positions.items()):
        last = fetch_last_price_openbb(sym, provider=provider)
        if last is None:
            print(f"{sym:<10} {pos.quantity:>12.4f} {pos.avg_cost:>12.4f} {'N/A':>12} {'N/A':>14} {'N/A':>14}")
            continue
        mv = pos.quantity * last
        cost = pos.quantity * pos.avg_cost
        pnl = mv - cost
        total_value += mv
        total_cost += cost
        print(f"{sym:<10} {pos.quantity:>12.4f} {pos.avg_cost:>12.4f} {last:>12.4f} {mv:>14.2f} {pnl:>14.2f}")

    if total_value > 0:
        print("-" * 80)
        print(f"{'TOTAL':<10} {'':>12} {'':>12} {'':>12} {total_value:>14.2f} {(total_value-total_cost):>14.2f}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record and view current investment portfolio.")
    p.add_argument("--file", default=str(DEFAULT_PORTFOLIO_PATH), help="Portfolio file path (default: data/portfolio.json)")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List current positions.")

    p_add = sub.add_parser("add", help="Add/accumulate a position.")
    p_add.add_argument("symbol")
    p_add.add_argument("quantity", type=float)
    p_add.add_argument("price", type=float, help="Fill price used to update avg cost.")
    p_add.add_argument("--currency", default="USD")

    p_sell = sub.add_parser("sell", help="Reduce a position quantity (avg cost unchanged).")
    p_sell.add_argument("symbol")
    p_sell.add_argument("quantity", type=float)

    p_val = sub.add_parser("value", help="Show market value using OpenBB last close (best-effort).")
    p_val.add_argument("--provider", default=None, help="OpenBB provider, e.g. yfinance")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.file)
    positions = load_portfolio(path)

    if args.cmd == "list":
        print_positions(positions)
        return 0

    if args.cmd == "add":
        add_position(positions, args.symbol, args.quantity, args.price, currency=args.currency)
        save_portfolio(path, positions)
        print(f"Added: {args.symbol.upper()} qty={args.quantity} price={args.price} {args.currency}")
        return 0

    if args.cmd == "sell":
        sell_position(positions, args.symbol, args.quantity)
        save_portfolio(path, positions)
        print(f"Sold: {args.symbol.upper()} qty={args.quantity}")
        return 0

    if args.cmd == "value":
        return command_value(positions, provider=args.provider)

    raise RuntimeError("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
