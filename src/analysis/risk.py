from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Dict, Iterable, List


def _to_float_list(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values]


def risk_metrics(close_prices: Iterable[float]) -> Dict[str, float]:
    closes = _to_float_list(close_prices)
    if len(closes) < 2:
        return {
            "volatility": 0.0,
            "max_drawdown": 0.0,
            "latest_close": closes[-1] if closes else 0.0,
        }

    returns = [(closes[i] / closes[i - 1] - 1.0) for i in range(1, len(closes))]
    daily_vol = pstdev(returns) if len(returns) > 1 else 0.0
    annualized_vol = daily_vol * sqrt(252)

    peak = closes[0]
    max_dd = 0.0
    for price in closes[1:]:
        if price > peak:
            peak = price
        drawdown = (price - peak) / peak if peak else 0.0
        max_dd = min(max_dd, drawdown)

    return {
        "volatility": round(annualized_vol, 6),
        "max_drawdown": round(max_dd, 6),
        "latest_close": round(closes[-1], 6),
        "avg_return": round(mean(returns), 6) if returns else 0.0,
    }


def compute_portfolio_snapshot(portfolio: Dict[str, object], prices: Dict[str, object]) -> Dict[str, object]:
    holdings = portfolio.get("holdings", [])
    snapshot = []
    total_value = 0.0

    for holding in holdings:
        ticker = holding.get("ticker")
        shares = float(holding.get("shares", 0))
        price_rows = prices.get(ticker)
        last_close = 0.0
        if isinstance(price_rows, list) and price_rows:
            last_close = float(price_rows[-1].get("close", 0.0))
        elif hasattr(price_rows, "__iter__"):
            price_list = list(price_rows)
            if price_list:
                last_close = float(price_list[-1])
        value = shares * last_close
        total_value += value
        snapshot.append(
            {
                "ticker": ticker,
                "shares": shares,
                "last_close": last_close,
                "market_value": value,
            }
        )

    return {
        "holdings": snapshot,
        "total_value": total_value,
    }
