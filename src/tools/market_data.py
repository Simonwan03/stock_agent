import csv
import io
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 关于数据源：Yahoo Finance（不用api）Stooq（免费api）Alpha Vantage（免费api但有调用频率限制）IEX Cloud（免费api但有调用频率限制）Tiingo（免费api但有调用频率限制）Polygon.io（免费api但有调用频率限制）

def get_price_history(ticker: str, start_date: str, interval: str = "1d") -> object:
    from openbb import obb

    return obb.equity.price.historical(symbol=ticker, start_date=start_date, interval=interval).to_df()

def get_income_statement(ticker: str, period: str = "quarter") -> object:
    from openbb import obb

    return obb.equity.fundamental.income(symbol=ticker, period=period).to_df()


@dataclass(frozen=True)
class PriceSeries:
    ticker: str
    df: List[Dict[str, float]]


def _stooq_symbol(ticker: str) -> str:
    return f"{ticker.lower()}.us"


def _fetch_stooq_daily(ticker: str) -> Optional[List[Dict[str, float]]]:
    symbol = _stooq_symbol(ticker)
    url = f"https://stooq.pl/q/d/l/?s={symbol}&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (stock-agent)"})
    with urllib.request.urlopen(req, timeout=20) as response:
        content = response.read().decode("utf-8", errors="replace")
    if not content.strip():
        return None

    reader = csv.DictReader(io.StringIO(content))
    rows: List[Dict[str, float]] = []
    for row in reader:
        close = row.get("Close")
        date_str = row.get("Date")
        if not close or not date_str:
            continue
        try:
            rows.append({"date": date_str, "close": float(close)})
        except ValueError:
            continue
    return rows or None


def fetch_daily_closes(
    tickers: List[str],
    lookback_days: int = 30,
    provider: str = "yfinance",
) -> Dict[str, PriceSeries]:
    """
    Best-effort daily close series with minimal dependencies.
    Returns a dict of ticker -> PriceSeries where df is a list of {date, close}.
    """
    today = datetime.utcnow().date()
    series: Dict[str, PriceSeries] = {}
    for ticker in tickers:
        rows: Optional[List[Dict[str, float]]] = None
        if provider.lower() == "stooq":
            try:
                rows = _fetch_stooq_daily(ticker)
            except Exception:
                rows = None
        if rows is None:
            rows = []
            for offset in range(lookback_days):
                day = today - timedelta(days=lookback_days - offset)
                rows.append({"date": day.isoformat(), "close": 0.0})
        series[ticker] = PriceSeries(ticker=ticker, df=rows)
    return series

def main(args):
    ticker = args[0] if len(args) > 0 else "AAPL"
    start_date = args[1] if len(args) > 1 else "2026-01-01"
    end_date = args[2] if len(args) > 2 else None
    
    print(f"Fetching price history for {ticker} since {start_date}...")
    df_price_history = get_price_history(ticker, start_date)
    print(df_price_history)

    print(f"\nFetching income statement for {ticker}...")
    df_income = get_income_statement(ticker)
    # print(df_income.head())
    cols = ["period_ending", "revenue", "gross_profit", "operating_income", "net_income"]
    available = [c for c in cols if c in df_income.columns]
    print(df_income[available].head())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
