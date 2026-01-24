import sys
import pandas as pd
from openbb import obb

# 关于数据源：Yahoo Finance（不用api）Stooq（免费api）Alpha Vantage（免费api但有调用频率限制）IEX Cloud（免费api但有调用频率限制）Tiingo（免费api但有调用频率限制）Polygon.io（免费api但有调用频率限制）

def get_price_history(ticker: str, start_date: str, interval: str = "1d") -> "pd.DataFrame":
    # 返回 pandas.DataFrame，通常含 open/high/low/close/volume
    return obb.equity.price.historical(symbol=ticker, start_date=start_date, interval=interval).to_df()

def get_income_statement(ticker: str, period: str = "quarter") -> "pd.DataFrame":
    # period: annual or quarterly
    return obb.equity.fundamental.income(symbol=ticker, period=period).to_df()

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


