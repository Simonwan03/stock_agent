from openbb import obb

def get_price_history(ticker: str, start_date: str):
    # 返回 pandas.DataFrame，通常含 open/high/low/close/volume
    return obb.equity.price.historical(symbol=ticker, start_date=start_date).to_df()
