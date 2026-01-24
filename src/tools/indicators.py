import pandas as pd

# 计算年化波动率和最大回撤
def risk_metrics(close: pd.Series) -> dict:
    rets = close.pct_change().dropna()

    ann_vol = float(rets.std() * (252 ** 0.5))

    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    max_dd = float(dd.min())

    return {"ann_vol": ann_vol, "max_drawdown": max_dd}
