def build_report_md(ticker: str, metrics: dict, filing_form: str, filing_summary: str) -> str:
    return f"""# {ticker} 投研速览（MVP）

## 风险指标（基于历史价格）
- 年化波动率（估算）：{metrics.get("ann_vol", 0):.2%}
- 最大回撤（历史区间）：{metrics.get("max_drawdown", 0):.2%}

## 最新 {filing_form} 摘要（LLM）
{filing_summary}

> 免责声明：以上为信息整理与分析，不构成投资建议。
"""
