from typing_extensions import TypedDict
from typing import Optional, Any, Dict

class AgentState(TypedDict, total=False):
    ticker: str
    start_date: str

    prices_df: Any              # pandas DataFrame
    risk_metrics: Dict[str, float]

    filing_form: str
    filing_text: str
    filing_summary: str

    report_md: str
