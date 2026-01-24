import os
from edgar import Company, set_identity

def init_sec_identity():
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if ua:
        set_identity(ua)

def get_latest_filing_text(ticker: str, form: str = "10-Q") -> str:
    init_sec_identity()
    c = Company(ticker)
    filings = c.get_filings(form=form)
    latest = filings.latest()

    # 不同版本的 EdgarTools 对“正文/附件”访问方式可能略不同；
    # 这里先做“最保守”：把 filing 对象字符串化，MVP 能跑通。
    return str(latest)
