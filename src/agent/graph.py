from langgraph.graph import StateGraph, END
from agent.schemas import AgentState

from tools.market_data import get_price_history
from tools.indicators import risk_metrics
from tools.sec_filings import get_latest_filing_text
from analysis.report import build_report_md


def fetch_market_data(state: AgentState) -> dict:
    df = get_price_history(state["ticker"], state["start_date"])
    return {"prices_df": df}

def compute_metrics(state: AgentState) -> dict:
    df = state["prices_df"]
    metrics = risk_metrics(df["close"])
    return {"risk_metrics": metrics}

def fetch_sec_filing(state: AgentState) -> dict:
    form = state.get("filing_form", "10-Q")
    text = get_latest_filing_text(state["ticker"], form=form)
    return {"filing_form": form, "filing_text": text}

def summarize_filing(state: AgentState) -> dict:
    # MVP：先别接 LLM，跑通流程最重要
    raw = state["filing_text"]
    summary = f"-（占位）已获取 filing 文本长度：{len(raw)}\n- 下一步：接入 LLM 做“要点/风险/关键数字”抽取"
    return {"filing_summary": summary}

def write_report(state: AgentState) -> dict:
    md = build_report_md(
        ticker=state["ticker"],
        metrics=state["risk_metrics"],
        filing_form=state["filing_form"],
        filing_summary=state["filing_summary"],
    )
    return {"report_md": md}

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("fetch_market_data", fetch_market_data)
    g.add_node("compute_metrics", compute_metrics)
    g.add_node("fetch_sec_filing", fetch_sec_filing)
    g.add_node("summarize_filing", summarize_filing)
    g.add_node("write_report", write_report)

    g.set_entry_point("fetch_market_data")
    g.add_edge("fetch_market_data", "compute_metrics")
    g.add_edge("compute_metrics", "fetch_sec_filing")
    g.add_edge("fetch_sec_filing", "summarize_filing")
    g.add_edge("summarize_filing", "write_report")
    g.add_edge("write_report", END)

    return g.compile()
