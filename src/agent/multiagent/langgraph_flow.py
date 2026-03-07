from __future__ import annotations
from typing import TypedDict, Dict, Any, Optional, List
from pathlib import Path
import json

from langgraph.graph import StateGraph, START, END

MODULES = ["market_data", "financials", "indicators", "news", "portfolio"]

class AgentState(TypedDict, total=False):
    ticker: str
    outputs_dir: str
    llm_enabled: bool

    # 运行中间产物
    source_files: Dict[str, str]                # module -> filepath
    module_results: Dict[str, Dict[str, Any]]   # module -> {summary, data, ...}
    advice: Dict[str, Any]                      # advisor output


def pick_latest_file(outputs_dir: str, module: str, ticker: str) -> Optional[str]:
    """
    这里按你的项目命名规则实现：
    - 例如 market_data 可能是 outputs/AAPL.json
    - news 可能是 outputs/AAPL_news.json
    - indicators 可能是 outputs/aapl_indicators.json
    你也可以直接复用现有 orchestrator 里的“挑最新文件”函数。
    """
    od = Path(outputs_dir)
    patterns = {
        "market_data": [f"{ticker}.json", f"{ticker.upper()}.json"],
        "news": [f"{ticker}_news.json", f"{ticker.upper()}_news.json"],
        "indicators": [f"{ticker.lower()}_indicators.json", f"{ticker.upper()}_indicators.json"],
        "portfolio": ["portfolio.json"],
        "financials": [f"aapl_compact.json", f"{ticker.lower()}_compact.json", f"{ticker.upper()}_compact.json"],
    }.get(module, [])

    candidates: List[Path] = []
    for p in patterns:
        candidates.extend(od.glob(p))

    if not candidates:
        return None
    latest = max(candidates, key=lambda x: x.stat().st_mtime)
    return str(latest)


def module_node(module: str):
    def _node(state: AgentState) -> AgentState:
        ticker = state["ticker"]
        outputs_dir = state["outputs_dir"]
        llm_enabled = state.get("llm_enabled", True)

        fp = pick_latest_file(outputs_dir, module, ticker)
        state.setdefault("source_files", {})
        state.setdefault("module_results", {})

        if not fp:
            # 没文件就记录并跳过（也可以用 conditional edges 做“直接不走这个 node”）
            state["source_files"][module] = ""
            state["module_results"][module] = {
                "module": module,
                "summary": f"No input JSON found for {module}.",
                "data": {},
            }
            return state

        state["source_files"][module] = fp
        payload = json.loads(Path(fp).read_text(encoding="utf-8"))

        # ✅ 这里复用你现有的 agent：比如 agents.py 里类似 summarize_market_data(payload)
        # 下面用伪函数表示：
        if llm_enabled:
            summary, data = run_llm_summary(module, ticker, payload)  # 你已有的 LLM client + prompt
        else:
            summary, data = rule_based_summary(module, payload)

        state["module_results"][module] = {
            "module": module,
            "source_files": [fp],
            "summary": summary,
            "data": data,
        }
        return state

    return _node


def advisor_node(state: AgentState) -> AgentState:
    # 把 module_results 汇总给 advisor（你现在 run.py 的逻辑）:contentReference[oaicite:7]{index=7}
    modules = state.get("module_results", {})
    if state.get("llm_enabled", True):
        advice = run_llm_advisor(state["ticker"], modules)
    else:
        advice = {"summary": "LLM disabled; no advice generated.", "signals": [], "risk_notes": []}
    state["advice"] = advice
    return state


def build_graph():
    g = StateGraph(AgentState)

    for m in MODULES:
        g.add_node(m, module_node(m))
    g.add_node("advisor", advisor_node)

    g.add_edge(START, "market_data")
    g.add_edge("market_data", "financials")
    g.add_edge("financials", "indicators")
    g.add_edge("indicators", "news")
    g.add_edge("news", "portfolio")
    g.add_edge("portfolio", "advisor")
    g.add_edge("advisor", END)

    return g.compile()  # LangGraph 必须 compile 后才能 invoke/stream :contentReference[oaicite:8]{index=8}
