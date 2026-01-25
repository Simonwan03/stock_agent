# 目标：把每天任务串起来：读组合 → 拉行情/新闻 → 算风险 → 让 LLM 合成结构化报告 → 渲染成 Markdown → 写到 outputs。
# orchestrator.py       # 每天任务的“指挥官” --- IGNORE ---
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

from stock_agent.src.agent.config import Settings
from stock_agent.src.portfolio.store import load_portfolio
from stock_agent.src.tools.market_data import fetch_daily_closes
from stock_agent.src.tools.news import fetch_news, NewsItem
from stock_agent.src.analysis.risk import risk_metrics, compute_portfolio_snapshot
from stock_agent.src.llm.client import LLMClient
from stock_agent.src.llm.schemas import DailyReport
from stock_agent.src.render.render import render_markdown, write_text


def _news_to_digest(items: List[NewsItem]) -> List[Dict[str, Any]]:
    """
    把 NewsItem（dataclass）转成纯 dict，便于 JSON 序列化喂给 LLM。
    """
    out: List[Dict[str, Any]] = []
    for n in items:
        out.append(
            {
                "title": n.title,
                "url": n.url,
                "source": n.source,
                "published_at": n.published_at,
                "tickers": n.tickers,
                "summary": n.summary,
            }
        )
    return out


def run_daily(portfolio_path: str, settings: Settings) -> str:
    """
    主入口：生成当日组合日报，返回 markdown 输出文件路径。
    """
    # 0) 日期
    today = date.today().isoformat()

    # 1) 读组合
    portfolio = load_portfolio(portfolio_path)
    tickers = [h["ticker"] for h in portfolio.get("holdings", [])]
    benchmarks = portfolio.get("benchmarks", [])
    all_tickers = sorted(set(tickers + benchmarks))

    if not tickers:
        raise ValueError("portfolio.holdings 为空：请在 portfolio.json 里填入持仓。")

    # 2) 数据采集（此处由 tools 层负责具体实现）
    price_series = fetch_daily_closes(all_tickers, lookback_days=settings.price_lookback_days)
    news_items = fetch_news(tickers, lookback_hours=settings.news_lookback_hours)

    # 3) 计算指标（量化侧）
    prices_df = {t: ps.df for t, ps in price_series.items()}

    # 组合快照：权重/市值等（默认用 shares * last_close 估算）
    snap = compute_portfolio_snapshot(portfolio, {t: prices_df[t] for t in tickers})

    # 每个持仓：年化波动、最大回撤等
    per_ticker_risk: Dict[str, Any] = {}
    for t in tickers:
        close = prices_df[t]["close"]
        per_ticker_risk[t] = risk_metrics(close)

    # 4) 打包给 LLM 的输入（严格 JSON）
    payload = {
        "date": today,
        "portfolio": portfolio,
        "portfolio_snapshot": snap,
        "per_ticker_risk": per_ticker_risk,
        "news_digest": _news_to_digest(news_items),
    }

    # 5) LLM 合成结构化报告（JSON -> pydantic 校验）
    prompt_path = Path(__file__).with_name("llm").joinpath("prompts", "daily_report.md")
    prompt_tpl = prompt_path.read_text(encoding="utf-8")
    prompt = prompt_tpl.replace("{{INPUT_JSON}}", json.dumps(payload, ensure_ascii=False))

    llm = LLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    report: DailyReport = llm.generate_structured(prompt, DailyReport)

    # 6) 渲染与落盘
    template_dir = str(Path(__file__).with_name("render").joinpath("templates"))
    md = render_markdown(template_dir, "daily_report.md.j2", report.model_dump())

    out_dir = Path(settings.outputs_dir)
    out_md = out_dir / f"{today}_daily.md"
    out_sources = out_dir / f"{today}_sources.md"

    dedup_sources = sorted(set(report.sources))
    sources_md = "\n".join([f"- {u}" for u in dedup_sources]) + "\n"

    write_text(str(out_md), md)
    write_text(str(out_sources), sources_md)

    return str(out_md)
