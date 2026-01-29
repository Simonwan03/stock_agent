from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .schema import build_module_output, utc_now_iso


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_patterns(patterns: Iterable[str], ticker: str) -> List[str]:
    """Allow patterns to reference ticker (upper/lower)."""

    return [
        p.format(ticker=ticker, ticker_lower=ticker.lower(), ticker_upper=ticker.upper())
        for p in patterns
    ]


def _find_latest_file(outputs_dir: Path, patterns: Sequence[str]) -> Optional[Path]:
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(outputs_dir.glob(pattern))
    matches = [m for m in matches if m.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


@dataclass(frozen=True)
class ModuleAgent:
    name: str
    patterns: Sequence[str]

    def run(self, ticker: str, outputs_dir: Path) -> Dict[str, Any]:
        # Locate the newest JSON output for this module.
        path = _find_latest_file(outputs_dir, _format_patterns(self.patterns, ticker))
        if not path:
            return build_module_output(
                module=self.name,
                summary="No source file found.",
                data={"status": "missing"},
                source_files=[],
            )
        # Summarize the raw payload into the shared schema format.
        payload = _load_json(path)
        summary, data = self.summarize(payload)
        return build_module_output(
            module=self.name,
            summary=summary,
            data=data,
            source_files=[str(path)],
        )

    def summarize(self, payload: Any) -> tuple[str, Dict[str, Any]]:
        raise NotImplementedError


class MarketDataAgent(ModuleAgent):
    def summarize(self, payload: Any) -> tuple[str, Dict[str, Any]]:
        prices = payload.get("prices") if isinstance(payload, dict) else None
        if not isinstance(prices, list) or not prices:
            return "Market data present but no prices found.", {"status": "empty"}
        latest = prices[-1]
        prev = prices[-2] if len(prices) > 1 else None
        latest_close = latest.get("close") if isinstance(latest, dict) else None
        prev_close = prev.get("close") if isinstance(prev, dict) else None
        pct_change = None
        if latest_close is not None and prev_close:
            pct_change = (latest_close / prev_close) - 1.0
        summary = f"Latest close {latest_close}, change {pct_change:.2%}." if pct_change is not None else "Latest close available."
        return summary, {
            "latest_close": latest_close,
            "previous_close": prev_close,
            "daily_change_pct": pct_change,
            "asof_date": latest.get("date") if isinstance(latest, dict) else None,
        }


class FinancialsAgent(ModuleAgent):
    def summarize(self, payload: Any) -> tuple[str, Dict[str, Any]]:
        if not isinstance(payload, dict):
            return "Financials payload not recognized.", {"status": "invalid"}
        key_metrics = payload.get("key_metrics", {})
        revenue = key_metrics.get("revenue", {}).get("latest") if isinstance(key_metrics, dict) else None
        net_margin = key_metrics.get("net_margin", {}).get("latest") if isinstance(key_metrics, dict) else None
        summary = "Financial snapshot loaded."
        if revenue is not None:
            summary = f"Latest revenue {revenue:,}."
        return summary, {
            "asof": payload.get("asof"),
            "revenue_latest": revenue,
            "net_margin_latest": net_margin,
            "key_metrics": key_metrics,
        }


class IndicatorsAgent(ModuleAgent):
    def summarize(self, payload: Any) -> tuple[str, Dict[str, Any]]:
        if not isinstance(payload, dict):
            return "Indicators payload not recognized.", {"status": "invalid"}
        performance = payload.get("performance", {})
        ann_return = performance.get("ann_return") if isinstance(performance, dict) else None
        ann_vol = performance.get("ann_vol") if isinstance(performance, dict) else None
        summary = "Indicator snapshot loaded."
        if ann_return is not None:
            summary = f"Annualized return {ann_return:.2%}."
        return summary, {
            "performance": performance,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
        }


class NewsAgent(ModuleAgent):
    def summarize(self, payload: Any) -> tuple[str, Dict[str, Any]]:
        if not isinstance(payload, list):
            return "News payload not recognized.", {"status": "invalid"}
        sources = [item.get("source") for item in payload if isinstance(item, dict)]
        source_counts: Dict[str, int] = {}
        for src in sources:
            if not src:
                continue
            source_counts[src] = source_counts.get(src, 0) + 1
        top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        summary = f"{len(payload)} news items collected."
        return summary, {
            "count": len(payload),
            "top_sources": top_sources,
        }


class PortfolioAgent(ModuleAgent):
    def summarize(self, payload: Any) -> tuple[str, Dict[str, Any]]:
        if not isinstance(payload, dict):
            return "Portfolio payload not recognized.", {"status": "invalid"}
        holdings = payload.get("holdings", [])
        benchmarks = payload.get("benchmarks", [])
        summary = f"Portfolio loaded with {len(holdings)} holdings."
        return summary, {
            "holdings_count": len(holdings) if isinstance(holdings, list) else None,
            "benchmarks": benchmarks,
        }


@dataclass(frozen=True)
class AdvisorAgent:
    """Final agent that converts module outputs into actionable guidance."""

    name: str = "advisor"

    def run(self, ticker: str, modules: List[Dict[str, Any]]) -> Dict[str, Any]:
        module_map = {m.get("module"): m for m in modules}
        market = module_map.get("market_data", {}).get("data", {})
        indicators = module_map.get("indicators", {}).get("data", {})
        financials = module_map.get("financials", {}).get("data", {})
        news = module_map.get("news", {}).get("data", {})

        signals: List[str] = []
        risks: List[str] = []
        actions: List[str] = []

        daily_change = market.get("daily_change_pct")
        if daily_change is not None:
            if daily_change < -0.02:
                signals.append("近期出现明显下跌，短线情绪偏弱。")
                actions.append("关注是否有利空事件或市场整体回撤。")
            elif daily_change > 0.02:
                signals.append("近期上涨动能较强。")

        ann_vol = indicators.get("annualized_volatility")
        if ann_vol is not None and ann_vol > 0.4:
            risks.append("历史波动率偏高，短期风险较大。")
            actions.append("考虑控制仓位或设置止损。")

        net_margin = financials.get("net_margin_latest")
        if net_margin is not None and net_margin < 0.1:
            risks.append("净利率偏低，盈利质量需关注。")

        news_count = news.get("count")
        if isinstance(news_count, int) and news_count > 20:
            signals.append("近期新闻密集，建议关注情绪变化。")

        summary = "基于现有模块数据生成初步投资建议。"
        return {
            "agent": self.name,
            "ticker": ticker,
            "asof_utc": utc_now_iso(),
            "summary": summary,
            "signals": signals,
            "risk_notes": risks,
            "action_items": actions,
            "confidence": "low",
        }
