from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .schema import build_advice_output, build_module_output, utc_now_iso
from ..llm_client import LLMClient, LLMError


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


def _parse_json_response(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _coerce_llm_output(obj: Any) -> tuple[str, Dict[str, Any]]:
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object.")
    summary = obj.get("summary")
    data = obj.get("data")
    if not isinstance(summary, str) or not summary.strip():
        summary = "LLM summary generated."
    if not isinstance(data, dict):
        data = {"raw": data}
    return summary, data


def _clip_payload(module: str, payload: Any) -> Any:
    if module == "news":
        # keep only compact fields for first 25 items
        items = payload if isinstance(payload, list) else (payload.get("items") if isinstance(payload, dict) else None)
        if isinstance(items, list):
            clipped = []
            for item in items[:25]:
                if not isinstance(item, dict):
                    clipped.append(item)
                    continue
                clipped.append(
                    {
                        "title": item.get("title"),
                        "source": item.get("source"),
                        "published_utc": item.get("published_utc") or item.get("published_at"),
                        "url": item.get("url"),
                        "summary": item.get("summary") or item.get("text"),
                    }
                )
            return clipped
        return payload

    if not isinstance(payload, dict):
        return payload
    trimmed = dict(payload)
    if module == "market_data":
        prices = trimmed.get("prices")
        if isinstance(prices, list) and len(prices) > 80:
            trimmed["prices"] = prices[-80:]
        dy = trimmed.get("trailing_dividend_yield_series")
        if isinstance(dy, list) and len(dy) > 80:
            trimmed["trailing_dividend_yield_series"] = dy[-80:]
    return trimmed


def _module_schema_template(module: str) -> str:
    templates = {
        "market_data": {
            "summary": "string",
            "data": {
                "latest_close": "number|null",
                "previous_close": "number|null",
                "daily_change_pct": "number|null",
                "asof_date": "string|null",
                "key_facts": ["string"],
            },
        },
        "financials": {
            "summary": "string",
            "data": {
                "revenue_latest": "number|null",
                "net_margin_latest": "number|null",
                "key_metrics": "object",
                "notes": ["string"],
            },
        },
        "indicators": {
            "summary": "string",
            "data": {
                "annualized_return": "number|null",
                "annualized_volatility": "number|null",
                "technical_signals": ["string"],
            },
        },
        "news": {
            "summary": "string",
            "data": {
                "count": "int",
                "top_sources": ["[source, count]"],
                "themes": ["string"],
                "sentiment": "positive|neutral|negative|mixed",
            },
        },
        "portfolio": {
            "summary": "string",
            "data": {
                "holdings_count": "int",
                "benchmarks": ["string"],
                "notes": ["string"],
            },
        },
    }
    return json.dumps(templates.get(module, {"summary": "string", "data": "object"}), ensure_ascii=False)


def _build_module_prompt(module: str, ticker: str, payload: Any) -> List[Dict[str, str]]:
    payload_for_llm = _clip_payload(module, payload)
    schema_text = _module_schema_template(module)
    return [
        {
            "role": "system",
            "content": "You are a financial analysis assistant. Return ONLY valid JSON.",
        },
        {
            "role": "user",
            "content": (
                f"Module: {module}\n"
                f"Ticker: {ticker}\n"
                "Generate a concise module summary following this JSON schema:\n"
                f"{schema_text}\n"
                "Rules:\n"
                "- Output must be a single JSON object with keys: summary, data.\n"
                "- Use null for unknown values.\n"
                "- No markdown or extra text.\n"
                "Input payload (possibly truncated):\n"
                f"{json.dumps(payload_for_llm, ensure_ascii=False)}"
            ),
        },
    ]


def _llm_summarize_module(
    llm: LLMClient,
    module: str,
    ticker: str,
    payload: Any,
) -> tuple[str, Dict[str, Any]]:
    response = llm.chat(_build_module_prompt(module, ticker, payload), temperature=0.2)
    obj = _parse_json_response(response)
    return _coerce_llm_output(obj)


@dataclass(frozen=True)
class ModuleAgent:
    name: str
    patterns: Sequence[str]
    llm: Optional[LLMClient] = None

    def run(
        self,
        ticker: str,
        outputs_dir: Path,
        progress: Optional[callable] = None,
    ) -> Dict[str, Any]:
        def _p(msg: str) -> None:
            if progress:
                progress(msg)
        # Locate the newest JSON output for this module.
        _p(f"[{self.name}] searching outputs in {outputs_dir} ...")
        path = _find_latest_file(outputs_dir, _format_patterns(self.patterns, ticker))
        if not path:
            _p(f"[{self.name}] no source file found")
            return build_module_output(
                module=self.name,
                summary="No source file found.",
                data={"status": "missing"},
                source_files=[],
            )
        # Summarize the raw payload into the shared schema format.
        _p(f"[{self.name}] loading {path}")
        payload = _load_json(path)
        if self.llm:
            try:
                _p(f"[{self.name}] calling LLM")
                summary, data = _llm_summarize_module(self.llm, self.name, ticker, payload)
            except (LLMError, ValueError, json.JSONDecodeError) as exc:
                _p(f"[{self.name}] LLM failed, falling back: {exc}")
                summary, data = self.summarize(payload)
                data = dict(data)
                data["llm_error"] = str(exc)
        else:
            _p(f"[{self.name}] LLM disabled, using rule-based summary")
            summary, data = self.summarize(payload)
        _p(f"[{self.name}] done")
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
    llm: Optional[LLMClient] = None

    def _advice_schema(self) -> str:
        schema = {
            "summary": "string",
            "signals": ["string"],
            "risk_notes": ["string"],
            "action_items": ["string"],
            "stance": "bullish|neutral|bearish|mixed",
            "confidence": "low|medium|high",
            "time_horizon": "short-term|mid-term|long-term|unknown",
        }
        return json.dumps(schema, ensure_ascii=False)

    def _build_advice_prompt(self, ticker: str, modules: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are an investment analysis assistant. Return ONLY valid JSON.",
            },
            {
                "role": "user",
                "content": (
                    f"Ticker: {ticker}\n"
                    "Generate final investment advice based on module outputs.\n"
                    "Return a JSON object with this schema:\n"
                    f"{self._advice_schema()}\n"
                    "Rules:\n"
                    "- Output must be a single JSON object.\n"
                    "- Use concise, factual language and avoid hallucinating.\n"
                    "- Use 'unknown' or null if data is missing.\n"
                    "- No markdown or extra text.\n"
                    "Module outputs:\n"
                    f"{json.dumps(modules, ensure_ascii=False)}"
                ),
            },
        ]

    def run(self, ticker: str, modules: List[Dict[str, Any]]) -> Dict[str, Any]:
        module_map = {m.get("module"): m for m in modules}
        market = module_map.get("market_data", {}).get("data", {})
        indicators = module_map.get("indicators", {}).get("data", {})
        financials = module_map.get("financials", {}).get("data", {})
        news = module_map.get("news", {}).get("data", {})

        if self.llm:
            try:
                response = self.llm.chat(self._build_advice_prompt(ticker, modules), temperature=0.3)
                obj = _parse_json_response(response)
                summary, data = _coerce_llm_output(obj)
                return build_advice_output(
                    ticker=ticker,
                    summary=summary,
                    signals=list(data.get("signals") or []),
                    risk_notes=list(data.get("risk_notes") or []),
                    action_items=list(data.get("action_items") or []),
                    stance=str(data.get("stance") or "unknown"),
                    confidence=str(data.get("confidence") or "unknown"),
                    time_horizon=str(data.get("time_horizon") or "unknown"),
                    agent=self.name,
                )
            except (LLMError, ValueError, json.JSONDecodeError) as exc:
                fallback = self._rule_based_advice(ticker, market, indicators, financials, news)
                fallback["llm_error"] = str(exc)
                return fallback

        return self._rule_based_advice(ticker, market, indicators, financials, news)

    def _rule_based_advice(
        self,
        ticker: str,
        market: Dict[str, Any],
        indicators: Dict[str, Any],
        financials: Dict[str, Any],
        news: Dict[str, Any],
    ) -> Dict[str, Any]:
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
        return build_advice_output(
            ticker=ticker,
            summary=summary,
            signals=signals,
            risk_notes=risks,
            action_items=actions,
            stance="neutral",
            confidence="low",
            time_horizon="short-term",
            agent=self.name,
            asof_utc=utc_now_iso(),
        )
