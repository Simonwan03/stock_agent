from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .agents import (
    AdvisorAgent,
    FinancialsAgent,
    IndicatorsAgent,
    MarketDataAgent,
    ModuleAgent,
    NewsAgent,
    PortfolioAgent,
)
from .schema import build_pipeline_output
from ..llm_client import LLMClient


@dataclass(frozen=True)
class MultiAgentOrchestrator:
    """Coordinate module agents and produce a unified pipeline output."""

    outputs_dir: Path
    llm: Optional[LLMClient] = None

    def _agents(self) -> List[ModuleAgent]:
        # Each module has a list of file patterns; the newest match is used.
        # This allows you to drop new tool outputs into outputs_dir without
        # changing any code, as long as the filename matches a pattern.
        return [
            MarketDataAgent(
                name="market_data",
                patterns=[
                    "{ticker_upper}.json",
                    "{ticker_lower}.json",
                    "{ticker_upper}_*.json",
                    "{ticker_lower}_*.json",
                ],
                llm=self.llm,
            ),
            FinancialsAgent(
                name="financials",
                patterns=[
                    "*{ticker_upper}*compact*.json",
                    "*{ticker_lower}*compact*.json",
                    "*openbb_financials*{ticker_upper}*.json",
                ],
                llm=self.llm,
            ),
            IndicatorsAgent(
                name="indicators",
                patterns=["*{ticker_upper}*indicators*.json", "*{ticker_lower}*indicators*.json"],
                llm=self.llm,
            ),
            NewsAgent(
                name="news",
                patterns=[
                    "{ticker_upper}_news*.json",
                    "{ticker_lower}_news*.json",
                    "*news*.json",
                ],
                llm=self.llm,
            ),
            PortfolioAgent(
                name="portfolio",
                patterns=["portfolio.json", "*portfolio*.json"],
                llm=self.llm,
            ),
        ]

    def run(self, ticker: str, progress: Optional[callable] = None) -> Dict[str, object]:
        def _p(msg: str) -> None:
            if progress:
                progress(msg)
        # 1) Run every module agent to produce schema-aligned JSON.
        _p(f"[pipeline] start for {ticker}")
        modules = []
        for agent in self._agents():
            modules.append(agent.run(ticker=ticker, outputs_dir=self.outputs_dir, progress=progress))

        # 2) Feed the merged module outputs into the final advisor agent.
        _p("[advisor] running final advice")
        advisor = AdvisorAgent(llm=self.llm)
        advice = advisor.run(ticker=ticker, modules=modules)
        _p("[pipeline] done")
        # 3) Wrap everything into a single pipeline payload.
        return build_pipeline_output(ticker=ticker, modules=modules, advice=advice)
