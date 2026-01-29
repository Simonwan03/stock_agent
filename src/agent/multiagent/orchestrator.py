from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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


@dataclass(frozen=True)
class MultiAgentOrchestrator:
    """Coordinate module agents and produce a unified pipeline output."""

    outputs_dir: Path

    def _agents(self) -> List[ModuleAgent]:
        # Each module has a list of file patterns; the newest match is used.
        # This allows you to drop new tool outputs into outputs_dir without
        # changing any code, as long as the filename matches a pattern.
        return [
            MarketDataAgent(
                name="market_data",
                patterns=[
                    "{ticker_upper}_*.json",
                    "{ticker_lower}_*.json",
                ],
            ),
            FinancialsAgent(
                name="financials",
                patterns=[
                    "*{ticker_upper}*compact*.json",
                    "*{ticker_lower}*compact*.json",
                    "*openbb_financials*{ticker_upper}*.json",
                ],
            ),
            IndicatorsAgent(
                name="indicators",
                patterns=["*{ticker_upper}*indicators*.json", "*{ticker_lower}*indicators*.json"],
            ),
            NewsAgent(
                name="news",
                patterns=["*{ticker_upper}.json", "*{ticker_lower}.json", "*news*.json"],
            ),
            PortfolioAgent(
                name="portfolio",
                patterns=["portfolio.json", "*portfolio*.json"],
            ),
        ]

    def run(self, ticker: str) -> Dict[str, object]:
        # 1) Run every module agent to produce schema-aligned JSON.
        modules = []
        for agent in self._agents():
            modules.append(agent.run(ticker=ticker, outputs_dir=self.outputs_dir))

        # 2) Feed the merged module outputs into the final advisor agent.
        advisor = AdvisorAgent()
        advice = advisor.run(ticker=ticker, modules=modules)
        # 3) Wrap everything into a single pipeline payload.
        return build_pipeline_output(ticker=ticker, modules=modules, advice=advice)
