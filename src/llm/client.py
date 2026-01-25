from __future__ import annotations

import json
from typing import Any, Type

from llm.schemas import DailyReport


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def generate_structured(self, prompt: str, schema: Type[DailyReport]) -> DailyReport:
        """
        Stub implementation: parse the embedded JSON payload if present.
        """
        payload = {}
        if "{{INPUT_JSON}}" not in prompt:
            try:
                payload = json.loads(prompt)
            except json.JSONDecodeError:
                payload = {}

        summary = "Automated report generated without external LLM."
        guidance = []
        if isinstance(payload, dict):
            portfolio = payload.get("portfolio", {})
            holdings = portfolio.get("holdings", [])
            if holdings:
                guidance.append("Review today's key holdings for major price moves.")
            else:
                guidance.append("No holdings found in portfolio data.")

        return schema(
            title="Daily Portfolio Report",
            summary=summary,
            portfolio_guidance=guidance,
            watchlist=[],
            sources=[],
        )
