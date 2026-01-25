from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, Type

from llm.schemas import DailyReport


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def generate_structured(self, prompt: str, schema: Type[DailyReport]) -> DailyReport:
        """
        Use an OpenAI-compatible Chat Completions API if configured.
        """
        if not self.api_key:
            return self._fallback_report(schema)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a financial research assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            return self._parse_report(content, schema)
        except Exception:
            return self._fallback_report(schema)

    def _parse_report(self, content: str, schema: Type[DailyReport]) -> DailyReport:
        try:
            parsed: Dict[str, Any] = json.loads(content)
        except json.JSONDecodeError:
            return self._fallback_report(schema)
        return schema(**parsed)

    def _fallback_report(self, schema: Type[DailyReport]) -> DailyReport:
        return schema(
            title="Daily Portfolio Report",
            summary="Automated report generated without external LLM.",
            portfolio_guidance=[
                "Connect an LLM provider to generate a full report."
            ],
            watchlist=[],
            sources=[],
        )
