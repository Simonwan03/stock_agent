from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class LLMError(RuntimeError):
    pass


def _normalize_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


@dataclass(frozen=True)
class LLMClient:
    base_url: str
    api_key: str
    model: str
    timeout: float = 30.0

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        url = f"{_normalize_base_url(self.base_url)}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            raise LLMError(f"LLM HTTP error {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"LLM connection error: {exc}") from exc

        try:
            parsed = json.loads(body)
            return parsed["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover - defensive
            raise LLMError(f"LLM response parse error: {exc}") from exc


def llm_client_from_settings(settings: Any) -> Optional[LLMClient]:
    if not settings or not getattr(settings, "enabled", False):
        return None
    timeout = float(getattr(settings, "timeout", 30.0))
    return LLMClient(base_url=settings.base_url, api_key=settings.api_key, model=settings.model, timeout=timeout)
