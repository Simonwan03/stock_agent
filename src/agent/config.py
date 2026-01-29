from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py<3.11
    tomllib = None


@dataclass(frozen=True)
class Settings:
    """Configuration for the multi-agent pipeline.

    We keep this intentionally lightweight so it is easy to extend later.
    """

    outputs_dir: Path
    default_ticker: str
    report_dir: Path
    llm: "LLMSettings"


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    enabled: bool
    timeout: float


def _env_path(key: str, default: Path) -> Path:
    raw = os.getenv(key, "").strip()
    return Path(raw) if raw else default


def _load_toml(path: Path) -> Dict[str, Any]:
    if not path.exists() or tomllib is None:
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def get_settings() -> Settings:
    """Load settings from environment variables with sensible defaults."""

    root = Path(__file__).resolve().parents[2]
    toml_cfg = _load_toml(root / "config" / "config.toml")

    toml_outputs_raw = toml_cfg.get("outputs", {}).get("dir", "") if isinstance(toml_cfg, dict) else ""
    toml_outputs = Path(toml_outputs_raw) if toml_outputs_raw else None
    default_outputs = Path("src/tools/outputs")
    outputs_dir = _env_path("STOCK_AGENT_OUTPUTS_DIR", toml_outputs or default_outputs)
    if not outputs_dir.exists() and default_outputs.exists():
        outputs_dir = default_outputs
    report_dir = _env_path("STOCK_AGENT_REPORT_DIR", Path("out"))
    default_ticker = os.getenv("STOCK_AGENT_TICKER", "AAPL").strip() or "AAPL"

    llm_cfg = toml_cfg.get("llm", {}) if isinstance(toml_cfg, dict) else {}
    base_url = os.getenv("STOCK_AGENT_LLM_BASE_URL", llm_cfg.get("base_url", "")).strip()
    api_key = os.getenv("STOCK_AGENT_LLM_API_KEY", llm_cfg.get("api_key", "")).strip()
    model = os.getenv("STOCK_AGENT_LLM_MODEL", llm_cfg.get("model", "deepseek-chat")).strip()
    timeout_raw = os.getenv("STOCK_AGENT_LLM_TIMEOUT", "").strip()
    if timeout_raw:
        try:
            timeout = float(timeout_raw)
        except ValueError:
            timeout = 30.0
    else:
        timeout = float(llm_cfg.get("timeout", 30.0)) if isinstance(llm_cfg, dict) else 30.0
    enabled_env = os.getenv("STOCK_AGENT_LLM_ENABLED", "").strip().lower()
    enabled = enabled_env in {"1", "true", "yes"} if enabled_env else bool(base_url and api_key and model)

    llm = LLMSettings(base_url=base_url, api_key=api_key, model=model, enabled=enabled, timeout=timeout)
    return Settings(outputs_dir=outputs_dir, default_ticker=default_ticker, report_dir=report_dir, llm=llm)
