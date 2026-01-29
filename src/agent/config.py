from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Configuration for the multi-agent pipeline.

    We keep this intentionally lightweight so it is easy to extend later.
    """

    outputs_dir: Path
    default_ticker: str
    report_dir: Path


def _env_path(key: str, default: Path) -> Path:
    raw = os.getenv(key, "").strip()
    return Path(raw) if raw else default


def get_settings() -> Settings:
    """Load settings from environment variables with sensible defaults."""

    outputs_dir = _env_path("STOCK_AGENT_OUTPUTS_DIR", Path("src/tools/outputs"))
    report_dir = _env_path("STOCK_AGENT_REPORT_DIR", Path("out"))
    default_ticker = os.getenv("STOCK_AGENT_TICKER", "AAPL").strip() or "AAPL"
    return Settings(outputs_dir=outputs_dir, default_ticker=default_ticker, report_dir=report_dir)
