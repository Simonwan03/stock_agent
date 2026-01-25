# 目标：把“模型、数据回溯窗口、输出目录”等都集中管理，
# CLI 和 pipeline 都只从这里读配置。
# config.py              # agent 配置 --- IGNORE ---
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel

class Settings(BaseModel):
    # ========== LLM 配置 ==========
    # 兼容 OpenAI / DeepSeek 等“OpenAI-compatible”接口
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # ========== 数据窗口 ==========
    news_lookback_hours: int = 72
    price_lookback_days: int = 252

    # ========== 数据源 ==========
    market_data_provider: str = "yfinance"

    # ========== 输出 ==========
    outputs_dir: str = "outputs"

    # ========== 组合 ==========
    portfolio_path: str = "data/portfolio.json"


def _load_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return data


def _get_nested(data: Dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = data.get(section, {}).get(key, default)
    return value


def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    从配置文件读取设置，环境变量可覆盖。
    """
    config_file = Path(config_path) if config_path else Path("config/config.toml")
    data = _load_config_file(config_file)

    settings = Settings(
        llm_base_url=_get_nested(data, "llm", "base_url", Settings().llm_base_url),
        llm_api_key=_get_nested(data, "llm", "api_key", Settings().llm_api_key),
        llm_model=_get_nested(data, "llm", "model", Settings().llm_model),
        news_lookback_hours=int(
            _get_nested(data, "news", "lookback_hours", Settings().news_lookback_hours)
        ),
        price_lookback_days=int(
            _get_nested(data, "market_data", "lookback_days", Settings().price_lookback_days)
        ),
        market_data_provider=_get_nested(
            data, "market_data", "provider", Settings().market_data_provider
        ),
        outputs_dir=_get_nested(data, "outputs", "dir", Settings().outputs_dir),
        portfolio_path=_get_nested(data, "portfolio", "path", Settings().portfolio_path),
    )

    settings.llm_base_url = os.getenv("LLM_BASE_URL", settings.llm_base_url)
    settings.llm_api_key = os.getenv("LLM_API_KEY", settings.llm_api_key)
    settings.llm_model = os.getenv("LLM_MODEL", settings.llm_model)
    settings.market_data_provider = os.getenv(
        "MARKET_DATA_PROVIDER", settings.market_data_provider
    )
    settings.news_lookback_hours = int(
        os.getenv("NEWS_LOOKBACK_HOURS", str(settings.news_lookback_hours))
    )
    settings.price_lookback_days = int(
        os.getenv("PRICE_LOOKBACK_DAYS", str(settings.price_lookback_days))
    )
    settings.outputs_dir = os.getenv("OUTPUTS_DIR", settings.outputs_dir)
    settings.portfolio_path = os.getenv("PORTFOLIO_PATH", settings.portfolio_path)
    return settings
