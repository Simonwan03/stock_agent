# 目标：把“模型、数据回溯窗口、输出目录”等都集中管理，
# CLI 和 pipeline 都只从这里读配置。
# config.py              # agent 配置 --- IGNORE ---
from pydantic import BaseModel
import os

class Settings(BaseModel):
    # ========== LLM 配置 ==========
    # 兼容 OpenAI / DeepSeek 等“OpenAI-compatible”接口
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # ========== 数据窗口 ==========
    news_lookback_hours: int = int(os.getenv("NEWS_LOOKBACK_HOURS", "72"))
    price_lookback_days: int = int(os.getenv("PRICE_LOOKBACK_DAYS", "252"))

    # ========== 输出 ==========
    outputs_dir: str = os.getenv("OUTPUTS_DIR", "outputs")

def get_settings() -> Settings:
    """
    统一从环境变量读取配置。
    """
    return Settings()
