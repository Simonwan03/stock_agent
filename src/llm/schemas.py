from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class DailyReport(BaseModel):
    title: str = Field(default="Daily Portfolio Report")
    summary: str = Field(default="No summary available.")
    portfolio_guidance: List[str] = Field(default_factory=list)
    watchlist: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
