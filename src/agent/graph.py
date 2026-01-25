from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict

from agent.config import get_settings
from agent.orchestrator import run_daily


@dataclass
class ReportGraph:
    outputs_dir: str

    def invoke(self, state: Dict[str, Any]) -> Dict[str, str]:
        """
        Minimal graph interface used by run.py.

        If a portfolio path is provided, run the daily pipeline; otherwise, return
        a lightweight markdown report so local runs don't fail without data files.
        """
        portfolio_path = state.get("portfolio_path")
        if portfolio_path:
            settings = get_settings()
            settings.outputs_dir = self.outputs_dir
            report_path = run_daily(portfolio_path, settings=settings)
            report_md = Path(report_path).read_text(encoding="utf-8")
        else:
            ticker = state.get("ticker", "UNKNOWN")
            start_date = state.get("start_date", "N/A")
            filing_form = state.get("filing_form", "N/A")
            report_md = (
                f"# Daily Report ({date.today().isoformat()})\n\n"
                f"- Ticker: {ticker}\n"
                f"- Start date: {start_date}\n"
                f"- Filing form: {filing_form}\n"
            )

        return {"report_md": report_md}


def build_graph() -> ReportGraph:
    settings = get_settings()
    return ReportGraph(outputs_dir=settings.outputs_dir)
