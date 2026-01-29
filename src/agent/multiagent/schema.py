from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"
ADVICE_SCHEMA_VERSION = "1.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ModuleOutput:
    """Typed representation of a module agent output."""

    module: str
    schema_version: str
    asof_utc: str
    source_files: List[str]
    summary: str
    data: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "schema_version": self.schema_version,
            "asof_utc": self.asof_utc,
            "source_files": self.source_files,
            "summary": self.summary,
            "data": self.data,
        }


def build_module_output(
    module: str,
    summary: str,
    data: Dict[str, Any],
    source_files: Optional[List[str]] = None,
    asof_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Factory for module outputs that follow the shared schema."""

    return ModuleOutput(
        module=module,
        schema_version=SCHEMA_VERSION,
        asof_utc=asof_utc or utc_now_iso(),
        source_files=source_files or [],
        summary=summary,
        data=data,
    ).as_dict()


def build_pipeline_output(
    ticker: str,
    modules: List[Dict[str, Any]],
    advice: Dict[str, Any],
    asof_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Overall pipeline schema wrapper."""

    return {
        "schema_version": SCHEMA_VERSION,
        "ticker": ticker,
        "asof_utc": asof_utc or utc_now_iso(),
        "modules": modules,
        "advice": advice,
    }


def build_advice_output(
    ticker: str,
    summary: str,
    signals: List[str],
    risk_notes: List[str],
    action_items: List[str],
    stance: str,
    confidence: str,
    time_horizon: str,
    agent: str = "advisor",
    asof_utc: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": ADVICE_SCHEMA_VERSION,
        "agent": agent,
        "ticker": ticker,
        "asof_utc": asof_utc or utc_now_iso(),
        "summary": summary,
        "signals": signals,
        "risk_notes": risk_notes,
        "action_items": action_items,
        "stance": stance,
        "confidence": confidence,
        "time_horizon": time_horizon,
    }
