from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_portfolio(path: str) -> Dict[str, Any]:
    portfolio_path = Path(path)
    if not portfolio_path.exists():
        return {"holdings": [], "benchmarks": []}

    raw = portfolio_path.read_text(encoding="utf-8").strip()
    if not raw:
        return {"holdings": [], "benchmarks": []}

    payload = json.loads(raw)
    payload.setdefault("holdings", [])
    payload.setdefault("benchmarks", [])
    return payload
