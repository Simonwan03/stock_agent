#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch recent financial statements (income/balance/cash) via OpenBB and save to JSON.

Usage examples:
  python -m financial_statement --ticker AAPL --period quarter --limit 4 --provider yfinance
  python -m financial_statement --ticker MSFT --period annual --limit 3 --statements income,cash --stdout

Notes:
- OpenBB commands return an OBBject that supports to_dict()/to_df() formatters.  :contentReference[oaicite:1]{index=1}
- Providers may require API keys depending on your choice.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_OUTPUT_DIR = Path("outputs")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_safe(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    # pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return _json_safe(obj.model_dump())
        except Exception:
            pass
    # pydantic v1
    if hasattr(obj, "dict"):
        try:
            return _json_safe(obj.dict())
        except Exception:
            pass
    # datetime/date-like
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    try:
        return str(obj)
    except Exception:
        return None


def _obbject_to_payload(obbject: Any) -> Dict[str, Any]:
    """
    Convert OpenBB OBBject -> dict.
    OBBject usually contains: results/provider/warnings/chart/extra and supports to_dict(). :contentReference[oaicite:2]{index=2}
    """
    # Prefer the built-in formatter if present
    if hasattr(obbject, "to_dict"):
        try:
            return _json_safe(obbject.to_dict())  # includes results + metadata fields
        except Exception:
            pass

    out: Dict[str, Any] = {}
    for key in ("results", "provider", "warnings", "chart", "extra", "id"):
        if hasattr(obbject, key):
            try:
                out[key] = _json_safe(getattr(obbject, key))
            except Exception:
                out[key] = None
    return out


def _filter_fields(records: Any, keep: Optional[Sequence[str]]) -> Any:
    """
    If keep is provided, reduce each record dict to those keys (when possible).
    Works on list[dict] or dict-like structures; otherwise returns as-is.
    """
    if not keep:
        return records
    keep_set = set(keep)

    if isinstance(records, list):
        new_list = []
        for r in records:
            if isinstance(r, dict):
                new_list.append({k: r.get(k) for k in keep if k in r})
            else:
                new_list.append(r)
        return new_list

    if isinstance(records, dict):
        # common case: {"results": [...], ...}
        if "results" in records and isinstance(records["results"], list):
            copied = dict(records)
            copied["results"] = _filter_fields(records["results"], keep)
            return copied
        return {k: records.get(k) for k in keep if k in records}

    return records


def fetch_financials(
    ticker: str,
    period: str,
    limit: int,
    provider: str,
    statements: Sequence[str],
    fields: Optional[Sequence[str]] = None,
    include_meta: bool = True,
) -> Dict[str, Any]:
    try:
        from openbb import obb  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenBB 未安装或不可用。请先安装：pip install openbb"
        ) from e

    ticker = ticker.upper().strip()
    period = period.lower().strip()
    provider = provider.lower().strip()

    if period not in ("annual", "quarter"):
        raise ValueError("period 必须是 annual 或 quarter")
    if limit <= 0:
        raise ValueError("limit 必须 > 0")

    # map statement -> callable
    fetchers = {
        "income": lambda: obb.equity.fundamental.income(symbol=ticker, period=period, limit=limit, provider=provider),
        "balance": lambda: obb.equity.fundamental.balance(symbol=ticker, period=period, limit=limit, provider=provider),
        "cash": lambda: obb.equity.fundamental.cash(symbol=ticker, period=period, limit=limit, provider=provider),
        # 也可以用 reported_financials（字段更“原始/不定长”），这里先不默认开启
        # "reported": lambda: obb.equity.fundamental.reported_financials(
        #     symbol=ticker, period=period, statement_type="income", limit=limit, provider=provider
        # ),
    }

    unknown = [s for s in statements if s not in fetchers]
    if unknown:
        raise ValueError(f"statements 不支持：{unknown}；可选: income,balance,cash")

    out: Dict[str, Any] = {
        "ticker": ticker,
        "period": period,
        "limit": limit,
        "provider": provider,
        "retrieved_at": _utc_now_iso(),
        "statements": {},
    }

    for s in statements:
        obj = fetchers[s]()
        payload = _obbject_to_payload(obj)

        # 默认只保留 results（更轻量）；如果你想调试 provider/warnings/extra，再开 include_meta
        if not include_meta and isinstance(payload, dict) and "results" in payload:
            payload = payload["results"]

        payload = _filter_fields(payload, fields)
        out["statements"][s] = payload

    return out


def _default_output_path(ticker: str, period: str, provider: str, statements: Sequence[str]) -> Path:
    stm = "-".join(statements)
    return DEFAULT_OUTPUT_DIR / f"openbb_financials_{ticker.upper()}_{period}_{provider}_{stm}.json"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch recent financial statements via OpenBB (income/balance/cash).")
    p.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    p.add_argument("--period", default="quarter", choices=["annual", "quarter"], help="annual or quarter")
    p.add_argument("--limit", type=int, default=4, help="How many recent periods to fetch")
    p.add_argument("--provider", default="yfinance", help="Provider, e.g. yfinance, fmp, intrinio (depends on setup)")
    p.add_argument(
        "--statements",
        default="income,balance,cash",
        help="Comma-separated: income,balance,cash",
    )
    p.add_argument(
        "--fields",
        default="",
        help="Optional comma-separated field whitelist to reduce JSON size, e.g. date,revenue,net_income,total_assets",
    )
    p.add_argument("--no-meta", action="store_true", help="Only keep results (drop provider/warnings/extra/etc.)")
    p.add_argument("--out", default="", help="Output JSON path (default: outputs/openbb_financials_*.json)")
    p.add_argument("--stdout", action="store_true", help="Print JSON to stdout instead of writing a file")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    statements = [s.strip().lower() for s in args.statements.split(",") if s.strip()]
    fields = [s.strip() for s in args.fields.split(",") if s.strip()] or None

    payload = fetch_financials(
        ticker=args.ticker,
        period=args.period,
        limit=args.limit,
        provider=args.provider,
        statements=statements,
        fields=fields,
        include_meta=not args.no_meta,
    )

    if args.stdout:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    out_path = Path(args.out) if args.out else _default_output_path(args.ticker, args.period, args.provider, statements)
    write_json(out_path, payload)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
