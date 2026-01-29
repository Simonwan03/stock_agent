#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# python -m compact_financials --in src/tools/outputs/openbb_financials_AAPL_quarter_yfinance_income-balance-cash.json --out outputs/aapl_compact.json



# -----------------------------
# JSON loading / NaN handling
# -----------------------------
def load_json_allow_nan(text: str) -> Any:
    """
    Your sample includes NaN (not valid strict JSON).
    Python json.loads can accept it with parse_constant.
    We'll convert NaN/Infinity to None to keep the output clean.
    """
    def _parse_constant(x: str) -> Any:
        # x could be 'NaN', 'Infinity', '-Infinity'
        return None

    return json.loads(text, parse_constant=_parse_constant)


def json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return True
    return False


# -----------------------------
# Pivot: arrays -> records
# -----------------------------
def pivot_statement(statement: Dict[str, Any], date_key: str = "period_ending") -> List[Dict[str, Any]]:
    """
    Input: {"period_ending":[...], "total_revenue":[...], ...}
    Output: [{"period_ending": "...", "total_revenue": ...}, ...]
    """
    if not isinstance(statement, dict):
        return []

    dates = statement.get(date_key)
    if not isinstance(dates, list) or not dates:
        return []

    n = len(dates)
    records: List[Dict[str, Any]] = [{"period_ending": dates[i]} for i in range(n)]

    for k, v in statement.items():
        if k == date_key:
            continue
        if isinstance(v, list) and len(v) == n:
            for i in range(n):
                records[i][k] = None if _is_missing(v[i]) else v[i]
        else:
            # ignore malformed columns
            pass

    return records


def keep_fields(records: List[Dict[str, Any]], whitelist: Sequence[str]) -> List[Dict[str, Any]]:
    wl = set(whitelist)
    out = []
    for r in records:
        nr = {"period_ending": r.get("period_ending")}
        for k in whitelist:
            if k == "period_ending":
                continue
            if k in r:
                nr[k] = r.get(k)
        out.append(nr)
    return out


# -----------------------------
# Metric helpers
# -----------------------------
def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if b == 0:
        return None
    return a / b


def _qoq(latest: Optional[float], prev: Optional[float]) -> Optional[float]:
    if latest is None or prev is None or prev == 0:
        return None
    return (latest / prev) - 1.0


def _sum(vals: Sequence[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return float(sum(xs)) if xs else None


def _latest_and_prev(records: List[Dict[str, Any]], key: str) -> Tuple[Optional[float], Optional[float]]:
    if len(records) < 2:
        return None, None
    return records[0].get(key), records[1].get(key)


# -----------------------------
# Build compact payload
# -----------------------------
INCOME_CORE = [
    "total_revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "diluted_earnings_per_share",
    "research_and_development_expense",
    "selling_general_and_admin_expense",
]

CASH_CORE = [
    "operating_cash_flow",
    "capital_expenditure",
    "free_cash_flow",
    "repurchase_of_capital_stock",
    "cash_dividends_paid",
]

BALANCE_CORE = [
    "cash_and_cash_equivalents",
    "short_term_investments",
    "total_assets",
    "total_liabilities_net_minority_interest",
    "total_debt",
    "net_debt",
    "accounts_receivable",
    "accounts_payable",
    "working_capital",
]

def build_compact(payload: Dict[str, Any]) -> Dict[str, Any]:
    ticker = str(payload.get("ticker", "")).upper()
    period = payload.get("period", "")
    provider = payload.get("provider", "")
    retrieved_at = payload.get("retrieved_at", "")

    statements = payload.get("statements", {}) or {}
    income_raw = statements.get("income") or {}
    cash_raw = statements.get("cash") or {}
    balance_raw = statements.get("balance") or {}

    income_rec = pivot_statement(income_raw)
    cash_rec = pivot_statement(cash_raw)
    balance_rec = pivot_statement(balance_raw)

    # Ensure same ordering (your data is latest -> older already).
    periods = [r.get("period_ending") for r in income_rec] or \
              [r.get("period_ending") for r in cash_rec] or \
              [r.get("period_ending") for r in balance_rec]

    asof = periods[0] if periods else None

    # Keep only core fields (raw_core: for traceability, but much smaller)
    raw_core = {
        "income": keep_fields(income_rec, ["period_ending", *INCOME_CORE]),
        "cash": keep_fields(cash_rec, ["period_ending", *CASH_CORE]),
        "balance": keep_fields(balance_rec, ["period_ending", *BALANCE_CORE]),
    }

    # Latest values
    rev_latest, rev_prev = _latest_and_prev(income_rec, "total_revenue")
    eps_latest, eps_prev = _latest_and_prev(income_rec, "diluted_earnings_per_share")
    gp_latest = income_rec[0].get("gross_profit") if income_rec else None
    op_latest = income_rec[0].get("operating_income") if income_rec else None
    ni_latest = income_rec[0].get("net_income") if income_rec else None

    fcf_latest, fcf_prev = _latest_and_prev(cash_rec, "free_cash_flow")
    ocf_latest = cash_rec[0].get("operating_cash_flow") if cash_rec else None
    capex_latest = cash_rec[0].get("capital_expenditure") if cash_rec else None

    net_debt_latest, net_debt_prev = _latest_and_prev(balance_rec, "net_debt")
    ar_latest, ar_prev = _latest_and_prev(balance_rec, "accounts_receivable")

    # Margins (latest quarter)
    gross_margin = _safe_div(gp_latest, rev_latest)
    op_margin = _safe_div(op_latest, rev_latest)
    net_margin = _safe_div(ni_latest, rev_latest)
    fcf_margin = _safe_div(fcf_latest, rev_latest)

    # TTM (assume 4 quarters available)
    rev_ttm = _sum([r.get("total_revenue") for r in income_rec[:4]])
    ni_ttm = _sum([r.get("net_income") for r in income_rec[:4]])
    fcf_ttm = _sum([r.get("free_cash_flow") for r in cash_rec[:4]])

    # Shareholder return intensity (TTM)
    # Values are negative in your sample for buybacks/dividends -> flip sign to represent cash outflow positive.
    buyback_ttm_raw = _sum([r.get("repurchase_of_capital_stock") for r in cash_rec[:4]])
    div_ttm_raw = _sum([r.get("cash_dividends_paid") for r in cash_rec[:4]])
    buyback_ttm = (-buyback_ttm_raw) if buyback_ttm_raw is not None else None
    dividends_ttm = (-div_ttm_raw) if div_ttm_raw is not None else None
    sh_return_ttm = None
    if buyback_ttm is not None or dividends_ttm is not None:
        sh_return_ttm = (buyback_ttm or 0.0) + (dividends_ttm or 0.0)

    sh_return_vs_fcf = _safe_div(sh_return_ttm, fcf_ttm)

    # Watch items (simple heuristics you can tweak)
    watch_items: List[Dict[str, Any]] = []
    if ar_latest is not None and ar_prev is not None and ar_prev != 0:
        ar_qoq = (ar_latest / ar_prev) - 1.0
        if ar_qoq > 0.20:
            watch_items.append({
                "item": "Accounts receivable spike",
                "latest": ar_latest,
                "prev": ar_prev,
                "qoq": ar_qoq,
                "hint": "可能是季节性出货/回款节奏变化；建议结合管理层 commentary 或 DSO 指标确认。"
            })

    if sh_return_vs_fcf is not None and sh_return_vs_fcf > 1.0:
        watch_items.append({
            "item": "Shareholder returns exceed FCF (TTM)",
            "ratio": sh_return_vs_fcf,
            "hint": "回购+分红超过自由现金流，通常依赖现金储备/负债配合；若 FCF 下滑，回购强度可能下调。"
        })

    out = {
        "ticker": ticker,
        "period": period,
        "provider": provider,
        "retrieved_at": retrieved_at,
        "asof": asof,
        "periods": periods[:4] if periods else [],
        "key_metrics": {
            "revenue": {"latest": rev_latest, "qoq": _qoq(rev_latest, rev_prev), "ttm": rev_ttm},
            "eps": {"latest": eps_latest, "qoq": _qoq(eps_latest, eps_prev)},
            "gross_margin": {"latest": gross_margin},
            "op_margin": {"latest": op_margin},
            "net_margin": {"latest": net_margin},
            "fcf": {"latest": fcf_latest, "qoq": _qoq(fcf_latest, fcf_prev), "ttm": fcf_ttm},
            "fcf_margin": {"latest": fcf_margin},
            "shareholder_return": {
                "buyback_ttm": buyback_ttm,
                "dividends_ttm": dividends_ttm,
                "total_ttm": sh_return_ttm,
                "vs_fcf_ttm": sh_return_vs_fcf,
            },
            "net_debt": {"latest": net_debt_latest, "qoq": _qoq(net_debt_latest, net_debt_prev)},
            "operating_cash_flow": {"latest": ocf_latest},
            "capex": {"latest": capex_latest},
        },
        "watch_items": watch_items,
        "raw_core": raw_core,
    }
    return out


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compress OpenBB financial statements JSON into agent-friendly payload.")
    p.add_argument("--in", dest="in_path", default="", help="Input JSON path (OpenBB output). If omitted, read stdin.")
    p.add_argument("--out", dest="out_path", default="", help="Output JSON path. If omitted, print to stdout.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.in_path:
        text = Path(args.in_path).read_text(encoding="utf-8")
    else:
        import sys
        text = sys.stdin.read()

    raw = load_json_allow_nan(text)
    compact = build_compact(raw)

    if args.out_path:
        outp = Path(args.out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json_dump(compact), encoding="utf-8")
        print(f"Saved: {outp}")
    else:
        print(json_dump(compact))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
