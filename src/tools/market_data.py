import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from openbb import obb

try:
    # 如果环境里有 openbb_core，用它能更精确捕获 OpenBBError
    from openbb_core.app.model.abstract.error import OpenBBError
except Exception:  # pragma: no cover
    OpenBBError = Exception


def _normalize_col(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Pick the first matching column (case-insensitive, normalized)."""
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    norm_map = {_normalize_col(c): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
        norm = _normalize_col(c)
        if norm in norm_map:
            return norm_map[norm]
    return None


def _df_last_n_trading_days(
    df: pd.DataFrame,
    n_days: int,
    date_col_candidates=("date", "datetime", "time"),
) -> pd.DataFrame:
    """Sort by date and keep last n rows (assumed trading days)."""
    if df is None or df.empty:
        return df

    date_col = _pick_col(df, list(date_col_candidates))
    tmp = df.copy()

    if date_col:
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce", utc=True)
        tmp = tmp.dropna(subset=[date_col]).sort_values(date_col)
    else:
        tmp.index = pd.to_datetime(tmp.index, errors="coerce", utc=True)
        tmp = tmp.dropna(axis=0, how="all").sort_index()

    return tmp.tail(n_days)


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            s = x.strip()
            if not s or s.lower() in {"n/a", "na", "nan", "-", "--"}:
                return None
            is_negative = s.startswith("(") and s.endswith(")")
            if is_negative:
                s = s[1:-1].strip()
            s = s.replace(",", "")
            is_percent = s.endswith("%")
            if is_percent:
                s = s[:-1].strip()
            multiplier = 1.0
            if s and s[-1].isalpha():
                suffix = s[-1].upper()
                unit_map = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
                if suffix in unit_map:
                    multiplier = unit_map[suffix]
                    s = s[:-1]
            if not s:
                return None
            val = float(s) * multiplier
            if is_negative:
                val = -val
            if is_percent:
                val /= 100.0
            return val
        return float(x)
    except Exception:
        return None


def _ensure_providers(p: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(p, str):
        return [p]
    return [x for x in p if x]


def build_equity_json(
    symbol: str,
    n_days: int = 15,
    price_provider: str = "yfinance",
    metrics_provider: Union[str, Sequence[str]] = ("finviz", "yfinance"),
    div_yield_provider: Union[str, Sequence[str]] = ("yfinance", "tiingo"),
) -> Dict[str, Any]:
    """
    Build a compact JSON-friendly dict for LLM input:
      - last n trading days: open, close (+ optional volume)
      - latest pe + dividend yield (best-effort, free-first providers)
      - trailing dividend yield series (optional; skipped if credentials missing)
    """

    # timezone-aware UTC 
    now_utc = datetime.now(timezone.utc)
    end_date = now_utc.date()
    start_date = (end_date - timedelta(days=max(10, n_days * 2))).isoformat()

    # 1) Prices (OHLCV)
    price_df = (
        obb.equity.price.historical(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date.isoformat(),
            interval="1d",
            provider=price_provider,
        )
        .to_df()
    )
    print("########")
    print(price_df.columns)
    print("########")
    price_df = _df_last_n_trading_days(price_df, n_days)

    date_col = _pick_col(price_df, ["date", "datetime", "time"])
    open_col = _pick_col(price_df, ["open", "adj_open"])
    close_col = _pick_col(price_df, ["close", "adj_close", "adjusted_close", "last_price"])
    high_col = _pick_col(price_df, ["high", "adj_high"])
    low_col = _pick_col(price_df, ["low", "adj_low"])
    vol_col = _pick_col(price_df, ["volume", "adj_volume"])

    prices: List[Dict[str, Any]] = []
    if price_df is not None and (not price_df.empty) and open_col and close_col and high_col and low_col:
        if date_col:
            for _, r in price_df.iterrows():
                d = pd.to_datetime(r[date_col], errors="coerce", utc=True)
                if pd.isna(d):
                    continue
                item = {
                    "date": d.date().isoformat(),
                    "open": float(r[open_col]),
                    "close": float(r[close_col]),
                    "high": float(r[high_col]),
                    "low": float(r[low_col]),
                }
                if vol_col and pd.notna(r.get(vol_col)):
                    item["volume"] = float(r[vol_col])
                prices.append(item)
        else:
            for idx, r in price_df.iterrows():
                d = pd.to_datetime(idx, errors="coerce", utc=True)
                if pd.isna(d):
                    continue
                item = {
                    "date": d.date().isoformat(),
                    "open": float(r[open_col]),
                    "close": float(r[close_col]),
                    "high": float(r[high_col]),
                    "low": float(r[low_col]),
                }
                if vol_col and pd.notna(r.get(vol_col)):
                    item["volume"] = float(r[vol_col])
                prices.append(item)

    # 2) Fundamental metrics — free-first fallback
    metric_candidates: Dict[str, List[str]] = {
        # 规模/估值
        "market_cap": ["market_cap", "market cap", "market capitalization"],
        "pe_ratio": ["pe_ratio", "pe", "trailing_pe", "price_to_earnings", "p/e", "pe ratio"],
        "forward_pe": ["forward_pe", "forward p/e", "forward pe"],
        "enterprise_to_ebitda": [
            "enterprise_to_ebitda",
            "enterprise value to ebitda",
            "ev/ebitda",
        ],
        "enterprise_to_revenue": [
            "enterprise_to_revenue",
            "enterprise value to revenue",
            "ev/revenue",
            "ev/sales",
        ],
        "price_to_book": ["price_to_book", "price to book", "p/b", "pb"],
        # 增长
        "revenue_growth": ["revenue_growth", "revenue growth", "sales_growth", "sales growth", "rev growth"],
        "earnings_growth": ["earnings_growth", "earnings growth", "eps_growth", "eps growth"],
        # 盈利能力
        "gross_margin": ["gross_margin", "gross margin"],
        "operating_margin": ["operating_margin", "operating margin"],
        "profit_margin": ["profit_margin", "profit margin", "net_margin", "net margin"],
        "return_on_equity": ["return_on_equity", "return on equity", "roe"],
        # 偿债/杠杆
        "debt_to_equity": ["debt_to_equity", "debt to equity", "debt/equity"],
        "current_ratio": ["current_ratio", "current ratio"],
        # 分红/风险
        "dividend_yield": [
            "dividend_yield",
            "trailing_dividend_yield",
            "forward_dividend_yield",
            "dividend yield",
        ],
        "payout_ratio": ["payout_ratio", "payout ratio"],
        "beta": ["beta"],
        "price_return_1y": [
            "price_return_1y",
            "1y_return",
            "1-year return",
            "1 year return",
            "return_1y",
            "one_year_return",
            "price return 1y",
            "52-week return",
        ],
    }

    latest_metrics: Dict[str, Any] = {k: None for k in metric_candidates}
    used_metrics_provider: Optional[str] = None

    metrics_providers = _ensure_providers(metrics_provider)
    # 如果你传了单个 provider，这里也会先试它；再补上常见免 key 的兜底
    for p in ["finviz", "yfinance"]:
        if p not in metrics_providers:
            metrics_providers.append(p)

    for p in metrics_providers:
        try:
            metrics_df = obb.equity.fundamental.metrics(symbol=symbol, provider=p).to_df()
            if metrics_df is None or metrics_df.empty:
                continue
            m = metrics_df.iloc[0].to_dict()
            print("########")
            print(metrics_df.columns)
            print("########")
            for key, candidates in metric_candidates.items():
                if latest_metrics.get(key) is not None:
                    continue
                col = _pick_col(metrics_df, candidates)
                if not col:
                    continue
                val = _to_float(m.get(col))
                if val is not None:
                    latest_metrics[key] = val

            used_metrics_provider = p
            # 如果全部拿到，就不再继续
            if all(v is not None for v in latest_metrics.values()):
                break
        except Exception:
            continue

    # 3) Trailing dividend yield time series (optional) — best-effort, skip if missing creds
    div_yield_series: List[Dict[str, Any]] = []
    used_dy_provider: Optional[str] = None

    dy_providers = _ensure_providers(div_yield_provider)

    for p in dy_providers:
        try:
            dy_df = obb.equity.fundamental.trailing_dividend_yield(
                symbol=symbol,
                limit=max(252, n_days),  # 252≈1年交易日
                provider=p,
            ).to_df()
            if dy_df is None or dy_df.empty:
                continue

            dy_df = _df_last_n_trading_days(dy_df, n_days)

            dy_date_col = _pick_col(dy_df, ["date", "datetime", "time"])
            dy_val_col = _pick_col(dy_df, ["trailing_dividend_yield", "dividend_yield"])

            if not dy_val_col:
                continue

            series: List[Dict[str, Any]] = []
            if dy_date_col:
                for _, r in dy_df.iterrows():
                    d = pd.to_datetime(r[dy_date_col], errors="coerce", utc=True)
                    v = _to_float(r.get(dy_val_col))
                    if pd.isna(d) or v is None:
                        continue
                    series.append({"date": d.date().isoformat(), "trailing_dividend_yield": v})
            else:
                for idx, r in dy_df.iterrows():
                    d = pd.to_datetime(idx, errors="coerce", utc=True)
                    v = _to_float(r.get(dy_val_col))
                    if pd.isna(d) or v is None:
                        continue
                    series.append({"date": d.date().isoformat(), "trailing_dividend_yield": v})

            if series:
                div_yield_series = series
                used_dy_provider = p

                # 如果 metrics 没拿到 dividend_yield，就用序列最后值补上
                if latest_metrics.get("dividend_yield") is None:
                    latest_metrics["dividend_yield"] = series[-1]["trailing_dividend_yield"]
                break

        except OpenBBError:
            # 常见情况：缺 token（比如 tiingo_token）
            continue
        except Exception:
            continue

    payload: Dict[str, Any] = {
        "symbol": symbol,
        "window_trading_days": n_days,
        "asof_utc": now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "providers": {
            "price": price_provider,
            "metrics": used_metrics_provider,
            "trailing_dividend_yield": used_dy_provider,
        },
        "prices": prices,  # [{date, open, close, volume?}, ...]
        "fundamentals_latest": latest_metrics,  # includes selected fundamentals (None when unavailable)
        "trailing_dividend_yield_series": div_yield_series,  # optional; may be []
    }
    return payload


def save_equity_json(
    payload: Dict[str, Any],
    out_dir: Union[str, Path] = "outputs",
    filename: Optional[str] = None,
) -> Path:
    """
    Save payload to a JSON file under out_dir.
    Returns the path to the saved file.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    symbol = str(payload.get("symbol") or "equity").upper()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = filename or f"{symbol}_{ts}.json"

    file_path = out_path / fname
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return file_path


if __name__ == "__main__":
    data = build_equity_json("AAPL", n_days=30)
    out_file = save_equity_json(data, out_dir="outputs")
    print(f"saved: {out_file}")
