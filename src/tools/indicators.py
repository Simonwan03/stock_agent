#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indicators toolkit + CLI.

Usage examples:
  python src/tools/indicators.py --prices outputs/AAPL.json --out outputs/aapl_indicators.json
  python src/tools/indicators.py --ticker AAPL --outputs-dir src/tools/outputs --out src/tools/outputs/aapl_indicators.json
  python src/tools/indicators.py --prices prices.csv --benchmark spy.csv --latest-only
  python src/tools/indicators.py --financials outputs/openbb_financials_AAPL_quarter_yfinance_income-balance-cash.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _normalize_col(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
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


def _series_from_df(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[pd.Series]:
    col = _pick_col(df, candidates)
    if not col:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def _drawdown_series(close: pd.Series) -> pd.Series:
    rets = _returns(close).dropna()
    if rets.empty:
        return pd.Series(dtype=float)
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    return cum / peak - 1.0


def annualized_return(rets: pd.Series, periods_per_year: int = 252) -> Optional[float]:
    rets = rets.dropna()
    if rets.empty:
        return None
    total = (1 + rets).prod()
    return float(total ** (periods_per_year / len(rets)) - 1)


def annualized_volatility(rets: pd.Series, periods_per_year: int = 252) -> Optional[float]:
    rets = rets.dropna()
    if rets.empty:
        return None
    return float(rets.std() * np.sqrt(periods_per_year))


def sharpe_ratio(rets: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> Optional[float]:
    rets = rets.dropna()
    if rets.empty:
        return None
    excess = rets - rf / periods_per_year
    denom = excess.std()
    if denom == 0 or np.isnan(denom):
        return None
    return float(excess.mean() / denom * np.sqrt(periods_per_year))


def sortino_ratio(rets: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> Optional[float]:
    rets = rets.dropna()
    if rets.empty:
        return None
    excess = rets - rf / periods_per_year
    downside = excess[excess < 0]
    if downside.empty:
        return None
    downside_dev = np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year)
    if downside_dev == 0 or np.isnan(downside_dev):
        return None
    return float(excess.mean() * np.sqrt(periods_per_year) / downside_dev)


def max_drawdown(close: pd.Series) -> Optional[float]:
    dd = _drawdown_series(close)
    if dd.empty:
        return None
    return float(dd.min())


def ulcer_index(close: pd.Series) -> Optional[float]:
    dd = _drawdown_series(close)
    if dd.empty:
        return None
    return float(np.sqrt((dd**2).mean()))


def calmar_ratio(close: pd.Series, periods_per_year: int = 252) -> Optional[float]:
    rets = _returns(close).dropna()
    if rets.empty:
        return None
    ann_ret = annualized_return(rets, periods_per_year=periods_per_year)
    mdd = max_drawdown(close)
    if ann_ret is None or mdd in (None, 0):
        return None
    return float(ann_ret / abs(mdd))


def beta_alpha(
    close: pd.Series,
    benchmark_close: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> Tuple[Optional[float], Optional[float]]:
    r = _returns(close)
    b = _returns(benchmark_close)
    df = pd.concat([r, b], axis=1).dropna()
    if df.empty:
        return None, None
    r = df.iloc[:, 0]
    b = df.iloc[:, 1]
    cov = np.cov(r, b, ddof=0)
    var_b = cov[1, 1]
    if var_b == 0 or np.isnan(var_b):
        return None, None
    beta = cov[0, 1] / var_b
    ann_r = annualized_return(r, periods_per_year=periods_per_year)
    ann_b = annualized_return(b, periods_per_year=periods_per_year)
    if ann_r is None or ann_b is None:
        return float(beta), None
    alpha = ann_r - (rf + beta * (ann_b - rf))
    return float(beta), float(alpha)


def information_ratio(
    close: pd.Series,
    benchmark_close: pd.Series,
    periods_per_year: int = 252,
) -> Optional[float]:
    r = _returns(close)
    b = _returns(benchmark_close)
    active = (r - b).dropna()
    if active.empty:
        return None
    te = active.std()
    if te == 0 or np.isnan(te):
        return None
    return float(active.mean() / te * np.sqrt(periods_per_year))


def tracking_error(
    close: pd.Series,
    benchmark_close: pd.Series,
    periods_per_year: int = 252,
) -> Optional[float]:
    r = _returns(close)
    b = _returns(benchmark_close)
    active = (r - b).dropna()
    if active.empty:
        return None
    return float(active.std() * np.sqrt(periods_per_year))


def risk_metrics(
    close: pd.Series,
    periods_per_year: int = 252,
    rf: float = 0.0,
) -> Dict[str, Optional[float]]:
    rets = _returns(close).dropna()
    ann_vol = annualized_volatility(rets, periods_per_year=periods_per_year)
    mdd = max_drawdown(close)
    return {
        "ann_return": annualized_return(rets, periods_per_year=periods_per_year),
        "ann_vol": ann_vol,
        "sharpe": sharpe_ratio(rets, rf=rf, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(rets, rf=rf, periods_per_year=periods_per_year),
        "max_drawdown": mdd,
        "calmar": calmar_ratio(close, periods_per_year=periods_per_year),
        "ulcer_index": ulcer_index(close),
    }


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    k = 100 * (close - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    low_min = low.rolling(period).min()
    high_max = high.rolling(period).max()
    return -100 * (high_max - close) / (high_max - low_min).replace(0, np.nan)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = _atr(high, low, close, period=period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_technical_indicators(
    df: pd.DataFrame,
    sma_windows: Sequence[int] = (5, 10, 20, 50, 100, 200),
    ema_windows: Sequence[int] = (12, 26),
    rsi_period: int = 14,
    bb_window: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
    stoch_k_period: int = 14,
    stoch_d_period: int = 3,
    adx_period: int = 14,
    roc_period: int = 12,
    momentum_period: int = 10,
    vol_window: int = 20,
    donchian_window: int = 20,
    keltner_ema_period: int = 20,
    keltner_atr_period: int = 10,
    keltner_atr_mult: float = 2.0,
) -> pd.DataFrame:
    """
    Compute a broad set of technical indicators. Missing columns are skipped.
    Expected columns (best-effort): open/high/low/close/volume.
    """
    out = pd.DataFrame(index=df.index)

    close = _series_from_df(df, ["close", "adj_close", "adjusted_close", "last_price"])
    high = _series_from_df(df, ["high", "adj_high"])
    low = _series_from_df(df, ["low", "adj_low"])
    volume = _series_from_df(df, ["volume", "adj_volume"])

    if close is None:
        return out

    # Returns and volatility
    ret = _returns(close)
    out["ret_1d"] = ret
    out[f"ret_{roc_period}d"] = close.pct_change(roc_period)
    out["log_ret_1d"] = np.log(close / close.shift(1))
    out[f"vol_{vol_window}d"] = ret.rolling(vol_window).std() * np.sqrt(252)

    # Trend and momentum
    for w in sma_windows:
        out[f"sma_{w}"] = close.rolling(w).mean()
    for w in ema_windows:
        out[f"ema_{w}"] = _ema(close, w)

    out[f"rsi_{rsi_period}"] = _rsi(close, period=rsi_period)
    macd_line, macd_signal, macd_hist = _macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    out[f"roc_{roc_period}"] = 100 * (close / close.shift(roc_period) - 1)
    out[f"momentum_{momentum_period}"] = close - close.shift(momentum_period)

    # Bands and channels
    bb_upper, bb_mid, bb_lower = _bollinger(close, window=bb_window, num_std=bb_std)
    out[f"bb_upper_{bb_window}"] = bb_upper
    out[f"bb_mid_{bb_window}"] = bb_mid
    out[f"bb_lower_{bb_window}"] = bb_lower

    if high is not None and low is not None:
        out[f"donchian_high_{donchian_window}"] = high.rolling(donchian_window).max()
        out[f"donchian_low_{donchian_window}"] = low.rolling(donchian_window).min()

    # Volatility / range based indicators
    if high is not None and low is not None:
        out[f"atr_{atr_period}"] = _atr(high, low, close, period=atr_period)
        out[f"adx_{adx_period}"] = _adx(high, low, close, period=adx_period)
        k, d = _stochastic(high, low, close, k_period=stoch_k_period, d_period=stoch_d_period)
        out[f"stoch_k_{stoch_k_period}"] = k
        out[f"stoch_d_{stoch_k_period}"] = d
        out[f"williams_r_{stoch_k_period}"] = _williams_r(high, low, close, period=stoch_k_period)

        # Keltner Channels
        typical = (high + low + close) / 3
        kc_mid = _ema(typical, keltner_ema_period)
        kc_atr = _atr(high, low, close, period=keltner_atr_period)
        out[f"kc_upper_{keltner_ema_period}"] = kc_mid + keltner_atr_mult * kc_atr
        out[f"kc_mid_{keltner_ema_period}"] = kc_mid
        out[f"kc_lower_{keltner_ema_period}"] = kc_mid - keltner_atr_mult * kc_atr

    # Volume based indicators
    if volume is not None:
        direction = np.sign(close.diff()).fillna(0.0)
        out["obv"] = (direction * volume).cumsum()
        out["vol_sma_20"] = volume.rolling(20).mean()

        if high is not None and low is not None:
            typical = (high + low + close) / 3
            vwap = (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)
            out["vwap"] = vwap

            money_flow = typical * volume
            pos_mf = money_flow.where(typical > typical.shift(1), 0.0)
            neg_mf = money_flow.where(typical < typical.shift(1), 0.0)
            pos_mf_sum = pos_mf.rolling(rsi_period).sum()
            neg_mf_sum = neg_mf.rolling(rsi_period).sum()
            mfi = 100 - (100 / (1 + pos_mf_sum / neg_mf_sum.replace(0, np.nan)))
            out[f"mfi_{rsi_period}"] = mfi

    return out


def compute_performance_snapshot(
    close: pd.Series,
    benchmark_close: Optional[pd.Series] = None,
    periods_per_year: int = 252,
    rf: float = 0.0,
) -> Dict[str, Optional[float]]:
    metrics = risk_metrics(close, periods_per_year=periods_per_year, rf=rf)
    if benchmark_close is None:
        return metrics

    beta, alpha = beta_alpha(close, benchmark_close, rf=rf, periods_per_year=periods_per_year)
    metrics.update(
        {
            "beta": beta,
            "alpha": alpha,
            "information_ratio": information_ratio(close, benchmark_close, periods_per_year=periods_per_year),
            "tracking_error": tracking_error(close, benchmark_close, periods_per_year=periods_per_year),
        }
    )
    return metrics


def _df_from_statement(statement: Optional[Dict[str, Sequence[float]]]) -> Optional[pd.DataFrame]:
    if not statement:
        return None
    df = pd.DataFrame(statement)
    idx_col = _pick_col(df, ["period_ending", "date", "period"])
    if idx_col:
        df[idx_col] = pd.to_datetime(df[idx_col], errors="coerce")
        df = df.set_index(idx_col)
    return df


def _get_col(df: Optional[pd.DataFrame], candidates: Sequence[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    col = _pick_col(df, candidates)
    if not col:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def compute_financial_ratios(
    income: Optional[Dict[str, Sequence[float]]] = None,
    balance: Optional[Dict[str, Sequence[float]]] = None,
    cash: Optional[Dict[str, Sequence[float]]] = None,
    periods_per_year: int = 4,
) -> pd.DataFrame:
    """
    Compute common financial ratios from OpenBB-style statements (column-wise arrays).
    Returns a DataFrame indexed by period_ending when available.
    """
    income_df = _df_from_statement(income)
    balance_df = _df_from_statement(balance)
    cash_df = _df_from_statement(cash)

    out = pd.DataFrame()

    revenue = _get_col(income_df, ["total_revenue", "operating_revenue", "revenue"])
    gross_profit = _get_col(income_df, ["gross_profit"])
    operating_income = _get_col(income_df, ["operating_income", "ebit"])
    net_income = _get_col(income_df, ["net_income", "net_income_attributable_to_common_shareholders"])
    ebitda = _get_col(income_df, ["ebitda"])
    ebit = _get_col(income_df, ["ebit"])
    interest_expense = _get_col(income_df, ["interest_expense", "interest_expenses"])

    total_assets = _get_col(balance_df, ["total_assets"])
    total_equity = _get_col(balance_df, ["total_stockholder_equity", "total_equity", "total_common_equity"])
    current_assets = _get_col(balance_df, ["total_current_assets"])
    current_liabilities = _get_col(balance_df, ["total_current_liabilities"])
    cash_eq = _get_col(balance_df, ["cash_and_cash_equivalents", "cash_equivalents", "cash"])
    short_term_investments = _get_col(balance_df, ["short_term_investments"])
    receivables = _get_col(balance_df, ["net_receivables", "accounts_receivable"])
    total_debt = _get_col(balance_df, ["total_debt", "long_term_debt_and_capital_lease_obligation"])

    cfo = _get_col(cash_df, ["operating_cash_flow", "cash_flow_from_operating_activities"])
    capex = _get_col(cash_df, ["capital_expenditure", "capital_expenditures"])
    free_cf = _get_col(cash_df, ["free_cash_flow"])

    # Profitability margins
    if revenue is not None and gross_profit is not None:
        out["gross_margin"] = _safe_div(gross_profit, revenue)
    if revenue is not None and operating_income is not None:
        out["operating_margin"] = _safe_div(operating_income, revenue)
    if revenue is not None and net_income is not None:
        out["net_margin"] = _safe_div(net_income, revenue)
    if revenue is not None and ebitda is not None:
        out["ebitda_margin"] = _safe_div(ebitda, revenue)

    # Returns
    if net_income is not None and total_equity is not None:
        out["roe"] = _safe_div(net_income, total_equity)
    if net_income is not None and total_assets is not None:
        out["roa"] = _safe_div(net_income, total_assets)

    # Leverage & liquidity
    if total_debt is not None and total_equity is not None:
        out["debt_to_equity"] = _safe_div(total_debt, total_equity)
    if current_assets is not None and current_liabilities is not None:
        out["current_ratio"] = _safe_div(current_assets, current_liabilities)
    if current_liabilities is not None:
        quick_num = None
        if cash_eq is not None:
            quick_num = cash_eq.copy()
        if short_term_investments is not None:
            quick_num = short_term_investments if quick_num is None else quick_num + short_term_investments
        if receivables is not None:
            quick_num = receivables if quick_num is None else quick_num + receivables
        if quick_num is not None:
            out["quick_ratio"] = _safe_div(quick_num, current_liabilities)
        if cash_eq is not None:
            out["cash_ratio"] = _safe_div(cash_eq, current_liabilities)

    # Coverage
    if ebit is not None and interest_expense is not None:
        out["interest_coverage"] = _safe_div(ebit, interest_expense.abs())

    # Cash flow
    if free_cf is None and cfo is not None and capex is not None:
        # capex is often negative in raw statements
        free_cf = cfo + capex
    if free_cf is not None:
        out["free_cash_flow"] = free_cf
    if free_cf is not None and revenue is not None:
        out["fcf_margin"] = _safe_div(free_cf, revenue)
    if cfo is not None and revenue is not None:
        out["cfo_margin"] = _safe_div(cfo, revenue)

    # Growth
    if revenue is not None:
        out["revenue_growth_yoy"] = revenue.pct_change(periods=periods_per_year)
    if net_income is not None:
        out["net_income_growth_yoy"] = net_income.pct_change(periods=periods_per_year)

    if total_debt is not None and cash_eq is not None:
        out["net_debt"] = total_debt - cash_eq
        if ebitda is not None:
            out["net_debt_to_ebitda"] = _safe_div(out["net_debt"], ebitda)

    return out


def load_json_allow_nan(text: str) -> Any:
    def _parse_constant(_: str) -> Any:
        return None

    return json.loads(text, parse_constant=_parse_constant)


def _read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _load_json(path: str) -> Any:
    return load_json_allow_nan(_read_text(path))


def _load_prices_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raw = _load_json(path)
        if isinstance(raw, dict) and "prices" in raw:
            records = raw.get("prices") or []
        elif isinstance(raw, list):
            records = raw
        else:
            raise ValueError("Unsupported prices JSON: expected list[dict] or dict with 'prices'.")
        df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(f"Empty prices input: {path}")
    return df


def _prepare_prices_df(df: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick_col(df, ["date", "datetime", "time"])
    if date_col:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        df = df.set_index(date_col)
    return df


def _extract_statements(
    raw: Any,
) -> Tuple[Optional[Dict[str, Sequence[float]]], Optional[Dict[str, Sequence[float]]], Optional[Dict[str, Sequence[float]]]]:
    if isinstance(raw, dict) and "statements" in raw:
        raw = raw["statements"] or {}
    if not isinstance(raw, dict):
        return None, None, None
    return raw.get("income"), raw.get("balance"), raw.get("cash")


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    else:
        out.insert(0, "index", out.index)
    out = out.where(pd.notna(out), None)
    return out.to_dict(orient="records")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)


def _infer_ticker_from_stem(stem: str) -> Optional[str]:
    if not stem:
        return None
    base = stem.split("_", 1)[0]
    return base if base.isalpha() else None


def _is_prices_payload(path: Path) -> bool:
    try:
        raw = _load_json(str(path))
    except Exception:
        return False
    if isinstance(raw, dict) and isinstance(raw.get("prices"), list):
        return True
    if isinstance(raw, list):
        return True
    return False


def _find_latest_prices_file(outputs_dir: Path, ticker: str) -> Optional[Path]:
    patterns = [
        f"{ticker.upper()}.json",
        f"{ticker.lower()}.json",
        f"{ticker.upper()}_*.json",
        f"{ticker.lower()}_*.json",
    ]
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(outputs_dir.glob(pattern))
    matches = [m for m in matches if m.is_file() and _is_prices_payload(m)]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _default_outputs_dir(outputs_dir: str) -> Path:
    if outputs_dir:
        return Path(outputs_dir)
    repo_outputs = Path("src/tools/outputs")
    if repo_outputs.exists():
        return repo_outputs
    return Path("outputs")


def _resolve_prices_path(prices_arg: str, ticker: str, outputs_dir: str) -> str:
    if not prices_arg:
        if not ticker:
            return ""
        out_dir = _default_outputs_dir(outputs_dir)
        latest = _find_latest_prices_file(out_dir, ticker)
        if not latest:
            raise SystemExit(f"No prices file found for ticker {ticker} in {out_dir}.")
        return str(latest)

    path = Path(prices_arg)
    if path.exists():
        if path.is_dir():
            if not ticker:
                raise SystemExit("--prices points to a directory; use --ticker to resolve a file.")
            latest = _find_latest_prices_file(path, ticker)
            if not latest:
                raise SystemExit(f"No prices file found for ticker {ticker} in {path}.")
            return str(latest)
        return str(path)

    inferred = ticker or _infer_ticker_from_stem(path.stem)
    search_dirs: List[Path] = []
    if path.parent and path.parent != Path("."):
        search_dirs.append(path.parent)
    out_dir = _default_outputs_dir(outputs_dir)
    if out_dir not in search_dirs:
        search_dirs.append(out_dir)
    if inferred:
        for directory in search_dirs:
            if not directory.exists():
                continue
            latest = _find_latest_prices_file(directory, inferred)
            if latest:
                return str(latest)

    raise SystemExit(f"Prices file not found: {prices_arg}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute technical indicators, risk metrics, and financial ratios.")
    p.add_argument("--prices", default="", help="Prices JSON/CSV path. JSON list[dict] or dict with 'prices'.")
    p.add_argument("--ticker", default="", help="Ticker symbol to auto-resolve latest prices file.")
    p.add_argument("--outputs-dir", default="", help="Outputs directory for auto-resolving prices file.")
    p.add_argument("--benchmark", default="", help="Benchmark prices JSON/CSV path.")
    p.add_argument("--financials", default="", help="OpenBB financials JSON path.")
    p.add_argument("--latest-only", action="store_true", help="Only keep the latest row for time-series outputs.")
    p.add_argument("--rf", type=float, default=0.0, help="Annual risk-free rate used in ratios.")
    p.add_argument("--pp", type=int, default=252, help="Periods per year (default: 252).")
    p.add_argument("--out", default="", help="Output JSON path. If omitted, print to stdout.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.prices and not args.financials and not args.ticker:
        raise SystemExit("Need --prices and/or --financials (or --ticker for auto-resolve).")

    output: Dict[str, Any] = {}

    prices_path = _resolve_prices_path(args.prices, args.ticker, args.outputs_dir)
    if prices_path:
        prices_df = _prepare_prices_df(_load_prices_df(prices_path))
        tech_df = compute_technical_indicators(prices_df)
        if args.latest_only and not tech_df.empty:
            tech_df = tech_df.tail(1)
        output["technical"] = _df_to_records(tech_df)

        close = _series_from_df(prices_df, ["close", "adj_close", "adjusted_close", "last_price"])
        if close is not None:
            benchmark_close = None
            if args.benchmark:
                benchmark_df = _prepare_prices_df(_load_prices_df(args.benchmark))
                benchmark_close = _series_from_df(
                    benchmark_df,
                    ["close", "adj_close", "adjusted_close", "last_price"],
                )
            output["performance"] = compute_performance_snapshot(
                close,
                benchmark_close=benchmark_close,
                periods_per_year=args.pp,
                rf=args.rf,
            )

    if args.financials:
        raw = _load_json(args.financials)
        income, balance, cash = _extract_statements(raw)
        fin_df = compute_financial_ratios(income=income, balance=balance, cash=cash)
        if args.latest_only and not fin_df.empty:
            fin_df = fin_df.tail(1)
        output["financial_ratios"] = _df_to_records(fin_df)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_json_dump(output), encoding="utf-8")
        print(f"Saved: {out_path}")
    else:
        print(_json_dump(output))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
