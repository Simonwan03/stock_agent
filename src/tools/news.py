"""
Free, higher-quality news fetcher:
1) Primary: OpenBB company news (provider=yfinance)
2) Fallback: GDELT DOC API with domain allowlist
3) Merge + URL/title dedup + simple ranking

Drop this file into your project and import fetch_news_merged_free().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from openbb import obb

# ----------------------------
# Config
# ----------------------------
GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_DOMAIN_ALLOWLIST = [
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "cnbc.com",
    "marketwatch.com",
    "finance.yahoo.com",
    "theinformation.com",
]

DEFAULT_EXTRA_TERMS = [
    "earnings",
    "guidance",
    "revenue",
    "EPS",
    "SEC",
    "lawsuit",
    "acquisition",
    "merger",
    "antitrust",
    "downgrade",
    "upgrade",
]


# ----------------------------
# Data models
# ----------------------------
@dataclass(frozen=True)
class NewsItem:
    title: str
    url: str
    source: str
    published_at: str
    tickers: List[str]
    provider: str
    domain: str = ""
    text: str = ""


# ----------------------------
# Utilities
# ----------------------------
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso_z(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _norm_url(url: str) -> str:
    """Normalize URL for dedup: strip fragments; keep scheme+host+path+query."""
    url = (url or "").strip()
    if not url:
        return ""
    try:
        u = urllib.parse.urlsplit(url)
        # keep query because some sites use it as article id; remove fragment
        return urllib.parse.urlunsplit((u.scheme.lower(), u.netloc.lower(), u.path, u.query, ""))
    except Exception:
        return url


def _norm_title(title: str) -> str:
    t = (title or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t


def _hash(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


def _dedup(items: List[NewsItem]) -> List[NewsItem]:
    seen = set()
    out: List[NewsItem] = []
    for it in items:
        k1 = _hash(_norm_url(it.url))
        k2 = _hash((it.domain or "") + "|" + _norm_title(it.title))
        if k1 in seen or k2 in seen:
            continue
        seen.add(k1)
        seen.add(k2)
        out.append(it)
    return out


def _domain_weight(domain: str) -> int:
    d = (domain or "").lower()
    if "reuters.com" in d:
        return 100
    if "bloomberg.com" in d:
        return 95
    if "ft.com" in d:
        return 90
    if "wsj.com" in d:
        return 90
    if "cnbc.com" in d:
        return 60
    if "marketwatch.com" in d:
        return 60
    if "finance.yahoo.com" in d:
        return 50
    return 50  # unknown/other allowlisted


def _parse_dt_maybe(s: str) -> Optional[datetime]:
    if not s:
        return None
    # GDELT seendate is often like "20260127T123456Z" or "2026-01-27 12:34:56"
    s = s.strip()
    try:
        if s.endswith("Z") and "T" in s and len(s) >= 16 and s[8] == "T":
            # 20260127T123456Z
            core = s[:-1]
            return datetime.strptime(core, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        pass
    try:
        # ISO
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        pass
    try:
        return pd.to_datetime(s, utc=True, errors="coerce").to_pydatetime()
    except Exception:
        return None


def _score_item(it: NewsItem, tickers: Sequence[str]) -> float:
    dt = _parse_dt_maybe(it.published_at)
    recency = 0.0
    if dt:
        age_hours = (_utc_now() - dt).total_seconds() / 3600.0
        recency = max(0.0, 72.0 - age_hours)  # within 72h gets boosted

    title = _norm_title(it.title)
    ticker_hits = sum(1 for t in tickers if t.lower() in title)
    quality = _domain_weight(it.domain)

    # Provider bias: OpenBB company feed usually more company-relevant than GDELT
    provider_boost = 10 if it.provider.startswith("openbb") else 0

    return quality + recency + 5.0 * ticker_hits + provider_boost


# ----------------------------
# OpenBB (free) primary
# ----------------------------
def fetch_openbb_company_news(
    tickers: Sequence[str],
    lookback_hours: int = 72,
    max_per_ticker: int = 10,
) -> List[NewsItem]:
    cutoff = _utc_now() - timedelta(hours=lookback_hours)
    out: List[NewsItem] = []

    for t in tickers:
        try:
            df = obb.news.company(symbol=t, provider="yfinance").to_df()
            if df is None or df.empty:
                continue

            cols = {c.lower(): c for c in df.columns}
            title_c = cols.get("title")
            url_c = cols.get("url")
            source_c = cols.get("source") or cols.get("publisher")
            date_c = cols.get("date") or cols.get("published_at") or cols.get("datetime")
            text_c = cols.get("text") or cols.get("summary")
            domain_c = cols.get("domain")

            rows: List[Tuple[Optional[datetime], Dict[str, Any]]] = []
            for _, r in df.iterrows():
                dt = None
                if date_c and r.get(date_c) is not None:
                    dt = _parse_dt_maybe(str(r.get(date_c)))
                if dt and dt < cutoff:
                    continue

                item = {
                    "title": str(r.get(title_c, "")) if title_c else "",
                    "url": str(r.get(url_c, "")) if url_c else "",
                    "source": str(r.get(source_c, "")) if source_c else "",
                    "domain": str(r.get(domain_c, "")) if domain_c else "",
                    "published_at": _to_iso_z(dt) if dt else (str(r.get(date_c, "")) if date_c else ""),
                    "text": str(r.get(text_c, "")) if text_c and r.get(text_c) else "",
                }
                
                rows.append((dt, item))

            # newest first, cap
            rows.sort(key=lambda x: x[0] or datetime(1970, 1, 1, tzinfo=timezone.utc), reverse=True)
            for _, item in rows[:max_per_ticker]:
                out.append(
                    NewsItem(
                        title=item["title"],
                        url=item["url"],
                        source=item["source"],
                        domain=item["domain"],
                        published_at=item["published_at"],
                        tickers=[t],
                        provider="openbb:yfinance",
                        text=item["text"],
                    )
                )
        except Exception:
            continue

    return out


# ----------------------------
# GDELT fallback (free) + allowlist
# ----------------------------
def _normalize_query(q: str) -> str:
    q = q.strip()
    if " OR " in q and not (q.startswith("(") and q.endswith(")")):
        q = f"({q})"
    return q


def _domains_clause(domains: Sequence[str]) -> str:
    parts = [f"domain:{d}" for d in domains]
    return "(" + " OR ".join(parts) + ")" if parts else ""


def _build_gdelt_url(
    query: str,
    max_records: int,
    lookback_hours: Optional[int],
    domains_allowlist: Optional[Sequence[str]],
) -> str:
    query = _normalize_query(query)
    filters = [query, "sourcelang:eng"]
    if domains_allowlist:
        filters.append(_domains_clause(domains_allowlist))

    params = {
        "query": " ".join([f for f in filters if f]),
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": "datedesc",
    }

    if lookback_hours and lookback_hours > 0:
        end_dt = _utc_now()
        start_dt = end_dt - timedelta(hours=lookback_hours)
        params["startdatetime"] = start_dt.strftime("%Y%m%d%H%M%S")
        params["enddatetime"] = end_dt.strftime("%Y%m%d%H%M%S")

    return f"{GDELT_DOC_ENDPOINT}?{urllib.parse.urlencode(params)}"


def _gdelt_fetch_payload(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (news-cli)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    if not text.strip():
        raise RuntimeError("Empty GDELT response")
    return json.loads(text)


def fetch_gdelt_news(
    tickers: Sequence[str],
    lookback_hours: int = 72,
    max_records: int = 40,
    domains_allowlist: Optional[Sequence[str]] = None,
    extra_terms: Optional[Sequence[str]] = None,
) -> List[NewsItem]:
    # More precise than "AAPL OR MSFT": require some finance/market terms too
    base = " OR ".join([f'"{t}"' for t in tickers])
    if extra_terms:
        extras = " OR ".join([f'"{x}"' for x in extra_terms])
        query = f"({base}) AND ({extras})"
    else:
        query = base

    url = _build_gdelt_url(
        query=query,
        max_records=max_records,
        lookback_hours=lookback_hours,
        domains_allowlist=domains_allowlist,
    )

    payload = _gdelt_fetch_payload(url)

    out: List[NewsItem] = []
    for e in payload.get("articles", []):
        out.append(
            NewsItem(
                title=e.get("title", ""),
                url=e.get("url", ""),
                source=e.get("sourceCountry", "") or e.get("sourcecountry", ""),
                domain=e.get("domain", ""),
                published_at=e.get("seendate", ""),
                tickers=list(tickers),
                provider="gdelt",
                text="",  # GDELT artlist usually doesn't include full text
            )
        )
    return out


# ----------------------------
# Merged public API
# ----------------------------
def fetch_news_merged_free(
    tickers: Sequence[str],
    lookback_hours: int = 72,
    max_items: int = 20,
    gdelt_domains_allowlist: Sequence[str] = DEFAULT_DOMAIN_ALLOWLIST,
    gdelt_extra_terms: Sequence[str] = DEFAULT_EXTRA_TERMS,
) -> List[Dict[str, Any]]:
    """
    Returns JSON-ready list (dicts), merged + deduped + ranked.
    """
    tickers = [t for t in tickers if t]
    if not tickers:
        return []

    items: List[NewsItem] = []
    items.extend(fetch_openbb_company_news(tickers, lookback_hours=lookback_hours, max_per_ticker=max(5, max_items // max(1, len(tickers)))))
    # Fallback only if not enough
    if len(items) < max_items:
        try:
            items.extend(
                fetch_gdelt_news(
                    tickers,
                    lookback_hours=lookback_hours,
                    max_records=max_items * 3,
                    domains_allowlist=gdelt_domains_allowlist,
                    extra_terms=gdelt_extra_terms,
                )
            )
        except Exception:
            pass

    items = _dedup(items)

    # rank
    items.sort(key=lambda it: _score_item(it, tickers), reverse=True)

    # output as dicts
    out: List[Dict[str, Any]] = []
    for it in items[:max_items]:
        out.append(
            {
                "title": it.title,
                "url": it.url,
                "source": it.source,
                "domain": it.domain,
                "published_at": it.published_at,
                "tickers": it.tickers,
                "provider": it.provider,
                **({"text": it.text} if it.text else {}),
            }
        )
    return out


# ----------------------------
# Output helpers
# ----------------------------
def save_news_json(
    items: List[Dict[str, Any]],
    tickers: Sequence[str],
    out_dir: str | Path = "outputs",
    filename: Optional[str] = None,
) -> Path:
    """
    Save news items to a JSON file under out_dir.
    Returns the path to the saved file.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tickers_key = "_".join([t.upper() for t in tickers if t]) or "NEWS"
    ts = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    fname = filename or f"{tickers_key}_news_{ts}.json"

    fname_path = Path(fname)
    if fname_path.is_absolute() or fname_path.parent != Path("."):
        file_path = fname_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        file_path = out_path / fname
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return file_path


# ----------------------------
# CLI
# ----------------------------
def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch merged FREE news (OpenBB yfinance + GDELT allowlist).")
    p.add_argument("--tickers", nargs="+", default=["AAPL"], help="Tickers, e.g. AAPL MSFT")
    p.add_argument("--lookback_hours", type=int, default=72)
    p.add_argument("--max", type=int, default=20, help="Max items to return (alias: --limit).")
    p.add_argument("--limit", type=int, default=None, help="Alias for --max.")
    p.add_argument("--out_dir", default="outputs", help="Output directory (default: outputs)")
    p.add_argument("--out", default="", help="Output filename (.json). If empty, a timestamped name is used.")
    args = p.parse_args(argv)
    if args.limit is not None:
        args.max = args.limit
    return args


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    items = fetch_news_merged_free(args.tickers, lookback_hours=args.lookback_hours, max_items=args.max)
    out_file = save_news_json(items, args.tickers, out_dir=args.out_dir, filename=(args.out or None))
    print(f"saved: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
