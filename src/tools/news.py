"""Fetch financial news from the public GDELT 2.0 API."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List

# API official documentation:https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
# Example query: URL: https://api.gdeltproject.org/api/v2/doc/doc?query=%22islamic%20state%22&mode=timelinevolinfo&TIMELINESMOOTH=5
GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass(frozen=True)
class NewsArticle:
    title: str # title of the article
    url: str # link to the article
    source: str # source country of the article
    published_at: str # publication date of the article

def normalize_query(q: str) -> str:
    q = q.strip()
    # 如果包含 OR 但没用括号包住，自动包一层
    if " OR " in q and not (q.startswith("(") and q.endswith(")")):
        q = f"({q})"
    return q

def build_gdelt_url(query: str, max_records: int) -> str:
    query = normalize_query(query)
    params = {
        "query": f"{query} sourcelang:eng sourcecountry:US", # English articles only from the US
        "mode": "artlist", # article list mode
        "format": "json",  # response format
        "maxrecords": max_records,
        "sort": "datedesc",
    }
    return f"{GDELT_DOC_ENDPOINT}?{urllib.parse.urlencode(params)}" # complete URL


def fetch_financial_news(query: str, max_records: int = 20) -> List[NewsArticle]:
    # time.sleep(5)  # To avoid rate limiting during rapid testing
    if max_records < 1:
        raise ValueError("max_records must be >= 1")

    url = build_gdelt_url(query, max_records)
    print("Request URL:", url, file=sys.stderr)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (news-cli)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read()
            text = raw.decode("utf-8", errors="replace")
            if not text.strip():
                raise RuntimeError("Empty response body (可能是网络/限流/服务端异常)")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Non-JSON response (HTTP可能异常)。前200字符：\n{text[:200]}"
                ) from e
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTPError {e.code}. Body前200字符：\n{body[:200]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URLError: {e.reason}") from e

    articles: List[NewsArticle] = []
    for entry in payload.get("articles", []):
        articles.append(
            NewsArticle(
                title=entry.get("title", ""),
                url=entry.get("url", ""),
                source=entry.get("sourceCountry", ""),
                published_at=entry.get("seendate", ""),
            )
        )
    return articles


def render_articles(articles: Iterable[NewsArticle]) -> None:
    for article in articles:
        print(f"- {article.title}")
        print(f"  url: {article.url}")
        if article.source:
            print(f"  source: {article.source}")
        if article.published_at:
            print(f"  published_at: {article.published_at}")
        print()

# declare command-line argument parser
def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch financial news from the public GDELT 2.0 API.")
    parser.add_argument(
        "--query",
        default="finance OR stock OR market",
        help="Query string for GDELT search.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=20,
        help="Maximum number of records to return.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    try:
        articles = fetch_financial_news(args.query, args.max)
    except Exception as exc:  # noqa: BLE001 - show failure in CLI
        print(f"Error fetching news: {exc}", file=sys.stderr)
        return 1

    if not articles:
        print("No articles returned.")
        return 0

    render_articles(articles)
    return 0


if __name__ == "__main__":
    # user's input from command line(starting from index 1)
    raise SystemExit(main(sys.argv[1:]))
