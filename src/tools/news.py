"""Fetch financial news from the public GDELT 2.0 API."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass(frozen=True)
class NewsArticle:
    title: str
    url: str
    source: str
    published_at: str


def build_gdelt_url(query: str, max_records: int) -> str:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": "datedesc",
    }
    return f"{GDELT_DOC_ENDPOINT}?{urllib.parse.urlencode(params)}"


def fetch_financial_news(query: str, max_records: int = 20) -> List[NewsArticle]:
    if max_records < 1:
        raise ValueError("max_records must be >= 1")

    url = build_gdelt_url(query, max_records)
    with urllib.request.urlopen(url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

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
    raise SystemExit(main(sys.argv[1:]))
