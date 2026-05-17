import json
import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

URL = "https://google.serper.dev/news"
KEEP_FIELDS = ("title", "snippet", "link", "date", "source")


def _clean_items(raw_items: list[dict]) -> list[dict]:
    seen_links: set[str] = set()
    cleaned: list[dict] = []
    for item in raw_items:
        link = (item.get("link") or "").strip()
        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        if not (title or snippet):
            continue
        if link and link in seen_links:
            continue
        if link:
            seen_links.add(link)
        cleaned.append({k: (item.get(k) or "") for k in KEEP_FIELDS})
    return cleaned


def getNews(query: str, num: int = 10, days: int = 14, timeout: float = 8.0) -> list[dict]:
    """Fetch recent news for ``query`` from Serper.

    Returns cleaned items (title, snippet, link, date, source). Dedupes by URL
    within this response. Returns ``[]`` on failure instead of raising.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        logger.warning("SERPER_API_KEY not set; returning no news")
        return []

    if not query.strip():
        return []

    try:
        response = requests.post(
            URL,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query.strip(), "num": num, "tbs": f"qdr:d{days}"},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("serper request failed for %r: %s", query, exc)
        return []

    return _clean_items(data.get("news") or [])


def fetch_news_for_queries(
    queries: list[str],
    *,
    num_per_query: int = 5,
    max_items: int = 8,
    days: int = 14,
) -> list[dict]:
    """Run Serper for each query and merge results, deduping by URL across queries."""
    seen_links: set[str] = set()
    merged: list[dict] = []
    for query in queries:
        if len(merged) >= max_items:
            break
        for item in getNews(query, num=num_per_query, days=days):
            link = (item.get("link") or "").strip()
            if link and link in seen_links:
                continue
            if link:
                seen_links.add(link)
            merged.append(item)
            if len(merged) >= max_items:
                break
    return merged


def format_news_for_prompt(items: list[dict]) -> str:
    """Format news items for the forecaster's News Summaries slot."""
    if not items:
        return "News Summaries: (none available)"
    lines = ["News Summaries:"]
    for i, item in enumerate(items, 1):
        src = item.get("source") or "?"
        date = item.get("date") or "?"
        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        link = (item.get("link") or "").strip()
        lines.append(f"[{i}] ({src}, {date}) {title}")
        if snippet:
            lines.append(f"    {snippet}")
        if link:
            lines.append(f"    Source: {link}")
    return "\n".join(lines)


if __name__ == "__main__":
    qry = input("News: ")
    print(json.dumps(getNews(qry), indent=2))
