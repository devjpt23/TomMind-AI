import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta

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


def tbs_for_window(*, end: datetime, days: int = 14) -> str:
    """Google/Serper custom date range ending at ``end`` (inclusive window)."""
    end = end.astimezone(UTC)
    start = end - timedelta(days=days)
    return (
        f"cdr:1,cd_min:{start.month}/{start.day}/{start.year},"
        f"cd_max:{end.month}/{end.day}/{end.year}"
    )


def getNews(
    query: str,
    num: int = 10,
    days: int = 14,
    timeout: float = 8.0,
    *,
    as_of: datetime | None = None,
) -> list[dict]:
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

    if as_of is not None:
        tbs = tbs_for_window(end=as_of, days=days)
    else:
        tbs = f"qdr:d{days}"

    try:
        response = requests.post(
            URL,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query.strip(), "num": num, "tbs": tbs},
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
    as_of: datetime | None = None,
) -> list[dict]:
    """Run Serper for each query (parallel) and merge, deduping by URL across queries."""
    if not queries:
        return []

    workers = min(
        max(1, int(os.getenv("SERPER_MAX_WORKERS", "3"))),
        len(queries),
    )
    seen_links: set[str] = set()
    merged: list[dict] = []

    def _fetch_one(query: str) -> list[dict]:
        return getNews(query, num=num_per_query, days=days, as_of=as_of)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, q): q for q in queries}
        for fut in as_completed(futures):
            try:
                batch = fut.result()
            except Exception as exc:
                logger.warning("serper parallel fetch failed: %s", exc)
                continue
            for item in batch:
                if len(merged) >= max_items:
                    break
                link = (item.get("link") or "").strip()
                if link and link in seen_links:
                    continue
                if link:
                    seen_links.add(link)
                merged.append(item)
    return merged[:max_items]


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
