import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

URL = "https://google.serper.dev/news"
KEEP_FIELDS = ("title", "snippet", "link", "date", "source")
_KEY_ENV_NAMES = ("SERPER_API_KEY", "SERPER2_API_KEY")
_QUOTA_HINTS = (
    "quota",
    "credit",
    "limit",
    "exhausted",
    "insufficient",
    "no credits",
    "out of",
    "billing",
)

_key_lock = threading.Lock()
_primary_exhausted = False


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


def _configured_keys() -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for name in _KEY_ENV_NAMES:
        value = (os.getenv(name) or "").strip()
        if value:
            keys.append((name, value))
    return keys


def _response_quota_exhausted(response: requests.Response) -> bool:
    if response.status_code in (402, 429):
        return True
    try:
        body = response.json()
        text = json.dumps(body).lower()
    except ValueError:
        text = (response.text or "").lower()
    return any(hint in text for hint in _QUOTA_HINTS)


def _mark_primary_exhausted() -> None:
    global _primary_exhausted
    with _key_lock:
        if not _primary_exhausted:
            _primary_exhausted = True
            logger.warning(
                "SERPER_API_KEY exhausted or over quota; using SERPER2_API_KEY for all further requests"
            )


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
    keys = _configured_keys()
    if not keys:
        logger.warning("No Serper API keys set (SERPER_API_KEY / SERPER2_API_KEY); returning no news")
        return []

    if not query.strip():
        return []

    if as_of is not None:
        tbs = tbs_for_window(end=as_of, days=days)
    else:
        tbs = f"qdr:d{days}"

    payload = {"q": query.strip(), "num": num, "tbs": tbs}

    with _key_lock:
        start_idx = 1 if _primary_exhausted and len(keys) > 1 else 0

    for key_name, api_key in keys[start_idx:]:
        try:
            response = requests.post(
                URL,
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            logger.warning("serper request failed for %r (%s): %s", query, key_name, exc)
            return []

        if response.ok:
            try:
                data = response.json()
            except ValueError as exc:
                logger.warning("serper invalid JSON for %r (%s): %s", query, key_name, exc)
                return []
            return _clean_items(data.get("news") or [])

        if (
            _response_quota_exhausted(response)
            and key_name == "SERPER_API_KEY"
            and len(keys) > 1
        ):
            _mark_primary_exhausted()
            continue
        logger.warning(
            "serper request failed for %r (%s): HTTP %s %s",
            query,
            key_name,
            response.status_code,
            (response.text or "")[:200],
        )
        return []

    return []


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
