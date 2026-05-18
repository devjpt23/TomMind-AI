"""Shared analyst + Serper research (v3 and optional v2)."""

from __future__ import annotations

import openRouter
import serperSearch
from event_context import category_from_event_text, forecast_as_of


def _serper_queries(analysis: dict) -> list[str]:
    """Story queries + optional resolution query (resolution first when present)."""
    queries: list[str] = []
    rq = (analysis.get("resolution_query") or "").strip()
    if rq:
        queries.append(rq[:120])
    for q in analysis.get("queries") or []:
        if isinstance(q, str) and q.strip():
            q = q.strip()[:120]
            if q not in queries:
                queries.append(q)
    max_q = 4 if rq else 3
    return queries[:max_q]


def gather_news(
    event_text: str,
    close_time: str,
    *,
    analyst_model: str | None = None,
    category: str | None = None,
) -> str:
    """One research pass: query generation + news fetch anchored to market close."""
    as_of = forecast_as_of(close_time)
    cat = category or category_from_event_text(event_text)
    analysis = openRouter.analyst(
        event_text,
        model=analyst_model,
        category=cat,
    )
    if not analysis.get("should_search", True):
        return serperSearch.format_news_for_prompt([])

    queries = _serper_queries(analysis)
    if not queries:
        return serperSearch.format_news_for_prompt([])

    items = serperSearch.fetch_news_for_queries(queries, as_of=as_of)
    return serperSearch.format_news_for_prompt(items)
