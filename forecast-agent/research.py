"""Shared analyst + Serper research (v3 and optional v2)."""

from __future__ import annotations

import openRouter
import serperSearch
from event_context import forecast_as_of


def gather_news(event_text: str, close_time: str, *, analyst_model: str | None = None) -> str:
    """One research pass: query generation + news fetch anchored to market close."""
    as_of = forecast_as_of(close_time)
    analysis = openRouter.analyst(event_text, model=analyst_model)
    if not analysis.get("should_search", True):
        return serperSearch.format_news_for_prompt([])
    items = serperSearch.fetch_news_for_queries(
        analysis["queries"],
        as_of=as_of,
    )
    return serperSearch.format_news_for_prompt(items)
