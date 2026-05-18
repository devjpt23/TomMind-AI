"""Invoke the v3 LangGraph from FastAPI or local eval."""

from __future__ import annotations

from typing import Any

from event_context import build_event_text
from v3.config import max_concurrency
from v3.graph import GRAPH


def predict(
    *,
    market_ticker: str,
    title: str,
    close_time: str,
    subtitle: str | None = None,
    description: str | None = None,
    category: str | None = None,
    rules: str | None = None,
    event_text: str | None = None,
) -> dict[str, Any]:
    """Run the v3 graph synchronously. Returns ``p_yes``, ``rationale``, debug fields."""
    text = event_text or build_event_text(
        title=title,
        close_time=close_time,
        subtitle=subtitle,
        description=description,
        category=category,
        rules=rules,
    )
    initial = {
        "market_ticker": market_ticker,
        "title": title,
        "close_time": close_time,
        "category": category,
        "event_text": text,
        "forecasts": [],
    }
    config = {"configurable": {"max_concurrency": max_concurrency()}}
    final = GRAPH.invoke(initial, config)
    forecasts = final.get("forecasts", [])
    p_sources = [
        f"{v.get('model_id', '?')}:{v.get('p_source', '?')}"
        for v in forecasts
        if v.get("ok")
    ]
    return {
        "p_yes": final.get("p_yes", 0.5),
        "rationale": final.get("rationale", ""),
        "p_ensemble": final.get("p_ensemble"),
        "p_blend": final.get("p_blend"),
        "vote_spread": final.get("vote_spread"),
        "p_sources": p_sources,
        "forecasts": forecasts,
    }
