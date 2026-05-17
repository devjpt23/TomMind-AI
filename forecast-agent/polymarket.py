"""Optional Polymarket fallback prices via public Gamma API (no API key)."""

from __future__ import annotations

import json
import logging
import re

import requests

logger = logging.getLogger(__name__)

GAMMA_SEARCH = "https://gamma-api.polymarket.com/public-search"
TIMEOUT = 10.0


def _parse_prices(market: dict) -> float | None:
    raw_prices = market.get("outcomePrices")
    raw_outcomes = market.get("outcomes")
    if not raw_prices or not raw_outcomes:
        return None
    prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
    outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
    for outcome, price in zip(outcomes, prices, strict=False):
        if str(outcome).strip().lower() in ("yes", "true"):
            try:
                return max(0.01, min(0.99, float(price)))
            except (TypeError, ValueError):
                return None
    try:
        return max(0.01, min(0.99, float(prices[0])))
    except (TypeError, ValueError, IndexError):
        return None


def fetch_yes_price_by_search(title: str, *, limit: int = 5) -> float | None:
    q = re.sub(r"\s+", " ", title).strip()[:120]
    if not q:
        return None
    try:
        resp = requests.get(
            GAMMA_SEARCH,
            params={"q": q, "limit": limit, "events_status": "all"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.warning("polymarket search failed: %s", exc)
        return None

    q_lower = q.lower()
    fallback: float | None = None
    for event in data.get("events") or []:
        for market in event.get("markets") or []:
            question = (market.get("question") or market.get("title") or "").lower()
            price = _parse_prices(market)
            if price is None:
                continue
            if q_lower[:40] in question or question[:40] in q_lower:
                return price
            if fallback is None:
                fallback = price
    return fallback
