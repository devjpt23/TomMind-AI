"""Read Kalshi market prices from the public Trade API (no API key required)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.elections.kalshi.com"
TIMEOUT = 8.0


@dataclass(frozen=True)
class KalshiMarketSnapshot:
    market_ticker: str
    title: str
    status: str | None
    yes_bid: float | None
    yes_ask: float | None
    last_price: float | None
    p_yes: float | None

    def prompt_block(self) -> str:
        if self.p_yes is None:
            return "Kalshi market data: (unavailable for this ticker)"
        lines = [
            "Kalshi market snapshot:",
            f"  Implied P(Yes) used: {self.p_yes:.3f}",
        ]
        if self.yes_bid is not None and self.yes_ask is not None:
            lines.append(f"  Yes bid: {self.yes_bid:.3f} | Yes ask: {self.yes_ask:.3f}")
        if self.last_price is not None:
            lines.append(f"  Last trade: {self.last_price:.3f}")
        if self.status:
            lines.append(f"  Status: {self.status}")
        lines.append(
            "Treat this as the crowd's current price; update only if news strongly disagrees."
        )
        return "\n".join(lines)


def _base_url() -> str:
    return os.getenv("KALSHI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _parse_dollar(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def implied_yes_price(yes_bid: float | None, yes_ask: float | None, last_price: float | None) -> float | None:
    """Midpoint of bid/ask when possible; else ask, else last."""
    if yes_bid is not None and yes_ask is not None and yes_ask > 0:
        return max(0.01, min(0.99, (yes_bid + yes_ask) / 2.0))
    if yes_ask is not None and yes_ask > 0:
        return max(0.01, min(0.99, yes_ask))
    if yes_bid is not None and yes_bid > 0:
        return max(0.01, min(0.99, yes_bid))
    if last_price is not None and last_price > 0:
        return max(0.01, min(0.99, last_price))
    return None


def fetch_market_snapshot(market_ticker: str) -> KalshiMarketSnapshot | None:
    """Fetch a single market by ticker. Returns None if not found or on error."""
    ticker = market_ticker.strip()
    if not ticker:
        return None
    path = f"/trade-api/v2/markets/{ticker}"
    url = _base_url() + path
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        raw = resp.json().get("market", resp.json())
    except requests.RequestException as exc:
        logger.warning("kalshi fetch failed for %s: %s", ticker, exc)
        return None

    yes_bid = _parse_dollar(raw.get("yes_bid_dollars") or raw.get("yes_bid"))
    yes_ask = _parse_dollar(raw.get("yes_ask_dollars") or raw.get("yes_ask"))
    last_price = _parse_dollar(raw.get("last_price_dollars") or raw.get("last_price"))
    p_yes = implied_yes_price(yes_bid, yes_ask, last_price)

    return KalshiMarketSnapshot(
        market_ticker=ticker,
        title=str(raw.get("title") or ""),
        status=raw.get("status"),
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        last_price=last_price,
        p_yes=p_yes,
    )
