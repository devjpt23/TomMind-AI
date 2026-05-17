"""Market prior from Kalshi public API (by market_ticker)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import kalshi_prices

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketContext:
    source: str
    p_yes: float | None
    prompt_block: str

    @staticmethod
    def empty() -> MarketContext:
        return MarketContext(
            source="none",
            p_yes=None,
            prompt_block="Market prices: (none available)",
        )


def fetch_market_context(*, market_ticker: str) -> MarketContext:
    """Fetch Kalshi snapshot by ticker; empty context if unavailable."""
    snap = kalshi_prices.fetch_market_snapshot(market_ticker)
    if snap and snap.p_yes is not None:
        return MarketContext(
            source="kalshi",
            p_yes=snap.p_yes,
            prompt_block=snap.prompt_block(),
        )
    if snap:
        return MarketContext(
            source="kalshi",
            p_yes=None,
            prompt_block=snap.prompt_block(),
        )

    return MarketContext.empty()


def blend_with_market(p_model: float, p_market: float | None) -> float:
    """Blend model probability with market prior (Jibang-style selective + optional weight)."""
    if p_market is None:
        return p_model

    threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.15"))
    weight = float(os.getenv("MARKET_PRIOR_WEIGHT", "0.0"))

    if abs(p_model - 0.5) < threshold:
        logger.info(
            "low model confidence (|p-0.5|=%.3f < %.3f); using market prior %.3f",
            abs(p_model - 0.5),
            threshold,
            p_market,
        )
        return p_market

    if weight > 0:
        blended = (1.0 - weight) * p_model + weight * p_market
        logger.info(
            "market blend weight=%.2f: model=%.3f market=%.3f -> %.3f",
            weight,
            p_model,
            p_market,
            blended,
        )
        return blended

    return p_model
