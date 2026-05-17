"""Market prior: Kalshi first, Polymarket title search as fallback."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import kalshi_prices
import polymarket

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


def fetch_market_context(*, market_ticker: str, title: str) -> MarketContext:
    """Kalshi by ticker, then Polymarket search by title if enabled."""
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

    if os.getenv("POLYMARKET_FALLBACK", "true").lower() in ("1", "true", "yes"):
        p_poly = polymarket.fetch_yes_price_by_search(title)
        if p_poly is not None:
            return MarketContext(
                source="polymarket",
                p_yes=p_poly,
                prompt_block=(
                    f"Polymarket search fallback (Kalshi ticker not found):\n"
                    f"  Implied P(Yes): {p_poly:.3f}"
                ),
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
