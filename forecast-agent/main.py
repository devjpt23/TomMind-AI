"""Forecast agent server.

FastAPI agent that receives events from ``prophet forecast predict`` and returns
probability estimates in Prophet Arena format (``probabilities`` array).
"""

from __future__ import annotations

import asyncio
import logging
import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

import calibrate
import market_prior
import multi_forecast
import openRouter
import research
from event_context import build_event_text
from v3.runner import predict as predict_v3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Prophet Hacks Forecast Agent")

FORECAST_VERSION = os.getenv("FORECAST_VERSION", "v3").strip().lower()


class EventRequest(BaseModel):
    event_ticker: str
    market_ticker: str
    title: str
    subtitle: str | None = None
    description: str | None = None
    category: str
    rules: str | None = None
    close_time: str
    outcomes: list[str] | None = None


class MarketProbability(BaseModel):
    market: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    probabilities: list[MarketProbability]
    rationale: str | None = None


def _build_event_text(event: EventRequest) -> str:
    return build_event_text(
        title=event.title,
        close_time=event.close_time,
        subtitle=event.subtitle,
        description=event.description,
        category=event.category,
        rules=event.rules,
    )


def _event_markets(event: EventRequest) -> list[str]:
    if event.outcomes:
        return event.outcomes
    return [event.market_ticker]


def _predict_v2_binary(event: EventRequest) -> tuple[float, str]:
    """P(YES) for the market question (single binary leg)."""
    event_text = _build_event_text(event)
    ctx = market_prior.fetch_market_context(market_ticker=event.market_ticker)
    news_block = research.gather_news(
        event_text, event.close_time, category=event.category
    )
    prompt = f"{event_text}\n\n{ctx.prompt_block}\n\n{news_block}"
    parsed = openRouter.forecast_and_parse(prompt, category=event.category)
    p_raw = parsed["p_yes"]
    p_source = parsed["p_source"]
    raw = parsed["raw"]
    p_blended = market_prior.blend_with_market(p_raw, ctx.p_yes)
    p_yes = calibrate.calibrate(p_blended)
    logger.info(
        "%s p_raw=%.3f p_source=%s market=%s",
        event.market_ticker,
        p_raw,
        p_source,
        ctx.source,
    )
    return p_yes, raw


def _predict_v2(event: EventRequest) -> PredictionResponse:
    markets = _event_markets(event)

    if len(markets) > 2:
        dist, raw = multi_forecast.forecast_distribution(
            title=event.title,
            close_time=event.close_time,
            markets=markets,
            subtitle=event.subtitle,
            description=event.description,
            category=event.category,
            rules=event.rules,
        )
        return PredictionResponse(
            probabilities=[
                MarketProbability(market=m, probability=p) for m, p in dist
            ],
            rationale=raw,
        )

    p_yes, raw = _predict_v2_binary(event)
    if len(markets) == 1:
        probs = [MarketProbability(market=markets[0], probability=p_yes)]
    else:
        probs = [
            MarketProbability(market=markets[0], probability=p_yes),
            MarketProbability(market=markets[1], probability=max(0.0, 1.0 - p_yes)),
        ]
    return PredictionResponse(probabilities=probs, rationale=raw)


def _uniform_fallback(event: EventRequest, note: str) -> PredictionResponse:
    markets = _event_markets(event)
    u = 1.0 / len(markets)
    return PredictionResponse(
        probabilities=[
            MarketProbability(market=m, probability=u) for m in markets
        ],
        rationale=note,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": FORECAST_VERSION}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(event: EventRequest) -> PredictionResponse:
    logger.info("predict %s :: %s (%s)", event.market_ticker, event.title, FORECAST_VERSION)
    markets = _event_markets(event)
    try:
        if FORECAST_VERSION == "v3" and len(markets) <= 2:
            out = await asyncio.to_thread(
                predict_v3,
                market_ticker=event.market_ticker,
                title=event.title,
                close_time=event.close_time,
                subtitle=event.subtitle,
                description=event.description,
                category=event.category,
                rules=event.rules,
            )
            p_yes = float(out["p_yes"])
            logger.info(
                "%s p_ensemble=%s p_blend=%s p_yes=%.3f spread=%s p_sources=%s",
                event.market_ticker,
                out.get("p_ensemble"),
                out.get("p_blend"),
                p_yes,
                out.get("vote_spread"),
                out.get("p_sources"),
            )
            if len(markets) == 1:
                probs = [MarketProbability(market=markets[0], probability=p_yes)]
            else:
                probs = [
                    MarketProbability(market=markets[0], probability=p_yes),
                    MarketProbability(
                        market=markets[1], probability=max(0.0, 1.0 - p_yes)
                    ),
                ]
            return PredictionResponse(
                probabilities=probs, rationale=out.get("rationale", "")
            )

        return await asyncio.to_thread(_predict_v2, event)
    except Exception:
        logger.exception("predict failed for %s", event.market_ticker)
        return _uniform_fallback(
            event,
            "Fallback: upstream error; returning uniform distribution.",
        )


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
