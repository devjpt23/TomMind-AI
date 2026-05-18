"""Forecast agent server.

FastAPI agent that receives events from ``prophet forecast predict`` and returns
a probability estimate.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import calibrate
import market_prior
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


class PredictionResponse(BaseModel):
    p_yes: float
    rationale: str


def _build_event_text(event: EventRequest) -> str:
    return build_event_text(
        title=event.title,
        close_time=event.close_time,
        subtitle=event.subtitle,
        description=event.description,
        category=event.category,
        rules=event.rules,
    )


def _predict_v2(event: EventRequest) -> PredictionResponse:
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
    return PredictionResponse(p_yes=p_yes, rationale=raw)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": FORECAST_VERSION}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(event: EventRequest) -> PredictionResponse:
    logger.info("predict %s :: %s (%s)", event.market_ticker, event.title, FORECAST_VERSION)
    try:
        if FORECAST_VERSION == "v3":
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
            logger.info(
                "%s p_ensemble=%s p_blend=%s p_yes=%.3f spread=%s p_sources=%s",
                event.market_ticker,
                out.get("p_ensemble"),
                out.get("p_blend"),
                out["p_yes"],
                out.get("vote_spread"),
                out.get("p_sources"),
            )
            return PredictionResponse(p_yes=out["p_yes"], rationale=out["rationale"])

        return await asyncio.to_thread(_predict_v2, event)
    except Exception:
        logger.exception("predict failed for %s", event.market_ticker)
        return PredictionResponse(
            p_yes=0.5,
            rationale="Fallback: upstream error; returning uninformative prior 0.5.",
        )


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
