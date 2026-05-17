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
import serperSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Prophet Hacks Forecast Agent")


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
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    parts = [
        f"Today's Date: {today}",
        f"Event: {event.title}",
    ]
    if event.subtitle:
        parts.append(f"Subtitle: {event.subtitle}")
    if event.description:
        parts.append(f"Description: {event.description}")
    if event.rules:
        parts.append(f"Rules: {event.rules}")
    parts.append(f"Category: {event.category}")
    parts.append(f"Close time: {event.close_time}")
    return "\n".join(parts)


def _build_forecaster_prompt(
    event: EventRequest,
    news_block: str,
    market_block: str,
) -> str:
    return f"{_build_event_text(event)}\n\n{market_block}\n\n{news_block}"


def _gather_news(event_text: str) -> str:
    analysis = openRouter.analyst(event_text)
    if not analysis.get("should_search", True):
        return serperSearch.format_news_for_prompt([])
    items = serperSearch.fetch_news_for_queries(analysis["queries"])
    return serperSearch.format_news_for_prompt(items)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "v2"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(event: EventRequest) -> PredictionResponse:
    logger.info("predict %s :: %s", event.market_ticker, event.title)
    try:
        event_text = _build_event_text(event)
        ctx = await asyncio.to_thread(
            market_prior.fetch_market_context,
            market_ticker=event.market_ticker,
        )
        logger.info(
            "%s market source=%s p_market=%s",
            event.market_ticker,
            ctx.source,
            f"{ctx.p_yes:.3f}" if ctx.p_yes is not None else "n/a",
        )
        news_block = await asyncio.to_thread(_gather_news, event_text)
        prompt = _build_forecaster_prompt(event, news_block, ctx.prompt_block)
        raw = await asyncio.to_thread(openRouter.forecasterMain, prompt)
        p_raw = openRouter.parse_p_yes(raw)
        p_blended = market_prior.blend_with_market(p_raw, ctx.p_yes)
        p_yes = calibrate.calibrate(p_blended)
        logger.info(
            "%s p_raw=%.3f p_blend=%.3f p_yes=%.3f",
            event.market_ticker,
            p_raw,
            p_blended,
            p_yes,
        )
        return PredictionResponse(p_yes=p_yes, rationale=raw)
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
