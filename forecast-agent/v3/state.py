"""Graph state schema for the v3 ensemble pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ForecastVote(TypedDict, total=False):
    model_id: str
    prompt_id: str
    p_raw: float | None
    p_source: str
    rationale: str
    ok: bool
    error: str | None


class GraphState(TypedDict, total=False):
    # Input (set before invoke)
    market_ticker: str
    title: str
    close_time: str
    category: str | None
    event_text: str
    market_block: str
    p_market: float | None
    news_block: str
    ensemble_members: list[dict]  # serialized EnsembleMember

    # Fan-out dispatch
    model_id: str
    prompt_id: str

    # Fan-in via reducer (parallel forecast_one nodes)
    forecasts: Annotated[list[ForecastVote], operator.add]

    # Aggregation / output
    p_ensemble: float | None
    vote_spread: float | None
    p_blend: float | None
    p_yes: float | None
    rationale: str
