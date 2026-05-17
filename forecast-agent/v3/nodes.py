"""LangGraph node functions for v3 (wrap v2 I/O modules)."""

from __future__ import annotations

import logging

import calibrate
import market_prior
import openRouter
import research
from v3.aggregate import aggregate_p_yes
from v3.config import analyst_model, ensemble_members
from v3.state import ForecastVote, GraphState

logger = logging.getLogger(__name__)


def prepare_context(state: GraphState) -> dict:
    """Load market prior and ensemble member list (model + prompt per agent)."""
    ctx = market_prior.fetch_market_context(market_ticker=state["market_ticker"])
    members = [
        {"model_id": m.model_id, "prompt_id": m.prompt_id}
        for m in ensemble_members()
    ]
    return {
        "market_block": ctx.prompt_block,
        "p_market": ctx.p_yes,
        "ensemble_members": members,
    }


def shared_research(state: GraphState) -> dict:
    """Single analyst + Serper pass (close_time-anchored window)."""
    close_time = state["close_time"]
    event_text = state["event_text"]
    news_block = research.gather_news(
        event_text,
        close_time,
        analyst_model=analyst_model(),
    )
    return {"news_block": news_block}


def dispatch_forecasters(state: GraphState):
    """Map: one Send per ensemble member (parallel forecast only)."""
    from langgraph.types import Send

    payload_base = {
        "event_text": state["event_text"],
        "market_block": state["market_block"],
        "news_block": state["news_block"],
        "market_ticker": state["market_ticker"],
    }
    return [
        Send(
            "forecast_one",
            {
                **payload_base,
                "model_id": member["model_id"],
                "prompt_id": member["prompt_id"],
            },
        )
        for member in state["ensemble_members"]
    ]


def forecast_one(state: GraphState) -> dict:
    """Per-member forecaster on shared news. Never raises (keeps superstep alive)."""
    model_id = state["model_id"]
    prompt_id = state.get("prompt_id") or "superforecaster"
    vote: ForecastVote = {
        "model_id": model_id,
        "prompt_id": prompt_id,
        "p_raw": None,
        "rationale": "",
        "ok": False,
        "error": None,
    }
    try:
        prompt = (
            f"{state['event_text']}\n\n"
            f"{state['market_block']}\n\n"
            f"{state['news_block']}"
        )
        raw = openRouter.forecaster(prompt, model=model_id, prompt_id=prompt_id)
        p_raw = openRouter.parse_p_yes(raw)
        vote["p_raw"] = p_raw
        vote["rationale"] = raw[:500]
        vote["ok"] = True
        logger.info("v3 %s[%s] p_raw=%.3f", model_id, prompt_id, p_raw)
    except Exception as exc:
        vote["error"] = str(exc)
        logger.warning(
            "v3 forecast_one failed model=%s prompt=%s: %s",
            model_id,
            prompt_id,
            exc,
        )
    return {"forecasts": [vote]}


def aggregate_forecasts(state: GraphState) -> dict:
    votes = state.get("forecasts") or []
    p_ensemble, rationale, spread = aggregate_p_yes(votes)
    return {
        "p_ensemble": p_ensemble,
        "vote_spread": spread,
        "rationale": rationale,
    }


def blend_and_calibrate(state: GraphState) -> dict:
    p_ensemble = state.get("p_ensemble")
    if p_ensemble is None:
        p_ensemble = 0.5
    p_blend = market_prior.blend_ensemble_with_market(
        p_ensemble,
        state.get("p_market"),
        vote_spread=state.get("vote_spread"),
    )
    p_yes = calibrate.calibrate(p_blend)
    rationale = state.get("rationale") or ""
    spread = state.get("vote_spread")
    spread_s = f"{spread:.3f}" if spread is not None else "n/a"
    rationale = f"{rationale} | spread={spread_s} blend={p_blend:.3f} final={p_yes:.3f}"
    return {"p_blend": p_blend, "p_yes": p_yes, "rationale": rationale}
