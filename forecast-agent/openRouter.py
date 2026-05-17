import json
import logging
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

FORECAST_MODEL = os.getenv("FORECAST_MODEL", "openai/gpt-4o")

ANALYST_SYSTEM = """You help forecasters find recent news for prediction markets.
Given an event, output 2-3 short Google News search queries (3-8 keywords each).
Do not use full questions — use keyword phrases (names, dates, topic).
If recent news is unlikely to help (historical macro already published, no live coverage), set should_search to false.
Reply with ONLY valid JSON: {"queries": ["q1", "q2"], "should_search": true}
Use at most 3 queries."""

FORECASTER_SYSTEM = """You are an expert superforecaster, familiar with Structured Analytic Techniques as well as Superforecasting by Philip Tetlock and related work.

Predict the probability that the event described in the user message will resolve true/yes. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES.

Use the event details and news summaries in the user message. Base your forecast on that information, not on unstated assumptions.

Output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end of the decimal (Ex: *0.42*)."""


def _fallback_query_from_event(event_text: str) -> str:
    for line in event_text.splitlines():
        if line.startswith("Event:"):
            return line.removeprefix("Event:").strip()[:120]
    return event_text.strip()[:120]


def _normalize_analyst_output(data: dict, fallback_query: str) -> dict:
    should_search = bool(data.get("should_search", True))
    queries: list[str] = []
    for q in data.get("queries") or []:
        if isinstance(q, str) and q.strip():
            queries.append(q.strip()[:120])
    queries = queries[:3]
    if should_search and not queries:
        queries = [fallback_query]
    if not should_search:
        queries = []
    return {"queries": queries, "should_search": should_search}


def analyst(event_text: str) -> dict:
    """Generate Serper search queries for a forecast event.

    Returns:
        {"queries": list[str], "should_search": bool}
    """
    fallback_query = _fallback_query_from_event(event_text)
    default = {"queries": [fallback_query], "should_search": True}

    try:
        response = client.chat.completions.create(
            model=FORECAST_MODEL,
            messages=[
                {"role": "system", "content": ANALYST_SYSTEM},
                {"role": "user", "content": event_text},
            ],
            temperature=0.3,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("analyst failed, using fallback: %s", exc)
        return default

    return _normalize_analyst_output(data, fallback_query)


def forecasterMain(user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=FORECAST_MODEL,
        messages=[
            {"role": "system", "content": FORECASTER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content or ""


def parse_p_yes(text: str, fallback: float = 0.5) -> float:
    """Extract p_yes from a forecaster response.

    Looks for the last `*<number>*` token. Clamps to [0.01, 0.99]. Returns ``fallback`` if no valid number
    is found.
    """
    if not text:
        return fallback
    matches = re.findall(r"\*\s*([0-9]*\.?[0-9]+)\s*\*", text)
    if not matches:
        return fallback
    try:
        p = float(matches[-1])
    except ValueError:
        return fallback
    return max(0.01, min(0.99, p))
