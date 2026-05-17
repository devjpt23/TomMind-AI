import json
import logging
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

FORECAST_MODEL = os.getenv("FORECAST_MODEL", "openai/gpt-4o")


def _get_client() -> OpenAI | None:
    """Lazy client so the server can start before env vars are injected (e.g. App Platform)."""
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY is not set")
        return None
    _client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return _client

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

BASE_RATE_FORECASTER_SYSTEM = """You are a superforecaster emphasizing the OUTSIDE VIEW (reference class / base rate).

Before using headlines, estimate how often similar events in this category resolve YES.
Then update modestly using only the news summaries provided — do not overweight vivid stories.

Predict P(YES). You MUST output a probability between 0 and 1.
Avoid extremes (below 0.05 or above 0.95) unless the evidence is overwhelming.

Output only your final probability with asterisks: *0.42*"""

STRUCTURED_7STEP_FORECASTER_SYSTEM = """You are an expert forecaster. Work through these steps briefly, then give a final probability.

1. Rephrase the resolution question in one sentence.
2. List the 2 strongest reasons the market resolves NO.
3. List the 2 strongest reasons the market resolves YES.
4. Weigh evidence (Tetlock-style: neither anchor on hope nor fear).
5. State an initial probability.
6. Calibration check: could you be overconfident? Adjust if needed.
7. Final probability for YES.

You MUST end with a single number between 0 and 1 in asterisks (Ex: *0.42*)."""

FORECASTER_PROMPTS: dict[str, str] = {
    "superforecaster": FORECASTER_SYSTEM,
    "base_rate": BASE_RATE_FORECASTER_SYSTEM,
    "structured_7step": STRUCTURED_7STEP_FORECASTER_SYSTEM,
}


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


def analyst(event_text: str, *, model: str | None = None) -> dict:
    """Generate Serper search queries for a forecast event.

    Returns:
        {"queries": list[str], "should_search": bool}
    """
    fallback_query = _fallback_query_from_event(event_text)
    default = {"queries": [fallback_query], "should_search": True}

    client = _get_client()
    if client is None:
        return default

    try:
        response = client.chat.completions.create(
            model=model or FORECAST_MODEL,
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


def forecaster(
    user_prompt: str,
    *,
    model: str | None = None,
    prompt_id: str = "superforecaster",
) -> str:
    client = _get_client()
    if client is None:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    system = FORECASTER_PROMPTS.get(prompt_id, FORECASTER_SYSTEM)
    response = client.chat.completions.create(
        model=model or FORECAST_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content or ""


def forecasterMain(user_prompt: str) -> str:
    """Backward-compatible alias for v2 single-model forecaster."""
    return forecaster(user_prompt)


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
