"""Multi-outcome probability forecasts (Prophet Arena HTTP format)."""

from __future__ import annotations

import json
import logging
from typing import Any

import openRouter
import research
from event_context import build_event_text

logger = logging.getLogger(__name__)

DIST_SYSTEM = """You are an expert forecaster estimating a probability distribution over mutually exclusive outcomes.

Rules:
- Use the exact outcome labels provided.
- Probabilities must be decimals in [0, 1] and sum to 1.
- Avoid overconfidence unless evidence is strong.

Respond with ONLY valid JSON:
{"probabilities": [{"market": "<label>", "probability": <float>}, ...]}"""


def _parse_distribution_json(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model response")
    data = json.loads(text[start : end + 1])
    raw = data["probabilities"]
    if isinstance(raw, dict):
        return [{"market": k, "probability": v} for k, v in raw.items()]
    if not isinstance(raw, list):
        raise TypeError("probabilities must be a list or object")
    return raw


def _normalize(
    raw: list[dict[str, Any]], expected: list[str]
) -> list[tuple[str, float]]:
    expected_set = set(expected)
    values: list[tuple[str, float]] = []
    for item in raw:
        market = str(item["market"])
        if expected_set and market not in expected_set:
            continue
        p = float(item["probability"])
        if p > 1.0:
            p /= 100.0
        values.append((market, max(0.0, min(1.0, p))))

    if not values and expected:
        u = 1.0 / len(expected)
        return [(m, u) for m in expected]

    if len(values) < len(expected):
        have = {m for m, _ in values}
        missing = [m for m in expected if m not in have]
        remainder = max(0.0, 1.0 - sum(p for _, p in values))
        share = remainder / len(missing) if missing else 0.0
        for m in missing:
            values.append((m, share))

    total = sum(p for _, p in values)
    if total <= 0:
        u = 1.0 / len(expected)
        return [(m, u) for m in expected]
    return [(m, p / total) for m, p in values]


def forecast_distribution(
    *,
    title: str,
    close_time: str,
    markets: list[str],
    subtitle: str | None = None,
    description: str | None = None,
    category: str | None = None,
    rules: str | None = None,
) -> tuple[list[tuple[str, float]], str]:
    """Return (market, probability) pairs and rationale text."""
    event_text = build_event_text(
        title=title,
        close_time=close_time,
        subtitle=subtitle,
        description=description,
        category=category,
        rules=rules,
    )
    try:
        news_block = research.gather_news(event_text, close_time, category=category)
    except Exception as exc:
        logger.warning("news gather failed for multi-outcome: %s", exc)
        news_block = "News Summaries: (unavailable)"
    user = (
        f"{event_text}\n\n"
        f"Mutually exclusive outcomes: {', '.join(markets)}\n\n"
        f"{news_block}\n\n"
        "Assign a probability to each outcome. Use the exact labels above."
    )

    client = openRouter._get_client()
    if client is None:
        u = 1.0 / len(markets)
        return [(m, u) for m in markets], "Fallback: no API key; uniform distribution."

    try:
        response = client.chat.completions.create(
            model=openRouter.FORECAST_MODEL,
            messages=[
                {"role": "system", "content": DIST_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_distribution_json(raw)
        dist = _normalize(parsed, markets)
        return dist, raw
    except Exception as exc:
        logger.warning("multi-outcome forecast failed: %s", exc)
        u = 1.0 / len(markets)
        return [(m, u) for m in markets], f"Fallback: {exc}; uniform distribution."
