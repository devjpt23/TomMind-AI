import json
import logging
import os
import re
from typing import Any, Literal

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

Output JSON only:
{
  "queries": ["story query 1", "story query 2"],
  "resolution_query": null,
  "should_search": true
}

- queries: 2-3 short Google News keyword phrases (3-8 words). Focus on what happened (names, dates, event).
- resolution_query: null unless the user message says resolution-targeted search is REQUIRED.
  When required: one keyword phrase for whether the exact YES condition in Resolution Criteria was met
  (vote counts, exact thresholds, who was eliminated per rules — not the headline story alone).
- should_search: false if recent news is unlikely to help.

Use at most 3 entries in queries."""

ANALYST_RESOLUTION_NOTE = (
    "Resolution-targeted search: REQUIRED. "
    "Fill resolution_query with a search for the literal YES condition in Resolution Criteria "
    "(counts, exact outcomes, elimination per rules) — not just the headline topic."
)

RESOLUTION_RESTATE_INSTRUCTION = (
    "Before forecasting, restate in one sentence what exactly must be true for "
    "this to resolve YES. Use Resolution Criteria only — not the headline."
)

EVIDENCE_CHECK_INSTRUCTION = (
    "Did you find a direct, explicit result confirming or denying the YES condition? "
    "Answer FOUND or NOT FOUND. If NOT FOUND, your probability must stay between 0.35 and 0.65."
)

# Turtel et al. 2502.05253 Figure 4 — DeepSeek-R1 14B zero-shot prompt (verbatim).
FORECASTER_SYSTEM = f"""You are an expert superforecaster, familiar with Structured Analytic Techniques as well as Superforecasting by Philip Tetlock and related work. Predict the probability that the following question will be resolved as true/yes. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES.

The user message provides: Question, Question Background, Resolution Criteria, Today's/Question Close Date, and News Summaries.

{RESOLUTION_RESTATE_INSTRUCTION}

{EVIDENCE_CHECK_INSTRUCTION}

Output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end of the decimal (Ex: *<probability>*)."""

BASE_RATE_FORECASTER_SYSTEM = f"""You are a superforecaster emphasizing the OUTSIDE VIEW (reference class / base rate).

{RESOLUTION_RESTATE_INSTRUCTION}

{EVIDENCE_CHECK_INSTRUCTION}

Before using headlines, estimate how often similar events in this category resolve YES.
Then update modestly using only the news summaries provided — do not overweight vivid stories.

Predict P(YES). You MUST output a probability between 0 and 1.
Avoid extremes (below 0.05 or above 0.95) unless the evidence is overwhelming.

Output only your final probability with asterisks: *0.42*"""

PHI4_SCRATCHPAD_FORECASTER_SYSTEM = f"""The user message provides: Question, Question Background, Resolution Criteria, Today's/Question Close Date, and News Summaries.

Instructions:
1. {RESOLUTION_RESTATE_INSTRUCTION}
Insert your one-sentence YES condition.
2. {EVIDENCE_CHECK_INSTRUCTION}
Insert FOUND or NOT FOUND and brief justification (direct final score, vote count, etc. — not adjacent stories like relegation or chaos).
3. Given the above question, rephrase and expand it to help you do better answering. Maintain all information in the original question.
Insert rephrased and expanded question.
4. Using your knowledge of the world and topic, as well as the information provided, provide a few reasons why the answer might be no. Rate the strength of each reason.
Insert your thoughts
5. Using your knowledge of the world and topic, as well as the information provided, provide a few reasons why the answer might be yes. Rate the strength of each reason.
Insert your thoughts
6. Aggregate your considerations. Think like a superforecaster (e.g. Nate Silver).
Insert your aggregated considerations
7. Output an initial probability (prediction) given steps 3–6.
Insert initial probability.
8. Evaluate whether your calculated probability is excessively confident or not confident enough. Also, consider anything else that might affect the forecast that you did not before consider (e.g. base rate of the event).
Insert your thoughts
9. Output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
Insert your answer"""

STRUCTURED_7STEP_FORECASTER_SYSTEM = PHI4_SCRATCHPAD_FORECASTER_SYSTEM

FORECASTER_PROMPTS: dict[str, str] = {
    "superforecaster": FORECASTER_SYSTEM,
    "base_rate": BASE_RATE_FORECASTER_SYSTEM,
    "phi4_scratchpad": PHI4_SCRATCHPAD_FORECASTER_SYSTEM,
    "structured_7step": PHI4_SCRATCHPAD_FORECASTER_SYSTEM,
}


def _fallback_query_from_event(event_text: str) -> str:
    for line in event_text.splitlines():
        if line.startswith("Question:"):
            return line.removeprefix("Question:").strip()[:120]
        if line.startswith("Event:"):
            return line.removeprefix("Event:").strip()[:120]
    return event_text.strip()[:120]


def _normalize_analyst_output(
    data: dict,
    fallback_query: str,
    *,
    want_resolution_query: bool,
) -> dict:
    should_search = bool(data.get("should_search", True))
    queries: list[str] = []
    for q in data.get("queries") or []:
        if isinstance(q, str) and q.strip():
            queries.append(q.strip()[:120])
    queries = queries[:3]

    resolution_query = ""
    raw_rq = data.get("resolution_query")
    if want_resolution_query and isinstance(raw_rq, str) and raw_rq.strip():
        resolution_query = raw_rq.strip()[:120]

    if should_search and not queries and not resolution_query:
        queries = [fallback_query]
    if not should_search:
        queries = []
        resolution_query = ""

    return {
        "queries": queries,
        "resolution_query": resolution_query,
        "should_search": should_search,
    }


def analyst(
    event_text: str,
    *,
    model: str | None = None,
    category: str | None = None,
) -> dict:
    """Generate Serper search queries for a forecast event.

    Returns:
        {"queries": list[str], "resolution_query": str, "should_search": bool}
    """
    from event_context import category_from_event_text, needs_resolution_query

    fallback_query = _fallback_query_from_event(event_text)
    cat = category or category_from_event_text(event_text)
    want_resolution = needs_resolution_query(cat, event_text)
    default = {
        "queries": [fallback_query],
        "resolution_query": "",
        "should_search": True,
    }

    client = _get_client()
    if client is None:
        return default

    user_content = event_text
    if want_resolution:
        user_content = f"{event_text}\n\n{ANALYST_RESOLUTION_NOTE}"

    try:
        response = client.chat.completions.create(
            model=model or FORECAST_MODEL,
            messages=[
                {"role": "system", "content": ANALYST_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=280,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("analyst failed, using fallback: %s", exc)
        return default

    return _normalize_analyst_output(
        data, fallback_query, want_resolution_query=want_resolution
    )


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


PSource = Literal["model", "parser_fallback", "commitment", "evidence_clamp"]

EVIDENCE_NOT_FOUND_BAND = (0.35, 0.65)

FORMAT_RETRY_INSTRUCTION = (
    "You must output your probability as a number between asterisks like *0.73*. "
    "If you did not, output only the number now."
)

COMMITMENT_TEST_INSTRUCTION = (
    "You said 50%. Look at the news again. Is there any evidence at all that favors YES "
    "over NO, or NO over YES? If yes, move your estimate by at least 5 points in that "
    "direction and explain why. Output your updated probability with asterisks like *0.55*."
)

_ASTERISK_P_RE = re.compile(r"\*\s*([0-9]*\.?[0-9]+)\s*\*")
_BARE_P_RE = re.compile(
    r"(?:^|[\s:])((?:0?\.\d+)|(?:1(?:\.0+)?)|(?:0(?:\.0+)?))(?:\s|$|[\n\r])"
)


def _clamp_p(p: float) -> float:
    return max(0.01, min(0.99, p))


def _is_exact_half(p: float) -> bool:
    return abs(p - 0.5) < 1e-9


def _evidence_answer(text: str) -> str | None:
    """Last standalone FOUND / NOT FOUND in model output (ignores NOT FOUND substrings)."""
    last: tuple[str, int] | None = None
    for m in re.finditer(r"\bNOT\s+FOUND\b", text, re.I):
        last = ("NOT FOUND", m.start())
    for m in re.finditer(r"\bFOUND\b", text, re.I):
        start = m.start()
        prefix = text[max(0, start - 4) : start].upper()
        if prefix.rstrip().endswith("NOT"):
            continue
        last = ("FOUND", start)
    return last[0] if last else None


def _apply_evidence_band(text: str, p: float, src: PSource) -> tuple[float, PSource]:
    """Enforce 0.35–0.65 when the model says evidence was NOT FOUND."""
    if _evidence_answer(text) != "NOT FOUND":
        return p, src
    lo, hi = EVIDENCE_NOT_FOUND_BAND
    clamped = max(lo, min(hi, p))
    if abs(clamped - p) > 1e-9:
        logger.info("evidence NOT FOUND: clamped p %.3f -> %.3f", p, clamped)
        return clamped, "evidence_clamp"
    return p, src


def try_parse_p_yes(text: str, *, allow_bare_number: bool = False) -> tuple[float, PSource] | None:
    """Parse probability from forecaster text. Returns ``None`` if not found."""
    if not text or not text.strip():
        return None
    matches = _ASTERISK_P_RE.findall(text)
    if matches:
        try:
            return _clamp_p(float(matches[-1])), "model"
        except ValueError:
            pass
    if allow_bare_number:
        bare = _BARE_P_RE.findall(text.strip()[-80:])
        if bare:
            try:
                return _clamp_p(float(bare[-1])), "model"
            except ValueError:
                pass
    return None


def parse_p_yes(text: str, fallback: float = 0.5) -> float:
    """Extract p_yes (legacy). Prefer ``forecast_and_parse`` for ``p_source`` tracking."""
    parsed = try_parse_p_yes(text)
    if parsed:
        return parsed[0]
    return fallback


def _format_retry_completion(
    client: OpenAI,
    *,
    model: str,
    prior_raw: str,
) -> str:
    tail = prior_raw[-2500:] if len(prior_raw) > 2500 else prior_raw
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"{FORMAT_RETRY_INSTRUCTION}\n\nYour previous response:\n{tail}",
            },
        ],
        temperature=0.0,
        max_tokens=32,
    )
    return response.choices[0].message.content or ""


def _commitment_test_completion(
    client: OpenAI,
    *,
    model: str,
    user_prompt: str,
    prior_raw: str,
) -> str:
    tail = prior_raw[-2000:] if len(prior_raw) > 2000 else prior_raw
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": tail},
            {"role": "user", "content": COMMITMENT_TEST_INSTRUCTION},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content or ""


def _finalize_forecast(
    p: float,
    src: PSource,
    raw: str,
    *,
    client: OpenAI | None,
    model_id: str,
    prompt_id: str,
    user_prompt: str,
    category: str | None = None,
) -> dict[str, Any]:
    """Evidence band + optional commitment pass when the model hedged at exactly 50%."""
    from event_context import commitment_eligible

    p, src = _apply_evidence_band(raw, p, src)
    if (
        client is None
        or src not in ("model", "evidence_clamp")
        or not _is_exact_half(p)
        or not commitment_eligible(category, user_prompt)
    ):
        return {"p_yes": p, "p_source": src, "raw": raw}

    try:
        commit_raw = _commitment_test_completion(
            client,
            model=model_id,
            user_prompt=user_prompt,
            prior_raw=raw,
        )
    except Exception as exc:
        logger.warning("commitment test failed: %s", exc)
        return {"p_yes": p, "p_source": src, "raw": raw}

    combined = f"{raw}\n\n[commitment-test]\n{commit_raw}"
    for chunk in (commit_raw, combined):
        parsed = try_parse_p_yes(chunk)
        if parsed:
            cp, _ = parsed
            if not _is_exact_half(cp):
                logger.info("commitment nudge: p 0.500 -> %.3f", cp)
                return {"p_yes": cp, "p_source": "commitment", "raw": combined}
            break

    return {"p_yes": p, "p_source": src, "raw": combined if commit_raw else raw}


def forecast_and_parse(
    user_prompt: str,
    *,
    model: str | None = None,
    prompt_id: str = "superforecaster",
    category: str | None = None,
) -> dict[str, Any]:
    """Run forecaster, parse ``p_yes``, retry once on format failure.

    Returns:
        ``p_yes``, ``p_source`` (``model`` | ``commitment`` | ``parser_fallback``), ``raw``.
    """
    from event_context import category_from_event_text

    model_id = model or FORECAST_MODEL
    client = _get_client()
    cat = category or category_from_event_text(user_prompt)
    raw = forecaster(user_prompt, model=model_id, prompt_id=prompt_id)
    parsed = try_parse_p_yes(raw)
    if parsed:
        p, src = parsed
        return _finalize_forecast(
            p,
            src,
            raw,
            client=client,
            model_id=model_id,
            prompt_id=prompt_id,
            user_prompt=user_prompt,
            category=cat,
        )

    if client is None:
        logger.warning("parse_p_yes: no asterisk probability and no API client for retry")
        return {"p_yes": 0.5, "p_source": "parser_fallback", "raw": raw}

    try:
        retry_raw = _format_retry_completion(client, model=model_id, prior_raw=raw)
    except Exception as exc:
        logger.warning("format-retry LLM call failed: %s", exc)
        retry_raw = ""

    combined = raw
    if retry_raw:
        combined = f"{raw}\n\n[format-retry]\n{retry_raw}"

    for chunk, bare_ok in ((retry_raw, True), (combined, False)):
        if not chunk:
            continue
        parsed = try_parse_p_yes(chunk, allow_bare_number=bare_ok)
        if parsed:
            p, src = parsed
            logger.info("parse recovered p=%.3f after format-retry", p)
            return _finalize_forecast(
                p,
                src,
                combined,
                client=client,
                model_id=model_id,
                prompt_id=prompt_id,
                user_prompt=user_prompt,
                category=cat,
            )

    logger.warning("parser_fallback: no *probability* after format-retry")
    return {"p_yes": 0.5, "p_source": "parser_fallback", "raw": combined}
