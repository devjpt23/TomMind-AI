"""Event text and forecast-as-of dates for v2/v3 pipelines."""

from __future__ import annotations

from datetime import UTC, datetime

RESOLUTION_TARGETED_CATEGORIES = frozenset(
    {
        "legal",
        "counting",
        "entertainment_elimination",
    }
)

COMMITMENT_ELIGIBLE_CATEGORIES = frozenset({"legal", "counting"})

_LEGAL_COUNTING_HINTS = (
    "supreme court",
    "scotus",
    "justices",
    "justice",
    "how many",
    "exactly",
    "vote count",
    "concurring",
    "margin of victory",
    "senators voted",
    "fed chair",
    "threshold",
    "at least",
    "at most",
    "ballots",
)

_ENTERTAINMENT_ELIMINATION_HINTS = (
    "eliminated",
    "elimination",
    "survivor",
    "masked singer",
    "vote off",
    "voted off",
    "roast subject",
    "tournament of champions",
)


def category_from_event_text(event_text: str) -> str | None:
    for line in event_text.splitlines():
        if line.startswith("Category:"):
            return line.removeprefix("Category:").strip() or None
    return None


def needs_resolution_query(category: str | None, event_text: str = "") -> bool:
    """Markets where a literal YES-condition search reduces contract misreads."""
    cat = (category or category_from_event_text(event_text) or "").strip().lower()
    cat_key = cat.replace(" ", "_").replace("-", "_")
    if cat_key in RESOLUTION_TARGETED_CATEGORIES:
        return True

    blob = event_text.lower()
    if cat_key in ("politics", "elections", "economics", "mentions", "other"):
        if any(h in blob for h in _LEGAL_COUNTING_HINTS):
            return True
    if cat_key == "entertainment":
        if any(h in blob for h in _ENTERTAINMENT_ELIMINATION_HINTS):
            return True
    return False


def commitment_eligible(category: str | None, event_text: str = "") -> bool:
    """Commitment nudge only when a determinable YES/NO answer is expected (not sports)."""
    cat = (category or category_from_event_text(event_text) or "").strip().lower()
    cat_key = cat.replace(" ", "_").replace("-", "_")
    if cat_key == "sports":
        return False
    return cat_key in COMMITMENT_ELIGIBLE_CATEGORIES


def parse_close_time(close_time: str) -> datetime:
    """Parse ISO close_time to UTC-aware datetime."""
    raw = close_time.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def forecast_as_of(close_time: str) -> datetime:
    """Point-in-time for forecasting: min(now, close_time)."""
    now = datetime.now(UTC)
    close = parse_close_time(close_time)
    return close if close < now else now


def _resolution_criteria(
    *,
    title: str,
    description: str | None,
    rules: str | None,
) -> str:
    crit = (rules or "").strip() or (description or "").strip() or title.strip()
    return crit


def _question_background(
    *,
    subtitle: str | None,
    description: str | None,
    rules: str | None,
) -> str:
    desc = (description or "").strip()
    crit = (rules or "").strip()
    parts: list[str] = []
    if subtitle and subtitle.strip():
        parts.append(subtitle.strip())
    if desc and desc != crit:
        parts.append(desc)
    if not parts:
        return "(none)"
    return "\n".join(parts)


def build_event_text(
    *,
    title: str,
    close_time: str,
    subtitle: str | None = None,
    description: str | None = None,
    category: str | None = None,
    rules: str | None = None,
) -> str:
    """Build forecaster/analyst context aligned with paper slot labels in system prompts."""
    as_of = forecast_as_of(close_time)
    criteria = _resolution_criteria(title=title, description=description, rules=rules)
    background = _question_background(
        subtitle=subtitle, description=description, rules=rules
    )

    lines = [
        (
            "Use only information that would have been knowable on or before Today's Date. "
            "Ignore later outcomes, post-resolution reporting, and spoilers."
        ),
        "",
        f"Question: {title.strip()}",
        "",
        "Question Background:",
        background,
        "",
        "Resolution Criteria:",
        f"**{criteria}**",
        "",
        "YES resolves to true only when the Resolution Criteria above are satisfied "
        "(not the headline alone).",
        f"**{criteria}**",
        "",
        (
            "Before forecasting, restate in one sentence what exactly must be true "
            "for this to resolve YES."
        ),
        (
            "Did you find a direct, explicit result confirming or denying the YES "
            "condition? Answer FOUND or NOT FOUND. If NOT FOUND, your probability "
            "must stay between 0.35 and 0.65."
        ),
        "",
        f"Today's Date: {as_of.strftime('%Y-%m-%d')} (UTC)",
        f"Close Time: {close_time}",
    ]
    if category and category.strip():
        lines.append(f"Category: {category.strip()}")
    return "\n".join(lines)
