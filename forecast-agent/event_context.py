"""Event text and forecast-as-of dates for v2/v3 pipelines."""

from __future__ import annotations

from datetime import UTC, datetime


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


def build_event_text(
    *,
    title: str,
    close_time: str,
    subtitle: str | None = None,
    description: str | None = None,
    category: str | None = None,
    rules: str | None = None,
) -> str:
    """Build forecaster/analyst context with honest as-of date (not always calendar today)."""
    as_of = forecast_as_of(close_time)
    parts = [
        f"Forecast as of: {as_of.strftime('%Y-%m-%d')} (UTC)",
        f"Market close: {close_time}",
        (
            "Use only information that would have been knowable on or before the forecast "
            "date. Ignore later outcomes, post-resolution reporting, and spoilers."
        ),
        f"Event: {title}",
    ]
    if subtitle:
        parts.append(f"Subtitle: {subtitle}")
    if description:
        parts.append(f"Description: {description}")
    if rules:
        parts.append(f"Resolution rules: {rules}")
    if category:
        parts.append(f"Category: {category}")
    return "\n".join(parts)
