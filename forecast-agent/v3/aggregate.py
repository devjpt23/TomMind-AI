"""Combine parallel forecaster votes into one probability."""

from __future__ import annotations

import statistics

from v3.config import (
    aggregation,
    disagreement_shrink_strength,
    disagreement_threshold,
)
from v3.state import ForecastVote


def _successful_values(votes: list[ForecastVote]) -> list[float]:
    ok = [v for v in votes if v.get("ok") and v.get("p_raw") is not None]
    return [float(v["p_raw"]) for v in ok]


def shrink_for_disagreement(p: float, values: list[float]) -> tuple[float, str]:
    """Extra shrink toward 0.5 when ensemble members disagree."""
    if len(values) < 2:
        return p, ""
    spread = max(values) - min(values)
    threshold = disagreement_threshold()
    if spread <= threshold:
        return p, ""
    strength = disagreement_shrink_strength()
    excess = min(1.0, (spread - threshold) / max(1e-6, 1.0 - threshold))
    factor = 1.0 - strength * excess
    p2 = 0.5 + factor * (p - 0.5)
    note = f"disagreement spread={spread:.2f} shrink→{p2:.3f}"
    return max(0.01, min(0.99, p2)), note


def aggregate_p_yes(
    votes: list[ForecastVote],
    *,
    method: str | None = None,
) -> tuple[float, str, float | None]:
    """Return ``(p_ensemble, rationale, vote_spread)``."""
    agg = method or aggregation()
    ok = [v for v in votes if v.get("ok") and v.get("p_raw") is not None]
    if not ok:
        return 0.5, "ensemble: no successful votes; fallback 0.5", None

    values = _successful_values(votes)
    spread = max(values) - min(values) if len(values) >= 2 else None

    if agg == "mean":
        p = statistics.fmean(values)
        label = "mean"
    else:
        p = statistics.median(values)
        label = "median"

    p, disagree_note = shrink_for_disagreement(p, values)
    parts = [
        f"{v.get('model_id', '?')}[{v.get('prompt_id', '?')}]={v['p_raw']:.3f}"
        for v in ok
    ]
    rationale = f"ensemble ({label} of {len(ok)}): " + ", ".join(parts) + f" → {p:.3f}"
    if disagree_note:
        rationale += f" | {disagree_note}"
    return max(0.01, min(0.99, p)), rationale, spread
