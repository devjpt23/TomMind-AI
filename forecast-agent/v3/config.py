"""V3 runtime configuration (env-driven)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from v3.prompts import DEFAULT_PROMPT_CYCLE

DEFAULT_ENSEMBLE_MODELS = (
    "openai/gpt-4o",
    "deepseek/deepseek-r1",
    "google/gemini-3.1-flash-lite",
)


@dataclass(frozen=True)
class EnsembleMember:
    model_id: str
    prompt_id: str


def ensemble_models() -> tuple[str, ...]:
    raw = os.getenv("ENSEMBLE_MODELS", "").strip()
    if not raw:
        return DEFAULT_ENSEMBLE_MODELS
    parts = [m.strip() for m in raw.split(",") if m.strip()]
    return tuple(parts) if parts else DEFAULT_ENSEMBLE_MODELS


def ensemble_members() -> tuple[EnsembleMember, ...]:
    """Map each model to a distinct forecaster prompt (cycle if fewer prompts than models)."""
    models = ensemble_models()
    return tuple(
        EnsembleMember(model_id=m, prompt_id=DEFAULT_PROMPT_CYCLE[i % len(DEFAULT_PROMPT_CYCLE)])
        for i, m in enumerate(models)
    )


def analyst_model() -> str:
    return os.getenv("ANALYST_MODEL", "google/gemini-3.1-flash-lite")


def aggregation() -> str:
    """``median`` (default, robust) or ``mean``."""
    return os.getenv("ENSEMBLE_AGGREGATION", "median").strip().lower()


def max_concurrency() -> int:
    return max(1, int(os.getenv("ENSEMBLE_MAX_CONCURRENCY", "10")))


def disagreement_threshold() -> float:
    return float(os.getenv("DISAGREEMENT_THRESHOLD", "0.25"))


def disagreement_shrink_strength() -> float:
    return float(os.getenv("DISAGREEMENT_SHRINK_STRENGTH", "0.5"))


def disagreement_market_weight() -> float:
    return float(os.getenv("DISAGREEMENT_MARKET_WEIGHT", "0.35"))
