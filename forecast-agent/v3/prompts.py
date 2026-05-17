"""Forecaster prompt variants for ensemble diversity."""

from __future__ import annotations

PROMPT_IDS = ("superforecaster", "base_rate", "phi4_scratchpad", "structured_7step")

DEFAULT_PROMPT_CYCLE = PROMPT_IDS

# Default v3 ensemble: model -> prompt (Turtel et al. 2502.05253 Figure 4).
DEFAULT_MODEL_PROMPTS: dict[str, str] = {
    "openai/gpt-4o": "superforecaster",
    "microsoft/phi-4": "phi4_scratchpad",
    "microsoft/phi-4-reasoning": "phi4_scratchpad",
    "google/gemini-3.1-flash-lite": "base_rate",
}
