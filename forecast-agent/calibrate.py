"""Extremity shrinkage calibrator (code-only, no LLM)."""

from __future__ import annotations

import os

DEFAULT_LAMBDA = 0.85
MIN_P = 0.01
MAX_P = 0.99


def calibrate(p_raw: float, lambda_: float | None = None) -> float:
    """Shrink ``p_raw`` toward 0.5, then clip to [0.01, 0.99].

    Formula: p_final = 0.5 + λ * (p_raw - 0.5)

    Set λ via argument or env ``CALIBRATE_LAMBDA`` (default 0.85).
    """
    if lambda_ is not None:
        lam = lambda_
    elif os.getenv("CALIBRATE_LAMBDA"):
        lam = float(os.environ["CALIBRATE_LAMBDA"])
    else:
        lam = DEFAULT_LAMBDA
    p = 0.5 + lam * (p_raw - 0.5)
    return max(MIN_P, min(MAX_P, p))
