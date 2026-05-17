"""Score the agent on resolved markets (Brier). Usage:

    uv run python evaluate_local.py
    uv run python evaluate_local.py --limit 5
    uv run python evaluate_local.py --events ../../eval_resolved.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import calibrate
import market_prior
import openRouter
import research
from main import EventRequest, _build_event_text


def resolved_to_yes(event: dict) -> float:
    """Map dataset resolved_outcome to 1.0 (YES) or 0.0 (NO)."""
    val = (event.get("resolved_outcome") or {}).get("value") or []
    if not val:
        raise ValueError(f"no resolved_outcome for {event.get('market_ticker')}")
    winner = val[0]
    if isinstance(winner, str) and winner.lower() in ("yes", "true"):
        return 1.0
    if isinstance(winner, str) and winner.lower() in ("no", "false"):
        return 0.0
    outcomes = event.get("outcomes") or []
    # Rules: "If {outcomes[0]} wins ... resolves to Yes" (sample-resolved sports)
    if outcomes and winner == outcomes[0]:
        return 1.0
    return 0.0


def _close_time_str(event: dict) -> str:
    ct = event["close_time"]
    return ct if isinstance(ct, str) else ct.isoformat().replace("+00:00", "Z")


async def predict_one(event: dict) -> tuple[float, float, float, str | None]:
    """Returns (p_raw, p_blend, p_yes, market_source). Matches main.py v2 pipeline."""
    req = EventRequest(
        event_ticker=event["event_ticker"],
        market_ticker=event["market_ticker"],
        title=event["title"],
        subtitle=event.get("subtitle"),
        description=event.get("description"),
        category=event["category"],
        rules=event.get("rules"),
        close_time=_close_time_str(event),
    )
    ctx = await asyncio.to_thread(
        market_prior.fetch_market_context,
        market_ticker=req.market_ticker,
    )
    event_text = _build_event_text(req)
    news_block = await asyncio.to_thread(
        research.gather_news, event_text, req.close_time
    )
    prompt = f"{event_text}\n\n{ctx.prompt_block}\n\n{news_block}"
    raw = await asyncio.to_thread(openRouter.forecasterMain, prompt)
    p_raw = openRouter.parse_p_yes(raw)
    p_blend = market_prior.blend_with_market(p_raw, ctx.p_yes)
    p_yes = calibrate.calibrate(p_blend)
    return p_raw, p_blend, p_yes, ctx.source


async def run_eval(events: list[dict]) -> None:
    rows: list[tuple[str, float, float]] = []
    for i, event in enumerate(events, 1):
        ticker = event["market_ticker"]
        actual = resolved_to_yes(event)
        print(f"[{i}/{len(events)}] {ticker} (actual={actual:.0f}) ...", flush=True)
        try:
            p_raw, p_blend, p_yes, mkt_src = await predict_one(event)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            p_raw, p_blend, p_yes, mkt_src = 0.5, 0.5, 0.5, None
        rows.append((ticker, p_yes, actual))
        err = (p_yes - actual) ** 2
        print(
            f"  market={mkt_src} p_raw={p_raw:.3f} p_blend={p_blend:.3f} "
            f"p_yes={p_yes:.3f}  brier_contrib={err:.4f}"
        )

    brier = sum((p - a) ** 2 for _, p, a in rows) / len(rows)
    print()
    print(f"Markets scored: {len(rows)}")
    print(f"Brier score:     {brier:.4f}  (lower is better)")
    print(f"Naive baseline:  0.2500  (always predict 0.5)")
    if brier < 0.25:
        print("Beat the 0.5 baseline.")
    else:
        print("Did not beat the 0.5 baseline on this set.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events",
        default="../../smallTest/events.json",
        help="Events JSON (default: smallTest 5-market slice; use ../../eval_resolved.json for full 26)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max events (0 = all)")
    args = parser.parse_args()

    path = Path(args.events)
    events = json.loads(path.read_text())
    if args.limit > 0:
        events = events[: args.limit]

    if not events:
        raise SystemExit("No events to evaluate.")

    asyncio.run(run_eval(events))


if __name__ == "__main__":
    main()
