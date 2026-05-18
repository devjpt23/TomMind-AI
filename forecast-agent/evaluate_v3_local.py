"""Brier eval for v3 LangGraph ensemble (same events file as evaluate_local.py).

    uv run python evaluate_v3_local.py --events ../../eval_resolved.json
    uv run python evaluate_v3_local.py --events ../../eval_resolved.json --concurrency 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from v3.runner import predict

from evaluate_local import _close_time_str, resolved_to_yes


async def _predict_one(event: dict) -> tuple[str, float, float, float | None]:
    ticker = event["market_ticker"]
    actual = resolved_to_yes(event)
    close_time = _close_time_str(event)
    t0 = time.perf_counter()
    try:
        out = await asyncio.to_thread(
            predict,
            market_ticker=ticker,
            title=event["title"],
            close_time=close_time,
            subtitle=event.get("subtitle"),
            description=event.get("description"),
            category=event.get("category"),
            rules=event.get("rules"),
        )
        p_yes = float(out["p_yes"])
        elapsed = time.perf_counter() - t0
        return ticker, p_yes, actual, elapsed
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"  FAILED {ticker} ({elapsed:.1f}s): {exc}", file=sys.stderr)
        return ticker, 0.5, actual, elapsed


async def run_eval(events: list[dict], concurrency: int) -> None:
    sem = asyncio.Semaphore(max(1, concurrency))
    total = len(events)

    async def _run(i: int, event: dict) -> tuple[int, str, float, float, float | None]:
        async with sem:
            ticker = event["market_ticker"]
            actual = resolved_to_yes(event)
            print(f"[{i}/{total}] {ticker} (actual={actual:.0f}) ...", flush=True)
            ticker, p_yes, actual, elapsed = await _predict_one(event)
            err = (p_yes - actual) ** 2
            print(
                f"  p_yes={p_yes:.3f}  brier_contrib={err:.4f}  ({elapsed:.1f}s)",
                flush=True,
            )
            return i, ticker, p_yes, actual, elapsed

    results = await asyncio.gather(*[_run(i, e) for i, e in enumerate(events, 1)])
    results.sort(key=lambda r: r[0])

    rows = [(t, p, a) for _, t, p, a, _ in results]
    total_s = sum(r[4] or 0.0 for r in results)
    brier = sum((p - a) ** 2 for _, p, a in rows) / len(rows)
    print()
    print(f"Markets scored:  {len(rows)}")
    print(f"Concurrency:     {concurrency}")
    print(f"Wall-ish time:   {total_s:.1f}s sum of per-market times")
    print(f"Brier (v3):      {brier:.4f}  (lower is better)")
    print(f"Naive baseline:  0.2500")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="../../smallTest/events.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("EVAL_CONCURRENCY", "1")),
        help="Markets in flight at once (default 1; try 2-3 for eval)",
    )
    args = parser.parse_args()

    events = json.loads(Path(args.events).read_text())
    if args.limit > 0:
        events = events[: args.limit]
    if not events:
        raise SystemExit("No events.")

    asyncio.run(run_eval(events, args.concurrency))


if __name__ == "__main__":
    main()
