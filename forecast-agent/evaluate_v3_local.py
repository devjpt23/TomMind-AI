"""Brier eval for v3 LangGraph ensemble (same events file as evaluate_local.py).

    uv run python evaluate_v3_local.py --events ../../eval_resolved.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from v3.runner import predict

from evaluate_local import resolved_to_yes, _close_time_str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="../../smallTest/events.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    events = json.loads(Path(args.events).read_text())
    if args.limit > 0:
        events = events[: args.limit]
    if not events:
        raise SystemExit("No events.")

    rows: list[tuple[str, float, float]] = []
    for i, event in enumerate(events, 1):
        ticker = event["market_ticker"]
        actual = resolved_to_yes(event)
        close_time = _close_time_str(event)
        print(f"[{i}/{len(events)}] {ticker} (actual={actual:.0f}) ...", flush=True)
        try:
            out = predict(
                market_ticker=ticker,
                title=event["title"],
                close_time=close_time,
                subtitle=event.get("subtitle"),
                description=event.get("description"),
                category=event.get("category"),
                rules=event.get("rules"),
            )
            p_yes = float(out["p_yes"])
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            p_yes = 0.5
        rows.append((ticker, p_yes, actual))
        print(f"  p_yes={p_yes:.3f}  brier_contrib={(p_yes - actual) ** 2:.4f}")

    brier = sum((p - a) ** 2 for _, p, a in rows) / len(rows)
    print(f"\nBrier (v3): {brier:.4f}")


if __name__ == "__main__":
    main()
