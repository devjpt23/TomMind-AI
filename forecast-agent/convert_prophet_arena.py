"""Convert Prophet-Arena-Subset-100 CSV to Event JSON for evaluate_remote."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path


def _parse(cell: str):
    return ast.literal_eval(cell.strip())


def convert_row(row: dict, *, one_per_event: bool) -> list[dict]:
    outcomes = _parse(row["market_outcome"])
    market_info = _parse(row["market_info"])
    close_time = row["close_time"].replace("+00:00", "Z")
    if not close_time.endswith("Z"):
        close_time = close_time + "Z" if "T" in close_time else close_time

    events: list[dict] = []
    items = list(outcomes.items())
    if one_per_event and items:
        # Prefer the market that resolved Yes (actual=1), else first.
        items = [max(items, key=lambda kv: kv[1])]

    for name, actual in items:
        info = market_info.get(name) or {}
        ticker = info.get("ticker") or f"{row['event_ticker']}-{name[:12]}"
        title = info.get("title") or row["title"]
        rules = info.get("rules_primary") or info.get("rules_secondary") or title
        events.append(
            {
                "event_ticker": row["event_ticker"],
                "market_ticker": ticker,
                "title": title,
                "subtitle": info.get("subtitle"),
                "description": rules,
                "category": row["category"],
                "rules": rules,
                "close_time": close_time,
                "outcomes": ["Yes", "No"],
                "resolved_outcome": {
                    "value": ["Yes"] if int(actual) == 1 else ["No"],
                    "resolved_at": row.get("submission_created_at") or close_time,
                    "source": ticker,
                },
            }
        )
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="../../data/prophet_arena_100.csv",
        help="Prophet Arena Subset-100 CSV",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="../../data/prophet_arena_100_events.json",
    )
    parser.add_argument(
        "--one-per-event",
        action="store_true",
        help="Emit one binary market per event (~100 rows). Default: all (~1061).",
    )
    args = parser.parse_args()

    all_events: list[dict] = []
    with Path(args.input).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_events.extend(convert_row(row, one_per_event=args.one_per_event))

    out = Path(args.output)
    out.write_text(json.dumps(all_events, indent=2))
    print(f"Wrote {len(all_events)} markets -> {out}")


if __name__ == "__main__":
    main()
