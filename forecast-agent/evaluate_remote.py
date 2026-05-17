"""Score a live /predict endpoint on resolved markets (Brier).

    uv run python evaluate_remote.py --agent-url https://tommind-ai-fjr3r.ondigitalocean.app/predict
    uv run python evaluate_remote.py --agent-url ... --events ../../eval_resolved.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from evaluate_local import resolved_to_yes


def _event_payload(event: dict) -> dict:
    ct = event["close_time"]
    close_time = ct if isinstance(ct, str) else ct.isoformat().replace("+00:00", "Z")
    return {
        "event_ticker": event["event_ticker"],
        "market_ticker": event["market_ticker"],
        "title": event["title"],
        "subtitle": event.get("subtitle"),
        "description": event.get("description"),
        "category": event["category"],
        "rules": event.get("rules"),
        "close_time": close_time,
    }


def predict_remote(
    agent_url: str,
    event: dict,
    timeout: float,
    *,
    resolve_ip: str | None = None,
) -> tuple[float, str]:
    payload = _event_payload(event)
    if resolve_ip:
        host = urlparse(agent_url).hostname
        port = urlparse(agent_url).port or 443
        result = subprocess.run(
            [
                "curl",
                "-s",
                "-f",
                "--resolve",
                f"{host}:{port}:{resolve_ip}",
                "-X",
                "POST",
                agent_url,
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(payload),
                "--max-time",
                str(int(timeout)),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "curl failed")
        data = json.loads(result.stdout)
    else:
        resp = requests.post(agent_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    p_yes = float(data["p_yes"])
    rationale = data.get("rationale") or ""
    return p_yes, rationale


def run_eval(
    agent_url: str,
    events: list[dict],
    timeout: float,
    *,
    resolve_ip: str | None = None,
) -> None:
    health_url = agent_url.rstrip("/").rsplit("/", 1)[0] + "/health"
    try:
        if resolve_ip:
            host = urlparse(health_url).hostname
            port = urlparse(health_url).port or 443
            h = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-w",
                    "%{http_code}",
                    "--resolve",
                    f"{host}:{port}:{resolve_ip}",
                    health_url,
                    "--max-time",
                    "30",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            body, code = h.stdout[:-3], h.stdout[-3:]
            print(f"Health: {health_url} -> HTTP {code} {body[:80]}")
        else:
            h = requests.get(health_url, timeout=30)
            print(f"Health: {health_url} -> HTTP {h.status_code} {h.text[:80]}")
    except (requests.RequestException, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"Health check failed: {exc}", file=sys.stderr)

    rows: list[tuple[str, float, float]] = []
    for i, event in enumerate(events, 1):
        ticker = event["market_ticker"]
        actual = resolved_to_yes(event)
        print(f"[{i}/{len(events)}] {ticker} (actual={actual:.0f}) ...", flush=True)
        t0 = time.perf_counter()
        try:
            p_yes, rationale = predict_remote(
                agent_url, event, timeout, resolve_ip=resolve_ip
            )
            note = ""
            if "Fallback" in rationale or "not set" in rationale:
                note = " [WARN: fallback]"
            print(
                f"  p_yes={p_yes:.3f}  brier_contrib={(p_yes - actual) ** 2:.4f}"
                f"  ({time.perf_counter() - t0:.1f}s){note}"
            )
        except Exception as exc:
            print(f"  FAILED ({time.perf_counter() - t0:.1f}s): {exc}", file=sys.stderr)
            p_yes = 0.5
            print(f"  p_yes=0.500  brier_contrib={(p_yes - actual) ** 2:.4f} (failed)")
        rows.append((ticker, p_yes, actual))

    brier = sum((p - a) ** 2 for _, p, a in rows) / len(rows)
    print()
    print(f"Endpoint:        {agent_url}")
    print(f"Markets scored:  {len(rows)}")
    print(f"Brier score:     {brier:.4f}  (lower is better)")
    print(f"Naive baseline:  0.2500")
    if brier < 0.25:
        print("Beat the 0.5 baseline.")
    else:
        print("Did not beat the 0.5 baseline on this set.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-url", required=True, help="Live POST /predict URL")
    parser.add_argument(
        "--events",
        default="../../eval_resolved.json",
        help="Resolved events JSON",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--resolve-ip",
        default="",
        help="Pin host to IP via curl --resolve (if local DNS fails)",
    )
    args = parser.parse_args()

    events = json.loads(Path(args.events).read_text())
    if args.limit > 0:
        events = events[: args.limit]

    resolve_ip = args.resolve_ip or None
    run_eval(args.agent_url, events, args.timeout, resolve_ip=resolve_ip)


if __name__ == "__main__":
    main()
