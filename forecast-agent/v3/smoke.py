"""Dry-run v3 graph structure without API keys.

    uv run python -m v3.smoke
"""

from __future__ import annotations

from v3.graph import GRAPH


def main() -> None:
    mermaid = GRAPH.get_graph().draw_mermaid()
    print("V3 LangGraph (mermaid):\n")
    print(mermaid)
    print("\nNodes:", list(GRAPH.get_graph().nodes.keys()))


if __name__ == "__main__":
    main()
