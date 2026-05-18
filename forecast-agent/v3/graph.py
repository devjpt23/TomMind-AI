"""LangGraph definition for v3.1 parallel ensemble forecasting."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from v3 import nodes
from v3.state import GraphState


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("prepare_context", nodes.prepare_context)
    builder.add_node("shared_research", nodes.shared_research)
    builder.add_node("forecast_one", nodes.forecast_one)
    builder.add_node("aggregate_forecasts", nodes.aggregate_forecasts)
    builder.add_node("blend_and_calibrate", nodes.blend_and_calibrate)

    builder.add_edge(START, "prepare_context")
    builder.add_edge("prepare_context", "shared_research")
    builder.add_conditional_edges(
        "shared_research",
        nodes.dispatch_forecasters,
        ["forecast_one"],
    )
    builder.add_edge("forecast_one", "aggregate_forecasts")
    builder.add_edge("aggregate_forecasts", "blend_and_calibrate")
    builder.add_edge("blend_and_calibrate", END)
    return builder.compile()


GRAPH = build_graph()
