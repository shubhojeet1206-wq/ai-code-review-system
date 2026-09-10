"""LangGraph orchestrator: parallel specialists, then a final aggregator."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.bug_agent import run_bug_agent
from agents.final_review_agent import run_final_review_agent
from agents.performance_agent import run_performance_agent
from agents.quality_agent import run_quality_agent
from agents.security_agent import run_security_agent
from agents.testing_agent import run_testing_agent
from orchestration.state import ReviewState, create_initial_state
from utils.logging_config import get_logger

logger = get_logger(__name__)

SPECIALIST_NODES: tuple[str, ...] = (
    "bug_agent",
    "security_agent",
    "performance_agent",
    "quality_agent",
    "testing_agent",
)

NODE_LABELS: dict[str, str] = {
    "bug_agent": "Bug Detection",
    "security_agent": "Security Analysis",
    "performance_agent": "Performance Analysis",
    "quality_agent": "Code Quality",
    "testing_agent": "Testing",
    "final_review": "Final Review",
}


def build_review_graph() -> Any:
    """Compile the fan-out / fan-in review graph."""
    graph = StateGraph(ReviewState)

    graph.add_node("bug_agent", run_bug_agent)
    graph.add_node("security_agent", run_security_agent)
    graph.add_node("performance_agent", run_performance_agent)
    graph.add_node("quality_agent", run_quality_agent)
    graph.add_node("testing_agent", run_testing_agent)
    graph.add_node("final_review", run_final_review_agent)

    # Run specialists one after another. The Gemini free tier allows 20
    # generate_content requests per day per model; parallel fan-out plus short
    # retries burns that quota and leaves later nodes stuck "running".
    previous = START
    for node in SPECIALIST_NODES:
        graph.add_edge(previous, node)
        previous = node
    graph.add_edge(previous, "final_review")
    graph.add_edge("final_review", END)
    return graph.compile()


@lru_cache(maxsize=1)
def get_review_graph() -> Any:
    """Return a process-wide compiled graph."""
    logger.info("Compiling LangGraph code-review workflow")
    return build_review_graph()


def run_code_review(source_code: str, language: str) -> ReviewState:
    """Synchronous convenience wrapper used by tests and scripts."""
    graph = get_review_graph()
    initial = create_initial_state(source_code, language)
    logger.info("Starting code-review workflow for language=%s", language)
    result = graph.invoke(initial)
    logger.info("Code-review workflow finished")
    return result
