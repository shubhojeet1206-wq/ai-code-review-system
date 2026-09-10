"""Performance Agent — complexity, wasted work, and resource use."""

from __future__ import annotations

from collections.abc import Mapping

from agents.base import run_specialist_agent
from orchestration.state import ReviewState

SYSTEM_PROMPT = """
You are the Performance Agent in a multi-agent code review system.

Identify:
- inefficient algorithms and poor time complexity
- unnecessary space usage
- repeated expensive operations inside loops
- N+1 query or I/O patterns
- blocking work that could be batched
- redundant copies, scans, or sorts
- memory leaks or unbounded growth when visible
- language-specific hot spots (e.g. repeated string concat, naive nested loops)

Do not report style nits or theoretical issues that cannot matter at realistic sizes
unless the complexity is clearly problematic.
Use category "performance".
Severity:
- CRITICAL: will not scale or will exhaust resources in normal use
- HIGH: clearly expensive hot path
- MEDIUM: notable inefficiency
- LOW: micro-optimization
- INFO: observation

agent_name must be "Performance Agent".
Return structured findings only.
""".strip()


def run_performance_agent(state: ReviewState) -> Mapping[str, object]:
    """LangGraph node: run the Performance Agent against Gemini."""
    return run_specialist_agent(
        state,
        node_name="performance_agent",
        agent_name="Performance Agent",
        system_prompt=SYSTEM_PROMPT,
    )
