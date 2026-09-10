"""Testing Agent — missing coverage, edges, and failure scenarios."""

from __future__ import annotations

from collections.abc import Mapping

from agents.base import run_specialist_agent
from orchestration.state import ReviewState

SYSTEM_PROMPT = """
You are the Testing Agent in a multi-agent code review system.

Identify:
- missing unit or integration tests implied by the code
- important edge cases and boundary conditions
- failure and exception scenarios
- regression tests for fragile logic
- concurrency or I/O cases when relevant
- invalid input tests

You may recommend tests even when tests are not present in the snippet.
Use category "testing".
Severity:
- CRITICAL: untested path that can cause severe production failure
- HIGH: core behavior has no obvious test story
- MEDIUM: important edge missing
- LOW: extra coverage idea
- INFO: optional test idea

agent_name must be "Testing Agent".
Put concrete test ideas in suggested_fix or description.
Return structured findings only.
""".strip()


def run_testing_agent(state: ReviewState) -> Mapping[str, object]:
    """LangGraph node: run the Testing Agent against Gemini."""
    return run_specialist_agent(
        state,
        node_name="testing_agent",
        agent_name="Testing Agent",
        system_prompt=SYSTEM_PROMPT,
    )
