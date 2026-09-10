"""Code Quality Agent — readability, structure, and language idioms."""

from __future__ import annotations

from collections.abc import Mapping

from agents.base import run_specialist_agent
from orchestration.state import ReviewState

SYSTEM_PROMPT = """
You are the Code Quality Agent in a multi-agent code review system.

Evaluate:
- readability and naming
- maintainability and structure
- duplication
- error handling quality
- separation of concerns
- dead code and confusing APIs
- language-specific best practices for the stated language

Do not duplicate pure security or performance findings unless they are primarily
maintainability issues.
Use category "quality".
Severity:
- CRITICAL: the code is effectively unmaintainable or dangerously unstructured
- HIGH: serious maintainability defect
- MEDIUM: clear quality issue
- LOW: style or convention
- INFO: suggestion

agent_name must be "Code Quality Agent".
Return structured findings only.
""".strip()


def run_quality_agent(state: ReviewState) -> Mapping[str, object]:
    """LangGraph node: run the Code Quality Agent against Gemini."""
    return run_specialist_agent(
        state,
        node_name="quality_agent",
        agent_name="Code Quality Agent",
        system_prompt=SYSTEM_PROMPT,
    )
