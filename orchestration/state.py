"""LangGraph state schema.

Specialized agents run in parallel and write into reducer-backed fields so
concurrent updates merge instead of colliding.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from models.review_models import AgentReview, FinalReview


def _extend_reviews(existing: list[AgentReview], incoming: list[AgentReview]) -> list[AgentReview]:
    return (existing or []) + (incoming or [])


def _extend_errors(existing: list[str], incoming: list[str]) -> list[str]:
    return (existing or []) + (incoming or [])


def _merge_status(existing: dict[str, str], incoming: dict[str, str]) -> dict[str, str]:
    merged = dict(existing or {})
    merged.update(incoming or {})
    return merged


SpecialistName = Literal[
    "bug_agent",
    "security_agent",
    "performance_agent",
    "quality_agent",
    "testing_agent",
]


class ReviewState(TypedDict):
    source_code: str
    language: str
    agent_reviews: Annotated[list[AgentReview], _extend_reviews]
    errors: Annotated[list[str], _extend_errors]
    agent_status: Annotated[dict[str, str], _merge_status]
    final_review: Optional[FinalReview]


def create_initial_state(source_code: str, language: str) -> ReviewState:
    """Build the graph input with empty reducer collections."""
    return ReviewState(
        source_code=source_code,
        language=language,
        agent_reviews=[],
        errors=[],
        agent_status={
            "bug_agent": "pending",
            "security_agent": "pending",
            "performance_agent": "pending",
            "quality_agent": "pending",
            "testing_agent": "pending",
            "final_review": "pending",
        },
        final_review=None,
    )
