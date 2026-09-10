"""Shared specialist-agent execution: Gemini call, validation, graceful failure."""

from __future__ import annotations

from collections.abc import Mapping

from llm.client import (
    ConfigurationError,
    GeminiAPIError,
    StructuredOutputError,
    invoke_structured,
    sanitize_error_message,
)
from models.review_models import AgentReview
from orchestration.state import ReviewState
from utils.file_utils import with_line_numbers
from utils.logging_config import get_logger, redact_secrets

logger = get_logger(__name__)

MAX_PROMPT_CHARS = 60_000


def _truncated_numbered_code(source_code: str) -> str:
    numbered = with_line_numbers(source_code)
    if len(numbered) <= MAX_PROMPT_CHARS:
        return numbered
    return numbered[:MAX_PROMPT_CHARS] + "\n... [truncated for prompt size]"


def build_user_prompt(*, language: str, source_code: str, extra: str = "") -> str:
    numbered = _truncated_numbered_code(source_code)
    parts = [
        f"Programming language: {language}",
        "",
        "Review the following source code. Line numbers are shown on the left.",
        "Cite line_start/line_end using those numbers when a finding maps to specific lines.",
        "If there are no issues in your specialty, return an empty findings list and say so in summary.",
        "",
        "Source code:",
        numbered,
    ]
    if extra:
        parts.extend(["", extra])
    return "\n".join(parts)


def run_specialist_agent(
    state: ReviewState,
    *,
    node_name: str,
    agent_name: str,
    system_prompt: str,
) -> Mapping[str, object]:
    """Invoke Gemini for one specialist and return a LangGraph state update."""
    logger.info("Agent start: %s", agent_name)
    try:
        review = invoke_structured(
            schema=AgentReview,
            system_prompt=system_prompt,
            user_prompt=build_user_prompt(
                language=state["language"],
                source_code=state["source_code"],
            ),
            agent_name=agent_name,
        )
        if not review.agent_name:
            review.agent_name = agent_name
        review.succeeded = True
        review.error_message = None
        logger.info(
            "Agent end: %s (findings=%s)",
            agent_name,
            len(review.findings),
        )
        return {
            "agent_reviews": [review],
            "agent_status": {node_name: "completed"},
            "errors": [],
        }
    except (ConfigurationError, GeminiAPIError, StructuredOutputError) as exc:
        message = sanitize_error_message(exc)
        logger.error(
            "Agent failure: %s (%s): %s",
            agent_name,
            type(exc).__name__,
            redact_secrets(str(exc))[:500],
        )
        failed = AgentReview(
            agent_name=agent_name,
            findings=[],
            summary=f"{agent_name} did not complete: {message}",
            succeeded=False,
            error_message=message,
        )
        return {
            "agent_reviews": [failed],
            "agent_status": {node_name: "failed"},
            "errors": [f"{agent_name}: {message}"],
        }
    except Exception as exc:  # noqa: BLE001 - keep the graph alive
        message = sanitize_error_message(exc)
        logger.exception("Unexpected agent failure: %s", agent_name)
        failed = AgentReview(
            agent_name=agent_name,
            findings=[],
            summary=f"{agent_name} failed unexpectedly: {message}",
            succeeded=False,
            error_message=message,
        )
        return {
            "agent_reviews": [failed],
            "agent_status": {node_name: "failed"},
            "errors": [f"{agent_name}: {message}"],
        }
