"""Final Review Agent — merge specialist findings into one structured review."""

from __future__ import annotations

import json
from collections.abc import Mapping

from llm.client import (
    ConfigurationError,
    GeminiAPIError,
    StructuredOutputError,
    invoke_structured,
    sanitize_error_message,
)
from models.review_models import AgentReview, FinalReview, Finding, RecommendedTest, Severity, SuggestedFix
from orchestration.state import ReviewState
from utils.logging_config import get_logger, redact_secrets

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are the Final Review Agent. You receive structured findings from:
Bug Detection, Security, Performance, Code Quality, and Testing agents.

You must:
1. Remove duplicate findings (same issue described twice).
2. Merge related findings into a stronger single finding when they describe one defect.
3. Resolve conflicts when possible (prefer higher-confidence, more specific, or more severe
   evidence; if unresolved, keep the more conservative/higher-severity view and note it).
4. Prioritize by severity: CRITICAL, HIGH, MEDIUM, LOW, INFO.
5. Preserve useful line numbers and code snippets.
6. Produce concise suggested fixes.
7. Produce recommended test cases (from the Testing agent plus any gaps you see).
8. Calculate overall_score from 0-100:
   start at 100, then deduct approximately:
   CRITICAL -20, HIGH -10, MEDIUM -5, LOW -2, INFO -0
   after de-duplication. Clamp to 0-100.
   If several specialist agents failed, lower the score modestly and say so in the summary.

Bucket findings:
- critical_issues: severity CRITICAL (all categories)
- bugs: bug-category findings that are not already listed as critical_issues
- security_vulnerabilities: security-category findings that are not already listed as critical_issues
- performance_problems: performance-category findings that are not already listed as critical_issues
- code_quality_issues: quality-category findings that are not already listed as critical_issues
Testing findings should inform recommended_test_cases; include the most serious testing
gaps in code_quality_issues only if they are truly quality/process issues.

Write a concise executive summary.
agent_summaries should be short per-agent notes including any agent failures.
""".strip()


def _as_agent_review(item: AgentReview | dict) -> AgentReview:
    if isinstance(item, AgentReview):
        return item
    return AgentReview.model_validate(item)


def _reviews_as_json(reviews: list[AgentReview]) -> str:
    payload = [review.model_dump(mode="json") for review in reviews]
    return json.dumps(payload, indent=2)


def _fallback_final_review(reviews: list[AgentReview], errors: list[str]) -> FinalReview:
    """Deterministic aggregation if the final Gemini call fails.

    Still used only after a real specialist run; this path exists so the UI can
    show partial results instead of crashing.
    """
    all_findings: list[Finding] = []
    seen: set[tuple[str, str, str | None]] = set()
    for review in reviews:
        for finding in review.findings:
            key = (finding.title.strip().lower(), finding.category.lower(), finding.code_snippet)
            if key in seen:
                continue
            seen.add(key)
            all_findings.append(finding)

    severity_rank = {
        Severity.CRITICAL: 0,
        Severity.HIGH: 1,
        Severity.MEDIUM: 2,
        Severity.LOW: 3,
        Severity.INFO: 4,
    }
    all_findings.sort(key=lambda item: (severity_rank[item.severity], -item.confidence))

    deductions = {
        Severity.CRITICAL: 20,
        Severity.HIGH: 10,
        Severity.MEDIUM: 5,
        Severity.LOW: 2,
        Severity.INFO: 0,
    }
    score = 100
    for finding in all_findings:
        score -= deductions[finding.severity]
    if errors:
        score -= min(15, 5 * len(errors))
    score = max(0, min(100, score))

    critical = [f for f in all_findings if f.severity == Severity.CRITICAL]
    critical_titles = {f.title for f in critical}

    def remaining(category: str) -> list[Finding]:
        return [
            f
            for f in all_findings
            if f.category.lower() == category and f.title not in critical_titles
        ]

    fixes = [
        SuggestedFix(
            title=f"Fix: {finding.title}",
            description=finding.suggested_fix or finding.description,
            severity=finding.severity,
            related_finding_title=finding.title,
        )
        for finding in all_findings[:12]
        if finding.severity in {Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM}
    ]
    tests = [
        RecommendedTest(
            title=finding.title,
            description=finding.description,
            test_type="unit",
        )
        for finding in all_findings
        if finding.category.lower() == "testing"
    ][:10]

    summaries = [
        f"{review.agent_name}: {'ok' if review.succeeded else 'failed'} — {review.summary}"
        for review in reviews
    ]
    summary = (
        "Unified review assembled locally because the Final Review Agent call failed. "
        "Specialist findings were de-duplicated by title/category."
    )
    if errors:
        summary += " Some specialist agents reported errors."

    return FinalReview(
        overall_score=score,
        summary=summary,
        critical_issues=critical,
        bugs=remaining("bug"),
        security_vulnerabilities=remaining("security"),
        performance_problems=remaining("performance"),
        code_quality_issues=remaining("quality"),
        suggested_fixes=fixes,
        recommended_test_cases=tests,
        agent_summaries=summaries,
    )


def run_final_review_agent(state: ReviewState) -> Mapping[str, object]:
    """LangGraph node: merge specialist AgentReview objects via Gemini."""
    logger.info("Agent start: Final Review Agent")
    reviews = [_as_agent_review(item) for item in (state.get("agent_reviews") or [])]
    errors = list(state.get("errors") or [])

    user_prompt = (
        f"Programming language: {state['language']}\n\n"
        "Structured specialist reviews (JSON):\n"
        f"{_reviews_as_json(reviews)}\n\n"
        f"Workflow errors: {json.dumps(errors)}\n"
        "Produce the unified FinalReview."
    )

    try:
        final = invoke_structured(
            schema=FinalReview,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            agent_name="Final Review Agent",
        )
        logger.info("Agent end: Final Review Agent (score=%s)", final.overall_score)
        return {
            "final_review": final,
            "agent_status": {"final_review": "completed"},
            "errors": [],
        }
    except (ConfigurationError, GeminiAPIError, StructuredOutputError) as exc:
        message = sanitize_error_message(exc)
        logger.error(
            "Agent failure: Final Review Agent (%s): %s",
            type(exc).__name__,
            redact_secrets(str(exc))[:500],
        )
        fallback = _fallback_final_review(reviews, errors)
        fallback.summary = f"{fallback.summary} Error: {message}"
        return {
            "final_review": fallback,
            "agent_status": {"final_review": "failed"},
            "errors": [f"Final Review Agent: {message}"],
        }
    except Exception as exc:  # noqa: BLE001
        message = sanitize_error_message(exc)
        logger.exception("Unexpected Final Review Agent failure")
        fallback = _fallback_final_review(reviews, errors)
        fallback.summary = f"{fallback.summary} Error: {message}"
        return {
            "final_review": fallback,
            "agent_status": {"final_review": "failed"},
            "errors": [f"Final Review Agent: {message}"],
        }
