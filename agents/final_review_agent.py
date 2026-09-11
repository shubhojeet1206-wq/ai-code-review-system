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

You MUST populate the finding lists. Copy each specialist finding into the correct bucket.
Do not leave critical_issues, bugs, security_vulnerabilities, performance_problems,
code_quality_issues, suggested_fixes, or recommended_test_cases empty if specialists
reported matching items. Nested lists are required, not only the summary.
""".strip()


def _as_agent_review(item: AgentReview | dict) -> AgentReview:
    if isinstance(item, AgentReview):
        return item
    return AgentReview.model_validate(item)


def _reviews_as_json(reviews: list[AgentReview]) -> str:
    payload = [review.model_dump(mode="json") for review in reviews]
    return json.dumps(payload, indent=2)


def _normalize_category(category: str) -> str:
    text = (category or "").strip().lower()
    aliases = {
        "bug": "bug",
        "bugs": "bug",
        "correctness": "bug",
        "logic": "bug",
        "logical": "bug",
        "runtime": "bug",
        "security": "security",
        "vulnerability": "security",
        "vulnerabilities": "security",
        "performance": "performance",
        "perf": "performance",
        "quality": "quality",
        "code quality": "quality",
        "maintainability": "quality",
        "style": "quality",
        "testing": "testing",
        "test": "testing",
        "tests": "testing",
    }
    if text in aliases:
        return aliases[text]
    for key, mapped in aliases.items():
        if key in text:
            return mapped
    return text or "quality"


def _dedupe_findings(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, str, str | None]] = set()
    unique: list[Finding] = []
    for finding in findings:
        key = (finding.title.strip().lower(), _normalize_category(finding.category), finding.code_snippet)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding.model_copy(update={"category": _normalize_category(finding.category)}))
    return unique


def _score_from_findings(findings: list[Finding], errors: list[str]) -> int:
    deductions = {
        Severity.CRITICAL: 20,
        Severity.HIGH: 10,
        Severity.MEDIUM: 5,
        Severity.LOW: 2,
        Severity.INFO: 0,
    }
    score = 100
    for finding in findings:
        score -= deductions.get(finding.severity, 0)
    if errors:
        score -= min(15, 5 * len(errors))
    return max(0, min(100, score))


def _bucket_findings(findings: list[Finding]) -> dict[str, list[Finding]]:
    used: set[int] = set()
    critical = [item for item in findings if item.severity == Severity.CRITICAL]
    used.update(id(item) for item in critical)

    def remaining(category: str) -> list[Finding]:
        items = [
            item
            for item in findings
            if item.category == category and id(item) not in used
        ]
        used.update(id(item) for item in items)
        return items

    bugs = remaining("bug")
    security = remaining("security")
    performance = remaining("performance")
    quality = remaining("quality") + remaining("testing")
    leftovers = [item for item in findings if id(item) not in used]
    return {
        "critical_issues": critical,
        "bugs": bugs,
        "security_vulnerabilities": security,
        "performance_problems": performance,
        "code_quality_issues": quality + leftovers,
    }


def _findings_from_final(review: FinalReview) -> list[Finding]:
    grouped: list[Finding] = []
    for group in (
        review.critical_issues,
        review.bugs,
        review.security_vulnerabilities,
        review.performance_problems,
        review.code_quality_issues,
    ):
        grouped.extend(group)
    return _dedupe_findings(grouped)


def _findings_from_specialists(reviews: list[AgentReview]) -> list[Finding]:
    grouped: list[Finding] = []
    for review in reviews:
        grouped.extend(review.findings)
    return _dedupe_findings(grouped)


def _fallback_final_review(reviews: list[AgentReview], errors: list[str]) -> FinalReview:
    """Deterministic aggregation from specialist findings."""
    findings = _findings_from_specialists(reviews)
    severity_rank = {
        Severity.CRITICAL: 0,
        Severity.HIGH: 1,
        Severity.MEDIUM: 2,
        Severity.LOW: 3,
        Severity.INFO: 4,
    }
    findings.sort(key=lambda item: (severity_rank[item.severity], -item.confidence))
    buckets = _bucket_findings(findings)
    tests = [
        RecommendedTest(title=item.title, description=item.description, test_type="unit")
        for item in findings
        if item.category == "testing"
    ][:10]
    fixes = [
        SuggestedFix(
            title=f"Fix: {item.title}",
            description=item.suggested_fix or item.description,
            severity=item.severity,
            related_finding_title=item.title,
        )
        for item in findings[:12]
        if item.severity in {Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM}
    ]
    summaries = [
        f"{review.agent_name}: {'ok' if review.succeeded else 'failed'} — {review.summary}"
        for review in reviews
    ]
    summary = "Unified review from specialist findings."
    if errors:
        summary += " Some specialist agents reported errors."
    return FinalReview(
        overall_score=_score_from_findings(findings, errors),
        summary=summary,
        suggested_fixes=fixes,
        recommended_test_cases=tests,
        agent_summaries=summaries,
        **buckets,
    )


def coalesce_final_review(
    gemini: FinalReview,
    reviews: list[AgentReview],
    errors: list[str],
) -> FinalReview:
    """Keep Gemini's summary, but never drop specialist findings.

    Smaller Gemini models often fill summary/score and leave nested lists empty.
    """
    local = _fallback_final_review(reviews, errors)
    gemini_findings = _findings_from_final(gemini)
    specialist_findings = _findings_from_specialists(reviews)
    combined = _dedupe_findings(gemini_findings + specialist_findings)
    buckets = _bucket_findings(combined)
    score = gemini.overall_score
    if not gemini_findings and combined:
        score = _score_from_findings(combined, errors)
    fixes = gemini.suggested_fixes or local.suggested_fixes
    tests = gemini.recommended_test_cases or local.recommended_test_cases
    notes = gemini.agent_summaries or local.agent_summaries
    summary = gemini.summary.strip() or local.summary
    return FinalReview(
        overall_score=score,
        summary=summary,
        suggested_fixes=fixes,
        recommended_test_cases=tests,
        agent_summaries=notes,
        **buckets,
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
        final = coalesce_final_review(final, reviews, errors)
        logger.info(
            "Agent end: Final Review Agent (score=%s findings=%s)",
            final.overall_score,
            len(_findings_from_final(final)),
        )
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
