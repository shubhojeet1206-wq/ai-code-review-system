"""Tests for Pydantic review models."""

from pydantic import ValidationError
import pytest

from models.review_models import AgentReview, FinalReview, Finding, ReviewRequest, Severity


def test_finding_requires_title_and_description() -> None:
    finding = Finding(
        category="bug",
        severity=Severity.HIGH,
        title="Off-by-one",
        description="Loop bound is exclusive.",
        line_start=4,
        line_end=6,
        confidence=0.8,
    )
    assert finding.severity is Severity.HIGH
    assert finding.line_end == 6


def test_finding_swaps_inverted_line_range() -> None:
    finding = Finding(
        category="bug",
        severity="MEDIUM",
        title="Range",
        description="Inverted lines should be corrected.",
        line_start=10,
        line_end=2,
    )
    assert finding.line_end == 10


def test_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        Finding(
            category="bug",
            severity=Severity.LOW,
            title="x",
            description="y",
            confidence=1.5,
        )


def test_agent_review_defaults() -> None:
    review = AgentReview(agent_name="Bug Detection Agent")
    assert review.findings == []
    assert review.succeeded is True


def test_final_review_score_clamped_by_field() -> None:
    review = FinalReview(overall_score=100, summary="ok")
    assert review.overall_score == 100
    with pytest.raises(ValidationError):
        FinalReview(overall_score=140, summary="too high")


def test_review_request_rejects_blank_code() -> None:
    with pytest.raises(ValidationError):
        ReviewRequest(source_code="   \n", language="Python")
