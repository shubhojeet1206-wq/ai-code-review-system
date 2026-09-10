"""Structured Pydantic models used by every review agent and the UI."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Severity(str, Enum):
    """Finding severity ordered from most to least urgent."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class Finding(BaseModel):
    """A single issue identified by a specialized review agent."""

    category: str = Field(
        description="Issue category, e.g. bug, security, performance, quality, testing."
    )
    severity: Severity
    title: str = Field(min_length=1, description="Short finding title.")
    description: str = Field(min_length=1, description="Clear explanation of the issue.")
    line_start: Optional[int] = Field(
        default=None, ge=1, description="Starting line number in the submitted code."
    )
    line_end: Optional[int] = Field(
        default=None, ge=1, description="Ending line number in the submitted code."
    )
    code_snippet: Optional[str] = Field(
        default=None, description="Relevant excerpt from the submitted code."
    )
    suggested_fix: Optional[str] = Field(
        default=None, description="Concrete recommended fix."
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Model confidence between 0 and 1.",
    )

    @field_validator("line_end")
    @classmethod
    def line_end_not_before_start(cls, value: Optional[int], info) -> Optional[int]:
        start = info.data.get("line_start")
        if value is not None and start is not None and value < start:
            return start
        return value


class AgentReview(BaseModel):
    """Structured result produced by one specialized agent."""

    agent_name: str
    findings: list[Finding] = Field(default_factory=list)
    summary: str = Field(default="")
    succeeded: bool = Field(default=True)
    error_message: Optional[str] = Field(
        default=None, description="Safe, non-secret error text if the agent failed."
    )


class SuggestedFix(BaseModel):
    """A prioritized fix recommended in the unified review."""

    title: str
    description: str
    severity: Severity = Severity.MEDIUM
    related_finding_title: Optional[str] = None


class RecommendedTest(BaseModel):
    """A test case recommended by the final review agent."""

    title: str
    description: str
    test_type: str = Field(
        default="unit",
        description="unit, integration, regression, or edge-case.",
    )


class FinalReview(BaseModel):
    """Unified review produced after all specialized agents complete."""

    overall_score: int = Field(ge=0, le=100)
    summary: str
    critical_issues: list[Finding] = Field(default_factory=list)
    bugs: list[Finding] = Field(default_factory=list)
    security_vulnerabilities: list[Finding] = Field(default_factory=list)
    performance_problems: list[Finding] = Field(default_factory=list)
    code_quality_issues: list[Finding] = Field(default_factory=list)
    suggested_fixes: list[SuggestedFix] = Field(default_factory=list)
    recommended_test_cases: list[RecommendedTest] = Field(default_factory=list)
    agent_summaries: list[str] = Field(
        default_factory=list,
        description="Short per-agent status notes for the dashboard.",
    )

    @field_validator("overall_score")
    @classmethod
    def clamp_score(cls, value: int) -> int:
        return max(0, min(100, value))


class ReviewRequest(BaseModel):
    """Validated input for a review run."""

    source_code: str = Field(min_length=1)
    language: str

    @field_validator("source_code")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Source code is empty.")
        return value
