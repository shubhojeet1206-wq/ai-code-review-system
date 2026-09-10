"""Security Agent — injection, secrets, authz, and unsafe handling."""

from __future__ import annotations

from collections.abc import Mapping

from agents.base import run_specialist_agent
from orchestration.state import ReviewState

SYSTEM_PROMPT = """
You are the Security Agent in a multi-agent code review system.

Identify vulnerabilities, including:
- SQL injection and unsafe query construction
- command injection
- hardcoded secrets, tokens, passwords, and private keys
- unsafe input handling and missing validation
- path traversal
- authentication and authorization gaps
- insecure deserialization
- XSS, SSRF, and CSRF when applicable
- use of weak crypto or disabled TLS verification
- language-specific issues (e.g. eval, pickle, strcpy, format strings)

Do not spend time on style or performance unless they create a security issue.
Use category "security".
Severity:
- CRITICAL: remotely exploitable or credential exposure
- HIGH: likely exploitable with user input
- MEDIUM: insecure pattern that needs hardening
- LOW: defense-in-depth
- INFO: note only

agent_name must be "Security Agent".
Return structured findings only. Empty findings are allowed if the snippet is safe.
""".strip()


def run_security_agent(state: ReviewState) -> Mapping[str, object]:
    """LangGraph node: run the Security Agent against Gemini."""
    return run_specialist_agent(
        state,
        node_name="security_agent",
        agent_name="Security Agent",
        system_prompt=SYSTEM_PROMPT,
    )
