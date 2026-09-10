"""One-off verification that Review Code hits live Gemini via LangGraph."""

from __future__ import annotations

from orchestration.graph import run_code_review

SAMPLE = """
import sqlite3

def login(user, password):
    conn = sqlite3.connect("app.db")
    q = "SELECT * FROM users WHERE name='" + user + "' AND pw='" + password + "'"
    return conn.execute(q).fetchall()

def run(cmd):
    return eval(cmd)
"""


def main() -> None:
    result = run_code_review(SAMPLE.strip(), "Python")
    statuses = result.get("agent_status") or {}
    errors = result.get("errors") or []
    review = result.get("final_review")
    print("statuses", statuses)
    print("errors", errors)
    print("final_review", type(review).__name__ if review is not None else None)
    if review is not None:
        print("score", review.overall_score)
        print("summary", review.summary[:240])
        print("critical", len(review.critical_issues))
        print("security", len(review.security_vulnerabilities))
    failed = [name for name, status in statuses.items() if status == "failed"]
    if failed:
        raise SystemExit(f"agents failed: {failed}")
    if review is None:
        raise SystemExit("no final review")
    print("workflow_ok")


if __name__ == "__main__":
    main()
