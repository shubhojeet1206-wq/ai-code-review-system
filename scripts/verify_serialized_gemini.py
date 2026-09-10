"""Confirm Gemini calls are serialized and succeed."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from llm.client import invoke_structured
from models.review_models import AgentReview

PROMPT = "Programming language: Python\n\nSource:\n1 | x = 1\n"


def _one(name: str) -> str:
    review = invoke_structured(
        schema=AgentReview,
        system_prompt=f"You are the {name}. Return AgentReview. category bug if needed.",
        user_prompt=PROMPT,
        agent_name=name,
    )
    return f"{name}: ok findings={len(review.findings)}"


def main() -> None:
    names = ["Bug Detection Agent", "Security Agent"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_one, names))
    for line in results:
        print(line)


if __name__ == "__main__":
    main()
