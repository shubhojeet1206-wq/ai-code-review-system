"""Google Gemini access through the official LangChain integration.

All agents call ``invoke_structured`` so model selection, API-key handling,
retries, and Pydantic validation stay in one place.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError

from utils.logging_config import get_logger, redact_secrets

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
MAX_STRUCTURED_ATTEMPTS = 3
STRUCTURED_OUTPUT_METHOD = "json_mode"
_GEMINI_GATE = threading.BoundedSemaphore(1)


def load_environment() -> None:
    """Load `.env` from the project root on every call.

    Streamlit keeps imported modules in memory across reruns, so a one-time
    load_dotenv() at import time will miss a key that is added later.
    """
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE, override=False)
    else:
        load_dotenv(override=False)


load_environment()


class ConfigurationError(RuntimeError):
    """Missing or invalid local configuration (not an API call failure)."""


class GeminiAPIError(RuntimeError):
    """Gemini API refused or failed a request."""


class StructuredOutputError(RuntimeError):
    """The model responded, but the payload could not be validated."""


def get_gemini_model_name() -> str:
    """Return the configured Gemini model id."""
    load_environment()
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def get_google_api_key() -> str:
    """Read GOOGLE_API_KEY from the environment."""
    load_environment()
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key or key == "your_gemini_api_key_here":
        raise ConfigurationError(
            "GOOGLE_API_KEY is missing. Copy .env.example to .env and set a valid Gemini API key."
        )
    return key


def sanitize_error_message(exc: BaseException) -> str:
    """Turn an exception into user-safe text with secrets removed."""
    raw = redact_secrets(str(exc) or exc.__class__.__name__)
    lowered = raw.lower()
    if "api key" in lowered and ("invalid" in lowered or "permission" in lowered or "unauth" in lowered):
        return "The Gemini API key is invalid or not authorized for this model."
    retry_after = _parse_retry_after(exc)
    if "generate_content_free_tier_requests" in lowered or "GenerateRequestsPerDay" in raw:
        wait = f" Google asked to retry in {retry_after:.0f}s." if retry_after else ""
        return (
            "Gemini free-tier request quota for this model is exhausted "
            f"(this key's limit is 20 requests/day for {get_gemini_model_name()})."
            f"{wait} Wait, or set GEMINI_MODEL to another model with remaining quota."
        )
    if "429" in raw or "resource exhausted" in lowered or "rate limit" in lowered:
        wait = f" Retry in {retry_after:.0f}s." if retry_after else ""
        return f"Gemini rate limit or quota was exceeded.{wait}"
    if "quota" in lowered:
        return "Gemini quota is exhausted for this API key or project."
    if "404" in raw or "not found" in lowered or "no longer available" in lowered:
        model = get_gemini_model_name()
        return (
            f"Gemini model '{model}' is not available for this account. "
            "Set GEMINI_MODEL in .env to a model your API key can access "
            "(for new Gemini API keys this is often gemini-3.5-flash-lite)."
        )
    if "timeout" in lowered or "timed out" in lowered:
        return "The Gemini request timed out. Try again with a smaller file."
    if "connect" in lowered or "network" in lowered or "dns" in lowered:
        return "Network error while calling the Gemini API. Check your internet connection."
    return raw[:500]


def _parse_retry_after(exc: BaseException | None) -> float | None:
    if exc is None:
        return None
    attr = getattr(exc, "retry_after", None)
    if isinstance(attr, (int, float)) and attr > 0:
        return float(attr)
    text = str(exc)
    match = re.search(r"Please retry in ([0-9]+(?:\.[0-9]+)?)s", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", text)
    if match:
        return float(match.group(1))
    return None


def _classify_api_error(exc: BaseException) -> GeminiAPIError | StructuredOutputError | ConfigurationError:
    message = sanitize_error_message(exc)
    lowered = (str(exc) or "").lower()
    if "api key" in lowered and ("missing" in lowered or "not found" in lowered):
        return ConfigurationError(message)
    return GeminiAPIError(message)


def _is_rate_limited(exc: BaseException | None) -> bool:
    if exc is None:
        return False
    text = str(exc).lower()
    return (
        "429" in text
        or "resource exhausted" in text
        or "rate limit" in text
        or "too many requests" in text
    )


def _wait_seconds_for_exception(exc: BaseException, attempt: int) -> float:
    retry_after = _parse_retry_after(exc)
    if retry_after is not None:
        return min(90.0, retry_after + 1.5)
    return min(30.0, 4.0 * (2 ** (attempt - 1)))


def get_chat_model(model_name: str | None = None) -> ChatGoogleGenerativeAI:
    """Create (and cache) the shared LangChain Gemini chat model.

    The API key is read at call time so importing this module does not require
    credentials. Streamlit can therefore start before a key is configured.
    """
    resolved = model_name or get_gemini_model_name()
    return _cached_chat_model(resolved)


@lru_cache(maxsize=4)
def _cached_chat_model(resolved: str) -> ChatGoogleGenerativeAI:
    api_key = get_google_api_key()
    logger.info("Initializing Gemini chat model '%s'", resolved)
    return ChatGoogleGenerativeAI(
        model=resolved,
        google_api_key=api_key,
        temperature=0.1,
        timeout=180,
        max_retries=0,
    )


def _extract_json_object(text: str) -> dict:
    """Parse a JSON object from model text, including fenced blocks."""
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    return json.loads(candidate)


def _validate_payload(payload: object, schema: type[T]) -> T:
    if isinstance(payload, schema):
        return payload
    if isinstance(payload, BaseModel):
        return schema.model_validate(payload.model_dump())
    if isinstance(payload, dict):
        return schema.model_validate(payload)
    raise StructuredOutputError(f"Unexpected structured payload type: {type(payload).__name__}")


def invoke_structured(
    *,
    schema: type[T],
    system_prompt: str,
    user_prompt: str,
    agent_name: str,
) -> T:
    """Call Gemini and validate the response as ``schema``.

    Only one Gemini request runs at a time so parallel LangGraph nodes do not
    trip provider rate limits. 429 responses are retried with backoff.
    """
    with _GEMINI_GATE:
        try:
            return _invoke_structured_locked(
                schema=schema,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                agent_name=agent_name,
            )
        finally:
            time.sleep(0.4)


def _invoke_structured_locked(
    *,
    schema: type[T],
    system_prompt: str,
    user_prompt: str,
    agent_name: str,
) -> T:
    llm = get_chat_model()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    last_error: BaseException | None = None

    for attempt in range(1, MAX_STRUCTURED_ATTEMPTS + 1):
        try:
            logger.info("%s: Gemini structured-output attempt %s", agent_name, attempt)
            structured = llm.with_structured_output(schema, method=STRUCTURED_OUTPUT_METHOD)
            result = structured.invoke(messages)
            validated = _validate_payload(result, schema)
            logger.info("%s: structured output validated on attempt %s", agent_name, attempt)
            return validated
        except ValidationError as exc:
            last_error = exc
            logger.warning(
                "%s: Pydantic validation failed on attempt %s (%s)",
                agent_name,
                attempt,
                exc.error_count(),
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "%s: Gemini exception %s on attempt %s: %s",
                agent_name,
                type(exc).__name__,
                attempt,
                redact_secrets(str(exc))[:500],
            )
            classified = _classify_api_error(exc)
            if isinstance(classified, ConfigurationError):
                raise classified from exc
            if _is_rate_limited(exc) and attempt < MAX_STRUCTURED_ATTEMPTS:
                wait = _wait_seconds_for_exception(exc, attempt)
                logger.info("%s: quota/rate limited; waiting %.1fs as requested by the API", agent_name, wait)
                time.sleep(wait)
                continue
            if attempt >= MAX_STRUCTURED_ATTEMPTS:
                raise GeminiAPIError(sanitize_error_message(exc)) from exc
        time.sleep(0.4 * attempt)

    logger.info("%s: falling back to JSON parse from a raw Gemini response", agent_name)
    fallback_messages = [
        SystemMessage(
            content=(
                system_prompt
                + "\n\nReturn ONLY valid JSON that matches the required schema. "
                "Do not wrap the JSON in commentary."
            )
        ),
        HumanMessage(content=user_prompt),
    ]
    try:
        for attempt in range(1, 4):
            try:
                raw = llm.invoke(fallback_messages)
                content = raw.content if isinstance(raw.content, str) else str(raw.content)
                payload = _extract_json_object(content)
                return schema.model_validate(payload)
            except ValidationError:
                raise
            except Exception as exc:  # noqa: BLE001
                if _is_rate_limited(exc) and attempt < 3:
                    time.sleep(_wait_seconds_for_exception(exc, attempt))
                    continue
                raise
        raise StructuredOutputError(f"{agent_name} could not produce valid structured output.")
    except ValidationError as exc:
        logger.error("%s: fallback JSON failed Pydantic validation", agent_name)
        raise StructuredOutputError(
            f"{agent_name} returned data that did not match the expected schema."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("%s: fallback JSON parse failed: %s", agent_name, type(exc).__name__)
        if _is_rate_limited(exc) or _is_rate_limited(last_error):
            raise GeminiAPIError(sanitize_error_message(exc)) from exc
        if last_error:
            raise StructuredOutputError(
                f"{agent_name} could not produce valid structured output."
            ) from last_error
        raise StructuredOutputError(
            f"{agent_name} could not produce valid structured output."
        ) from exc
