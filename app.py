"""Streamlit frontend for the multi-agent AI code review system."""

from __future__ import annotations

from typing import Any

import streamlit as st

from llm.client import ConfigurationError, get_gemini_model_name
from models.review_models import FinalReview, Finding, Severity
from orchestration.graph import NODE_LABELS, SPECIALIST_NODES, get_review_graph
from orchestration.state import create_initial_state
from utils.file_utils import (
    SUPPORTED_LANGUAGES,
    InvalidSourceFileError,
    UnsupportedFileTypeError,
    language_from_filename,
    read_uploaded_source,
)
from utils.logging_config import configure_logging, get_logger, redact_secrets

configure_logging()
logger = get_logger(__name__)

STATUS_ORDER = list(SPECIALIST_NODES) + ["final_review"]

APP_CSS = """
<style>
.stAppDeployButton { display: none; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
.block-container { padding-top: 1.6rem; max-width: 1180px; }
[data-testid="stHeader"] { background: transparent; }
div[data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important; }
.stButton > button { height: 2.6rem; }
</style>
"""


def _init_session() -> None:
    st.session_state.setdefault("agent_status", {name: "idle" for name in STATUS_ORDER})
    st.session_state.setdefault("final_review", None)
    st.session_state.setdefault("workflow_errors", [])
    st.session_state.setdefault("review_ran", False)
    st.session_state.setdefault("review_origin", "")


def _count_severities(review: FinalReview) -> dict[str, int]:
    buckets = {severity.value: 0 for severity in Severity}
    for finding in _all_findings(review):
        buckets[finding.severity.value] += 1
    return buckets


def _all_findings(review: FinalReview) -> list[Finding]:
    seen: set[tuple[str, str]] = set()
    ordered: list[Finding] = []
    for group in (
        review.critical_issues,
        review.bugs,
        review.security_vulnerabilities,
        review.performance_problems,
        review.code_quality_issues,
    ):
        for finding in group:
            key = (finding.title, finding.category)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(finding)
    return ordered


def _render_finding(finding: Finding) -> None:
    lines = ""
    if finding.line_start:
        end = finding.line_end or finding.line_start
        lines = f" · lines {finding.line_start}–{end}"
    st.markdown(f"**{finding.severity.value}** · {finding.title}{lines}")
    st.write(finding.description)
    if finding.code_snippet:
        st.code(finding.code_snippet, language=None)
    if finding.suggested_fix:
        st.markdown(f"**Suggested fix:** {finding.suggested_fix}")
    st.caption(f"{finding.category} · confidence {finding.confidence:.0%}")


def _render_finding_group(findings: list[Finding], *, empty_text: str) -> None:
    if not findings:
        st.caption(empty_text)
        return
    for finding in findings:
        expanded = finding.severity in {Severity.CRITICAL, Severity.HIGH}
        with st.expander(f"{finding.severity.value}: {finding.title}", expanded=expanded):
            _render_finding(finding)


def _resolve_source(uploaded, pasted: str) -> tuple[str, str]:
    if uploaded is not None:
        text = read_uploaded_source(uploaded, uploaded.name)
        return text, f"uploaded file ({uploaded.name})"
    return pasted, "pasted editor"


def _status_mark(status: str) -> str:
    return {
        "completed": "done",
        "failed": "failed",
        "pending": "running",
        "running": "running",
        "idle": "idle",
    }.get(status, status)


def _render_status_panel() -> None:
    st.subheader("Agents")
    statuses: dict[str, str] = st.session_state.agent_status
    for node in STATUS_ORDER:
        left, right = st.columns([3.2, 1.2])
        left.write(NODE_LABELS[node])
        right.caption(_status_mark(statuses.get(node, "idle")))


def _coerce_final_review(payload: Any) -> FinalReview | None:
    if payload is None:
        return None
    if isinstance(payload, FinalReview):
        return payload
    return FinalReview.model_validate(payload)


def _run_review(source_code: str, language: str, status_slot: Any) -> None:
    st.session_state.agent_status = {name: "pending" for name in STATUS_ORDER}
    st.session_state.final_review = None
    st.session_state.workflow_errors = []
    st.session_state.review_ran = True

    graph = get_review_graph()
    initial = create_initial_state(source_code, language)

    def paint() -> None:
        with status_slot.container():
            _render_status_panel()

    paint()
    logger.info("UI starting review workflow language=%s chars=%s", language, len(source_code))

    try:
        for event in graph.stream(initial, stream_mode="updates"):
            for node_name, update in event.items():
                if node_name in st.session_state.agent_status:
                    node_status = "completed"
                    if isinstance(update, dict):
                        reported = (update.get("agent_status") or {}).get(node_name)
                        if reported:
                            node_status = reported
                    st.session_state.agent_status[node_name] = node_status
                    if isinstance(update, dict) and update.get("errors"):
                        st.session_state.workflow_errors.extend(update["errors"])
                    if isinstance(update, dict) and update.get("final_review") is not None:
                        st.session_state.final_review = _coerce_final_review(update["final_review"])
                paint()
    except ConfigurationError as exc:
        logger.error("Review aborted: configuration error")
        st.session_state.workflow_errors.append(str(exc))
        st.session_state.agent_status = {
            name: "failed" if st.session_state.agent_status[name] != "completed" else "completed"
            for name in STATUS_ORDER
        }
        paint()
        return
    except Exception as exc:  # noqa: BLE001
        message = redact_secrets(str(exc))[:400]
        logger.error("Review workflow failed: %s", type(exc).__name__)
        st.session_state.workflow_errors.append(message)
        st.session_state.agent_status = {
            name: "failed" if st.session_state.agent_status[name] != "completed" else "completed"
            for name in STATUS_ORDER
        }
        paint()


def _render_dashboard(review: FinalReview) -> None:
    counts = _count_severities(review)
    st.subheader("Review")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Score", f"{review.overall_score}")
    m2.metric("Critical", counts[Severity.CRITICAL.value])
    m3.metric("High", counts[Severity.HIGH.value])
    m4.metric("Medium", counts[Severity.MEDIUM.value])
    m5.metric("Low", counts[Severity.LOW.value])
    m6.metric("Info", counts[Severity.INFO.value])
    st.write(review.summary)

    tabs = st.tabs(
        ["Critical", "Bugs", "Security", "Performance", "Quality", "Fixes", "Tests", "Agents"]
    )
    with tabs[0]:
        _render_finding_group(review.critical_issues, empty_text="No critical issues.")
    with tabs[1]:
        _render_finding_group(review.bugs, empty_text="No additional bug findings.")
    with tabs[2]:
        _render_finding_group(
            review.security_vulnerabilities,
            empty_text="No additional security findings.",
        )
    with tabs[3]:
        _render_finding_group(
            review.performance_problems,
            empty_text="No additional performance findings.",
        )
    with tabs[4]:
        _render_finding_group(
            review.code_quality_issues,
            empty_text="No additional quality findings.",
        )
    with tabs[5]:
        if not review.suggested_fixes:
            st.caption("No suggested fixes.")
        else:
            for fix in review.suggested_fixes:
                with st.expander(f"{fix.severity.value}: {fix.title}"):
                    st.write(fix.description)
                    if fix.related_finding_title:
                        st.caption(f"Related: {fix.related_finding_title}")
    with tabs[6]:
        if not review.recommended_test_cases:
            st.caption("No recommended tests.")
        else:
            st.dataframe(
                [
                    {
                        "Title": test.title,
                        "Type": test.test_type,
                        "Description": test.description,
                    }
                    for test in review.recommended_test_cases
                ],
                use_container_width=True,
                hide_index=True,
            )
    with tabs[7]:
        if review.agent_summaries:
            for note in review.agent_summaries:
                st.write(f"- {note}")
        else:
            st.caption("No agent notes.")


def main() -> None:
    st.set_page_config(
        page_title="AI Code Review",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _init_session()
    st.markdown(APP_CSS, unsafe_allow_html=True)

    title_col, model_col = st.columns([3.5, 1.5])
    with title_col:
        st.title("AI Code Review")
        st.caption("LangGraph specialists + Gemini. Upload wins over pasted code.")
    with model_col:
        st.caption("Model")
        st.code(get_gemini_model_name(), language=None)

    left, right = st.columns([1.05, 1.15], gap="large")

    with left:
        st.subheader("Source")
        language = st.selectbox("Language", SUPPORTED_LANGUAGES, index=0)
        pasted = st.text_area("Code", height=300, placeholder="Paste source code…")
        uploaded = st.file_uploader(
            "Or upload a file",
            type=["py", "java", "js", "mjs", "cjs", "ts", "tsx", "cpp", "cc", "cxx", "hpp", "c", "h", "go", "sql"],
        )
        detected = language_from_filename(uploaded.name) if uploaded is not None else None
        if detected:
            st.caption(f"Detected from filename: {detected}. Agents use the language selected above.")
        run_clicked = st.button("Review Code", type="primary", use_container_width=True)

    with right:
        status_slot = st.empty()
        with status_slot.container():
            _render_status_panel()

    if run_clicked:
        try:
            source, origin = _resolve_source(uploaded, pasted)
        except (UnsupportedFileTypeError, InvalidSourceFileError) as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(redact_secrets(str(exc)))
            return

        if not source.strip():
            st.error("Source code is empty. Paste code or upload a file.")
            return

        st.session_state.review_origin = origin
        _run_review(source, language, status_slot)

    with right:
        errors = st.session_state.workflow_errors
        if errors:
            st.error("Some agents reported errors. Partial results may still be shown.")
            for item in errors:
                st.caption(redact_secrets(item))

        review = st.session_state.final_review
        if review is not None:
            _render_dashboard(review)
        elif st.session_state.review_ran:
            st.info("The workflow finished without a final review payload.")
        else:
            st.caption("Results will appear here after you click Review Code.")


if __name__ == "__main__":
    main()
