"""Streamlit frontend for the multi-agent AI code review system."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st
from streamlit_ace import st_ace

from llm.client import ConfigurationError, get_gemini_model_name
from models.review_models import AgentReview, FinalReview, Finding, Severity
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
h3#source, [data-testid="stHeading"]:has(#source) { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
.block-container { padding-top: 1.4rem; max-width: 1280px; }
[data-testid="stHeader"] { background: transparent; }
div[data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
iframe[title*="streamlit_ace"] {
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
  overflow: hidden;
}

div.stButton > button {
  height: 3.05rem;
  border: 0 !important;
  border-radius: 14px !important;
  font-weight: 700 !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-size: 0.92rem !important;
  color: #042f2e !important;
  background: linear-gradient(135deg, #5eead4 0%, #2dd4bf 42%, #22d3ee 100%) !important;
  box-shadow: 0 12px 28px rgba(34, 211, 238, 0.22), inset 0 1px 0 rgba(255,255,255,0.35);
  transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
}
div.stButton > button:hover {
  transform: translateY(-2px);
  filter: brightness(1.06);
  box-shadow: 0 16px 34px rgba(34, 211, 238, 0.32), inset 0 1px 0 rgba(255,255,255,0.4);
}
div.stButton > button:active { transform: translateY(0); }
div.stButton > button:focus { outline: none; box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.35); }

.agent-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.65rem;
  margin: 0.2rem 0 0.4rem;
}
.agent-card {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.08);
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
  border-radius: 14px;
  padding: 0.85rem 0.95rem 0.95rem;
  animation: card-in 420ms ease both;
}
.agent-card::after {
  content: "";
  position: absolute;
  inset: auto 0 0 0;
  height: 2px;
  background: rgba(255,255,255,0.08);
}
.agent-card.is-running, .agent-card.is-pending {
  border-color: rgba(110, 231, 183, 0.35);
  box-shadow: 0 0 0 1px rgba(110, 231, 183, 0.12), 0 10px 30px rgba(16, 185, 129, 0.08);
}
.agent-card.is-running::after, .agent-card.is-pending::after {
  background: linear-gradient(90deg, transparent, #6ee7b7, transparent);
  animation: bar-slide 1.2s linear infinite;
}
.agent-card.is-completed {
  border-color: rgba(110, 231, 183, 0.28);
}
.agent-card.is-completed::after { background: #6ee7b7; }
.agent-card.is-failed {
  border-color: rgba(248, 113, 113, 0.45);
}
.agent-card.is-failed::after { background: #f87171; }
.agent-card.is-idle { opacity: 0.78; }
.agent-card-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
}
.agent-name { font-weight: 600; letter-spacing: 0.01em; }
.agent-meta { color: rgba(255,255,255,0.55); font-size: 0.8rem; margin-top: 0.28rem; }
.agent-pill {
  font-size: 0.72rem;
  text-transform: lowercase;
  letter-spacing: 0.04em;
  padding: 0.18rem 0.5rem;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
}
.agent-card.is-running .agent-pill, .agent-card.is-pending .agent-pill {
  background: rgba(110, 231, 183, 0.16);
  color: #bbf7d0;
  animation: pulse 1.4s ease-in-out infinite;
}
.agent-card.is-completed .agent-pill { background: rgba(110, 231, 183, 0.16); color: #bbf7d0; }
.agent-card.is-failed .agent-pill { background: rgba(248, 113, 113, 0.16); color: #fecaca; }
.finding-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
  margin-top: 0.6rem;
}
.finding-card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  padding: 0.85rem 0.95rem;
  margin: 0;
  background: rgba(255,255,255,0.03);
  animation: card-in 380ms ease both;
}
.finding-card .sev { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; }
.finding-card p { margin: 0.45rem 0 0; }
@media (max-width: 900px) {
  .agent-grid, .finding-grid { grid-template-columns: 1fr; }
}
@keyframes card-in {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: none; }
}
@keyframes bar-slide {
  from { transform: translateX(-40%); }
  to { transform: translateX(40%); }
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.45; }
}
</style>
"""


AGENT_NODE_NAMES: dict[str, str] = {
    "bug_agent": "Bug Detection Agent",
    "security_agent": "Security Agent",
    "performance_agent": "Performance Agent",
    "quality_agent": "Code Quality Agent",
    "testing_agent": "Testing Agent",
    "final_review": "Final Review Agent",
}

ACE_MODES: dict[str, str] = {
    "Python": "python",
    "Java": "java",
    "JavaScript": "javascript",
    "TypeScript": "typescript",
    "C++": "c_cpp",
    "C": "c_cpp",
    "Go": "golang",
    "SQL": "sql",
}


def _init_session() -> None:
    st.session_state.setdefault("agent_status", {name: "idle" for name in STATUS_ORDER})
    st.session_state.setdefault("final_review", None)
    st.session_state.setdefault("agent_reviews", [])
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


def _finding_card_html(finding: Finding) -> str:
    lines = ""
    if finding.line_start:
        end = finding.line_end or finding.line_start
        lines = f" · lines {finding.line_start}–{end}"
    title = html.escape(finding.title)
    desc = html.escape(finding.description)
    category = html.escape(finding.category)
    sev = html.escape(finding.severity.value)
    snippet = html.escape(finding.code_snippet) if finding.code_snippet else ""
    fix = html.escape(finding.suggested_fix) if finding.suggested_fix else ""
    snippet_html = f"<pre><code>{snippet}</code></pre>" if snippet else ""
    fix_html = f"<p><strong>Suggested fix:</strong> {fix}</p>" if fix else ""
    return f"""
<div class="finding-card">
  <div class="sev">{sev} · {title}{html.escape(lines)}</div>
  <p>{desc}</p>
  {snippet_html}
  {fix_html}
  <p style="opacity:.6;font-size:.8rem;">{category} · confidence {finding.confidence:.0%}</p>
</div>
"""


def _render_card_grid(cards: list[str], *, empty_text: str) -> None:
    if not cards:
        st.caption(empty_text)
        return
    st.markdown(
        f'<div class="finding-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _render_finding_group(findings: list[Finding], *, empty_text: str) -> None:
    _render_card_grid([_finding_card_html(item) for item in findings], empty_text=empty_text)


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


def _finding_count_for_node(node: str) -> int | None:
    target = AGENT_NODE_NAMES.get(node)
    if not target:
        return None
    for review in st.session_state.get("agent_reviews") or []:
        name = review.agent_name if isinstance(review, AgentReview) else review.get("agent_name")
        if name == target:
            findings = review.findings if isinstance(review, AgentReview) else review.get("findings") or []
            return len(findings)
    return None


def _render_status_panel() -> None:
    st.subheader("Agents")
    statuses: dict[str, str] = st.session_state.agent_status
    cards: list[str] = ['<div class="agent-grid">']
    for index, node in enumerate(STATUS_ORDER):
        status = statuses.get(node, "idle")
        mark = _status_mark(status)
        count = _finding_count_for_node(node)
        meta = "Waiting to start"
        if status in {"pending", "running"}:
            meta = "Reviewing source with Gemini…"
        elif status == "completed":
            if node == "final_review":
                meta = "Merged specialist findings"
            elif count is None:
                meta = "Finished"
            else:
                meta = f"{count} finding{'s' if count != 1 else ''}"
        elif status == "failed":
            meta = "Agent reported an error"
        delay = min(index * 70, 350)
        cards.append(
            f"""
<div class="agent-card is-{html.escape(status)}" style="animation-delay:{delay}ms">
  <div class="agent-card-top">
    <span class="agent-name">{html.escape(NODE_LABELS[node])}</span>
    <span class="agent-pill">{html.escape(mark)}</span>
  </div>
  <div class="agent-meta">{html.escape(meta)}</div>
</div>
"""
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


def _coerce_final_review(payload: Any) -> FinalReview | None:
    if payload is None:
        return None
    if isinstance(payload, FinalReview):
        return payload
    return FinalReview.model_validate(payload)


def _coerce_agent_reviews(payload: Any) -> list[AgentReview]:
    if not payload:
        return []
    reviews: list[AgentReview] = []
    for item in payload:
        if isinstance(item, AgentReview):
            reviews.append(item)
        else:
            reviews.append(AgentReview.model_validate(item))
    return reviews


def _run_review(source_code: str, language: str, status_slot: Any) -> None:
    st.session_state.agent_status = {name: "pending" for name in STATUS_ORDER}
    st.session_state.final_review = None
    st.session_state.agent_reviews = []
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
                    if isinstance(update, dict) and update.get("agent_reviews"):
                        st.session_state.agent_reviews.extend(
                            _coerce_agent_reviews(update["agent_reviews"])
                        )
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

    all_findings = _all_findings(review)
    tabs = st.tabs(
        [
            "All",
            "Critical",
            "Bugs",
            "Security",
            "Performance",
            "Quality",
            "Fixes",
            "Tests",
            "Agents",
        ]
    )
    with tabs[0]:
        _render_finding_group(all_findings, empty_text="No findings were returned.")
    with tabs[1]:
        _render_finding_group(review.critical_issues, empty_text="No critical issues.")
    with tabs[2]:
        _render_finding_group(review.bugs, empty_text="No bug findings.")
    with tabs[3]:
        _render_finding_group(
            review.security_vulnerabilities,
            empty_text="No security findings.",
        )
    with tabs[4]:
        _render_finding_group(
            review.performance_problems,
            empty_text="No performance findings.",
        )
    with tabs[5]:
        _render_finding_group(
            review.code_quality_issues,
            empty_text="No quality findings.",
        )
    with tabs[6]:
        _render_card_grid(
            [
                f"""
<div class="finding-card">
  <div class="sev">{html.escape(fix.severity.value)} · {html.escape(fix.title)}</div>
  <p>{html.escape(fix.description)}</p>
  <p style="opacity:.6;font-size:.8rem;">{html.escape(fix.related_finding_title or "")}</p>
</div>
"""
                for fix in review.suggested_fixes
            ],
            empty_text="No suggested fixes.",
        )
    with tabs[7]:
        _render_card_grid(
            [
                f"""
<div class="finding-card">
  <div class="sev">{html.escape(test.test_type)} · {html.escape(test.title)}</div>
  <p>{html.escape(test.description)}</p>
</div>
"""
                for test in review.recommended_test_cases
            ],
            empty_text="No recommended tests.",
        )
    with tabs[8]:
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
        language = st.selectbox("Language", SUPPORTED_LANGUAGES, index=0)
        st.caption("Code")
        pasted = st_ace(
            placeholder="Paste source code…",
            language=ACE_MODES.get(language, "python"),
            theme="tomorrow_night",
            keybinding="vscode",
            font_size=14,
            tab_size=4,
            height=280,
            wrap=True,
            show_gutter=True,
            show_print_margin=False,
            auto_update=True,
            key="source_code_editor",
        ) or ""
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

    errors = st.session_state.workflow_errors
    if errors:
        st.error("Some agents reported errors. Partial results may still be shown.")
        for item in errors:
            st.caption(redact_secrets(item))

    review = st.session_state.final_review
    specialist_reviews = _coerce_agent_reviews(st.session_state.get("agent_reviews"))
    if review is not None and specialist_reviews:
        from agents.final_review_agent import coalesce_final_review

        review = coalesce_final_review(review, specialist_reviews, errors)
    if review is not None:
        _render_dashboard(review)
    elif st.session_state.review_ran:
        st.info("The workflow finished without a final review payload.")


if __name__ == "__main__":
    main()
