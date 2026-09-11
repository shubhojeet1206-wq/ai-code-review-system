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
@import url("https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap");

html { scroll-behavior: smooth; }
.stAppDeployButton { display: none; }
h3#source, [data-testid="stHeading"]:has(#source) { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stHeader"] { background: transparent; }
div[data-testid="stFileUploaderDropzoneInstructions"] { display: none; }

.stApp {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  background:
    radial-gradient(1100px 520px at -8% -20%, rgba(94, 234, 212, 0.16), transparent 58%),
    radial-gradient(900px 480px at 108% -8%, rgba(56, 189, 248, 0.12), transparent 52%),
    radial-gradient(700px 420px at 70% 110%, rgba(167, 139, 250, 0.08), transparent 55%),
    #070b14;
}
.stApp::before {
  content: "";
  position: fixed;
  width: 420px;
  height: 420px;
  left: -80px;
  top: 120px;
  border-radius: 50%;
  background: rgba(45, 212, 191, 0.09);
  filter: blur(40px);
  pointer-events: none;
  animation: orb-drift 16s ease-in-out infinite;
  z-index: 0;
}
.block-container {
  position: relative;
  z-index: 1;
  padding-top: 1.35rem;
  max-width: 1280px;
  animation: page-in 640ms cubic-bezier(.22,1,.36,1) both;
}

h1, h2, h3, .hero-title, .agent-name {
  font-family: Outfit, "IBM Plex Sans", sans-serif;
}

.hero {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  margin: 0 0 1.35rem;
  padding-bottom: 1.05rem;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.hero-kicker {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  font-size: 0.72rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #7dd3c7;
  margin-bottom: 0.35rem;
}
.hero-kicker::before {
  content: "";
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #5eead4;
  box-shadow: 0 0 12px #5eead4;
  animation: pulse 2s ease-in-out infinite;
}
.hero-title {
  margin: 0;
  font-size: 2.05rem;
  font-weight: 650;
  letter-spacing: -0.03em;
  background: linear-gradient(120deg, #f8fafc 20%, #99f6e4 70%, #7dd3fc 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.hero-sub { margin: 0.35rem 0 0; color: rgba(232,238,249,0.62); font-size: 0.95rem; }
.model-chip {
  flex-shrink: 0;
  border: 1px solid rgba(94,234,212,0.22);
  background: rgba(16, 24, 38, 0.72);
  backdrop-filter: blur(10px);
  border-radius: 999px;
  padding: 0.55rem 0.9rem 0.55rem 0.7rem;
  animation: card-in 700ms cubic-bezier(.22,1,.36,1) both;
}
.model-chip span {
  display: block;
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: rgba(232,238,249,0.5);
}
.model-chip strong {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.86rem;
  color: #99f6e4;
}

iframe[title*="streamlit_ace"] {
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 16px !important;
  overflow: hidden;
  box-shadow: 0 18px 40px rgba(0,0,0,0.22);
  transition: box-shadow 280ms ease, border-color 280ms ease, transform 280ms ease;
}
iframe[title*="streamlit_ace"]:hover {
  border-color: rgba(94,234,212,0.28) !important;
  box-shadow: 0 22px 48px rgba(0,0,0,0.28);
}

[data-testid="stFileUploaderDropzone"] {
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
  border-radius: 14px !important;
  border: 1px dashed rgba(94,234,212,0.28) !important;
  background: rgba(255,255,255,0.03) !important;
  transition: border-color 240ms ease, background 240ms ease, transform 240ms ease;
}
[data-testid="stFileUploaderDropzone"] button {
  margin-left: auto !important;
  margin-right: auto !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: rgba(94,234,212,0.5) !important;
  background: rgba(94,234,212,0.05) !important;
}

div.stButton > button {
  position: relative;
  overflow: hidden;
  height: 3.15rem;
  border: 0 !important;
  border-radius: 16px !important;
  font-family: Outfit, sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.92rem !important;
  color: #042f2e !important;
  background: linear-gradient(135deg, #5eead4 0%, #2dd4bf 46%, #22d3ee 100%) !important;
  box-shadow: 0 14px 32px rgba(34, 211, 238, 0.22), inset 0 1px 0 rgba(255,255,255,0.38);
  transition: transform 220ms cubic-bezier(.22,1,.36,1), box-shadow 220ms ease, filter 220ms ease;
}
div.stButton > button:hover {
  transform: translateY(-2px) scale(1.01);
  filter: brightness(1.06);
  box-shadow: 0 18px 38px rgba(34, 211, 238, 0.34), inset 0 1px 0 rgba(255,255,255,0.45);
}
div.stButton > button:active { transform: translateY(0) scale(0.995); }
div.stButton > button:focus { outline: none; box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.32); }
div.stButton > button::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(110deg, transparent 20%, rgba(255,255,255,0.38) 48%, transparent 72%);
  transform: translateX(-130%);
  animation: shine 3.4s ease-in-out infinite;
}

.agent-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.7rem;
  margin: 0.15rem 0 0.2rem;
}
.agent-card {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.08);
  background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018));
  border-radius: 16px;
  padding: 0.9rem 0.95rem 1rem;
  animation: card-in 560ms cubic-bezier(.22,1,.36,1) both;
  transition: transform 280ms cubic-bezier(.22,1,.36,1), border-color 280ms ease, box-shadow 280ms ease;
}
.agent-card:hover {
  transform: translateY(-3px);
  border-color: rgba(255,255,255,0.16);
  box-shadow: 0 16px 36px rgba(0,0,0,0.22);
}
.agent-card::after {
  content: "";
  position: absolute;
  inset: auto 0 0 0;
  height: 2px;
  background: rgba(255,255,255,0.08);
}
.agent-card.is-running, .agent-card.is-pending {
  border-color: rgba(94, 234, 212, 0.38);
  box-shadow: 0 0 0 1px rgba(94, 234, 212, 0.1), 0 12px 32px rgba(16, 185, 129, 0.1);
}
.agent-card.is-running::after, .agent-card.is-pending::after {
  background: linear-gradient(90deg, transparent, #5eead4, transparent);
  animation: bar-slide 1.15s linear infinite;
}
.agent-card.is-completed { border-color: rgba(94, 234, 212, 0.3); }
.agent-card.is-completed::after { background: #5eead4; }
.agent-card.is-failed { border-color: rgba(251, 113, 133, 0.42); }
.agent-card.is-failed::after { background: #fb7185; }
.agent-card.is-idle { opacity: 0.86; }
.agent-card-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
}
.agent-name { font-weight: 600; letter-spacing: 0.01em; }
.agent-meta { color: rgba(232,238,249,0.52); font-size: 0.8rem; margin-top: 0.32rem; }
.agent-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.7rem;
  text-transform: lowercase;
  letter-spacing: 0.05em;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
}
.agent-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #64748b;
}
.agent-card.is-running .agent-pill, .agent-card.is-pending .agent-pill {
  background: rgba(94, 234, 212, 0.14);
  color: #bbf7d0;
}
.agent-card.is-running .agent-dot, .agent-card.is-pending .agent-dot {
  background: #5eead4;
  box-shadow: 0 0 0 0 rgba(94,234,212,0.55);
  animation: ping 1.5s ease-out infinite;
}
.agent-card.is-completed .agent-pill { background: rgba(94, 234, 212, 0.14); color: #bbf7d0; }
.agent-card.is-completed .agent-dot { background: #5eead4; }
.agent-card.is-failed .agent-pill { background: rgba(251, 113, 133, 0.14); color: #fecaca; }
.agent-card.is-failed .agent-dot { background: #fb7185; }

.review-panel {
  margin-top: 0.35rem;
  padding: 1.05rem 1.1rem 1.15rem;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02));
  animation: card-in 620ms cubic-bezier(.22,1,.36,1) both;
}
.review-kicker {
  font-size: 0.7rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #7dd3c7;
}
.review-panel h3 { margin: 0.15rem 0 0.85rem; font-size: 1.45rem; }
.metric-row {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 0.55rem;
}
.metric {
  padding: 0.7rem 0.65rem 0.75rem;
  border-radius: 14px;
  background: rgba(7, 11, 20, 0.45);
  border: 1px solid rgba(255,255,255,0.06);
  animation: card-in 640ms cubic-bezier(.22,1,.36,1) both;
}
.metric span {
  display: block;
  font-size: 0.7rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(232,238,249,0.5);
}
.metric strong {
  display: block;
  margin-top: 0.2rem;
  font-family: Outfit, sans-serif;
  font-size: 1.45rem;
  font-weight: 650;
}
.metric.score strong { color: #5eead4; }
.metric.critical strong { color: #fb7185; }
.metric.high strong { color: #fb923c; }
.metric.medium strong { color: #fbbf24; }
.metric.low strong { color: #38bdf8; }
.metric.info strong { color: #c4b5fd; }
.review-summary {
  margin: 0.95rem 0 0;
  color: rgba(232,238,249,0.82);
  line-height: 1.55;
}

.finding-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.8rem;
  margin-top: 0.7rem;
}
.finding-card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 0.9rem 0.95rem;
  margin: 0;
  background: rgba(255,255,255,0.03);
  animation: card-in 500ms cubic-bezier(.22,1,.36,1) both;
  transition: transform 260ms cubic-bezier(.22,1,.36,1), border-color 260ms ease, box-shadow 260ms ease;
}
.finding-card:hover {
  transform: translateY(-3px);
  border-color: rgba(255,255,255,0.16);
  box-shadow: 0 14px 30px rgba(0,0,0,0.2);
}
.finding-card .sev { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.04em; }
.finding-card p { margin: 0.45rem 0 0; color: rgba(232,238,249,0.82); line-height: 1.5; }
.finding-card pre {
  margin: 0.55rem 0 0;
  padding: 0.6rem 0.7rem;
  border-radius: 10px;
  background: rgba(0,0,0,0.28);
  overflow-x: auto;
  font-size: 0.8rem;
}
.finding-card.sev-critical { border-color: rgba(251,113,133,0.28); }
.finding-card.sev-critical .sev { color: #fb7185; }
.finding-card.sev-high { border-color: rgba(251,146,60,0.28); }
.finding-card.sev-high .sev { color: #fb923c; }
.finding-card.sev-medium { border-color: rgba(251,191,36,0.24); }
.finding-card.sev-medium .sev { color: #fbbf24; }
.finding-card.sev-low { border-color: rgba(56,189,248,0.24); }
.finding-card.sev-low .sev { color: #38bdf8; }
.finding-card.sev-info { border-color: rgba(196,181,253,0.24); }
.finding-card.sev-info .sev { color: #c4b5fd; }
.finding-grid .finding-card:nth-child(2) { animation-delay: 50ms; }
.finding-grid .finding-card:nth-child(3) { animation-delay: 90ms; }
.finding-grid .finding-card:nth-child(4) { animation-delay: 130ms; }

[data-baseweb="tab-list"] { gap: 0.15rem; }
button[data-baseweb="tab"] {
  transition: color 200ms ease, background 200ms ease !important;
}
.stTabs [data-baseweb="tab-highlight"] {
  transition: transform 280ms cubic-bezier(.22,1,.36,1) !important;
}

@media (max-width: 900px) {
  .agent-grid, .finding-grid, .metric-row { grid-template-columns: 1fr 1fr; }
  .hero { flex-direction: column; align-items: flex-start; }
}
@media (max-width: 640px) {
  .agent-grid, .finding-grid, .metric-row { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation: none !important;
    transition: none !important;
  }
}

@keyframes page-in {
  from { opacity: 0; transform: translateY(14px); }
  to { opacity: 1; transform: none; }
}
@keyframes card-in {
  from { opacity: 0; transform: translateY(10px) scale(0.985); }
  to { opacity: 1; transform: none; }
}
@keyframes bar-slide {
  from { transform: translateX(-45%); }
  to { transform: translateX(45%); }
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.45; }
}
@keyframes ping {
  0% { box-shadow: 0 0 0 0 rgba(94,234,212,0.55); }
  80%, 100% { box-shadow: 0 0 0 8px rgba(94,234,212,0); }
}
@keyframes shine {
  0%, 58% { transform: translateX(-130%); }
  78%, 100% { transform: translateX(130%); }
}
@keyframes orb-drift {
  0%, 100% { transform: translate(0, 0); }
  50% { transform: translate(70px, 40px); }
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
<div class="finding-card sev-{html.escape(finding.severity.value.lower())}">
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
    <span class="agent-pill"><span class="agent-dot"></span>{html.escape(mark)}</span>
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
    st.markdown(
        f"""
<div class="review-panel">
  <div class="review-kicker">Unified report</div>
  <h3>Review</h3>
  <div class="metric-row">
    <div class="metric score" style="animation-delay:40ms"><span>Score</span><strong>{review.overall_score}</strong></div>
    <div class="metric critical" style="animation-delay:80ms"><span>Critical</span><strong>{counts[Severity.CRITICAL.value]}</strong></div>
    <div class="metric high" style="animation-delay:120ms"><span>High</span><strong>{counts[Severity.HIGH.value]}</strong></div>
    <div class="metric medium" style="animation-delay:160ms"><span>Medium</span><strong>{counts[Severity.MEDIUM.value]}</strong></div>
    <div class="metric low" style="animation-delay:200ms"><span>Low</span><strong>{counts[Severity.LOW.value]}</strong></div>
    <div class="metric info" style="animation-delay:240ms"><span>Info</span><strong>{counts[Severity.INFO.value]}</strong></div>
  </div>
  <p class="review-summary">{html.escape(review.summary)}</p>
</div>
""",
        unsafe_allow_html=True,
    )

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
<div class="finding-card sev-{html.escape(fix.severity.value.lower())}">
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
            _render_card_grid(
                [
                    f'<div class="finding-card"><p>{html.escape(note)}</p></div>'
                    for note in review.agent_summaries
                ],
                empty_text="No agent notes.",
            )
        else:
            st.caption("No agent notes.")


def main() -> None:
    st.set_page_config(
        page_title="Code Sentinel",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _init_session()
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.markdown(
        f"""
<div class="hero">
  <div>
    <div class="hero-kicker">Multi-agent review</div>
    <h1 class="hero-title">Code Sentinel</h1>
    <p class="hero-sub">LangGraph specialists + Gemini. Upload wins over pasted code.</p>
  </div>
  <div class="model-chip">
    <span>Model</span>
    <strong>{html.escape(get_gemini_model_name())}</strong>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1.15], gap="large")

    with left:
        language = st.selectbox("Language", SUPPORTED_LANGUAGES, index=0)
        st.caption("Code")
        pasted = st_ace(
            placeholder="Paste source code…",
            language=ACE_MODES.get(language, "python"),
            theme="nord_dark",
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
