"""
# AI Code Review System

Multi-agent code review application built with **Python**, **LangChain**, **LangGraph**, **Google Gemini**, **Pydantic**, and **Streamlit**. Five specialist agents inspect the same source file in parallel. A final aggregator agent de-duplicates findings, ranks them by severity, and produces a unified review.

This project is designed to be explained in a technical interview: real Gemini calls, structured outputs, and an explicit graph workflow — not a single prompt pretending to be many reviewers.

---

## 1. Project overview

Engineers paste or upload source code, choose a language, and run **Review Code**. The UI shows each agent as it finishes, then a dashboard with:

- overall quality score (0–100)
- severity counts
- critical issues, bugs, security, performance, and quality findings
- suggested fixes
- recommended tests

Every specialist agent and the final aggregator call the **Google Gemini API** through LangChain. There are no hardcoded or simulated model replies.

---

## 2. Problem being solved

Manual review is slow and uneven. A single LLM prompt that says “review this code” tends to mix concerns, miss whole classes of issues, and return unstructured prose that is hard to display or merge.

This system splits review into **specialized roles**, forces **structured findings**, and **aggregates** them so the user sees one prioritized report.

---

## 3. Architecture

```
                    ┌─ Bug Detection Agent ─────┐
                    │                           │
                    ├─ Security Agent ──────────┤
 Input code +       │                           │
 language  ────────►├─ Performance Agent ───────┼──► Final Review Agent ──► Streamlit
 (Streamlit)        │                           │     (dedupe, score)
                    ├─ Code Quality Agent ──────┤
                    │                           │
                    └─ Testing Agent ───────────┘
                              ▲
                              │
                     LangGraph StateGraph
                     Gemini via llm/client.py
                     Pydantic AgentReview / FinalReview
```

Layers:

| Layer | Responsibility |
| --- | --- |
| `app.py` | Input, status, dashboard |
| `orchestration/` | LangGraph workflow and state |
| `agents/` | Prompts and node functions |
| `llm/client.py` | Gemini model, retries, schema validation |
| `models/` | Pydantic contracts |
| `utils/` | Logging and file validation |

---

## 4. Why LangGraph is used

LangGraph is the **orchestration layer**. It owns:

- shared **state** (code, language, list of `AgentReview`, errors, statuses, `FinalReview`)
- **edges** from `START` to five specialists (parallel fan-out)
- a **fan-in** edge so the Final Review Agent waits for every specialist
- **reducers** so parallel writes to `agent_reviews` and `errors` merge instead of colliding

Without a graph, you would glue `ThreadPoolExecutor` calls together and hope merge logic stays consistent. The graph makes the control flow visible and interviewable.

---

## 5. Why multiple specialized agents are used

A single “do everything” prompt competes with itself: security findings drown out tests; style crowds out bugs. Separate agents:

- keep **system prompts** focused
- fail **independently** (one timeout does not wipe the other reviews)
- can later be **scaled, cached, or swapped** per domain
- produce comparable `Finding` objects that the aggregator can merge

---

## 6. Role of each agent

| Agent | Looks for |
| --- | --- |
| **Bug Detection** | Logic errors, edge cases, runtime failures, wrong assumptions |
| **Security** | Injection, secrets, authz, path traversal, unsafe handling |
| **Performance** | Complexity, wasted work, N+1 I/O, memory growth |
| **Code Quality** | Naming, structure, duplication, error handling, idioms |
| **Testing** | Missing tests, boundaries, failure paths, regressions |
| **Final Review** | Dedupe, merge, prioritize, score, suggested fixes and tests |

---

## 7. LangGraph workflow

1. Streamlit validates input and builds `ReviewState`.
2. `graph.stream(...)` runs the five specialist nodes from `START`.
3. Each node calls Gemini with `AgentReview` structured output and appends to `agent_reviews`.
4. After all five complete, `final_review` runs.
5. The Final Review Agent receives **JSON of all `AgentReview` objects**, not raw essays.
6. Streamlit updates status from stream events and renders `FinalReview`.

Fan-in is explicit:

```python
graph.add_edge(list(SPECIALIST_NODES), "final_review")
```

---

## 8. Data flow

1. User pastes code and/or uploads a file (upload wins).
2. Language selector value is sent to every agent.
3. Code is numbered for citation (`utils/file_utils.py`) inside the Gemini prompt.
4. Each specialist returns `AgentReview` (list of `Finding`).
5. Final agent returns `FinalReview`.
6. UI groups findings and shows metrics.

State stays structured: lists of models, status maps, and error strings — not a growing blob of free text.

---

## 9. Pydantic structured outputs

`Finding`, `AgentReview`, and `FinalReview` live in `models/review_models.py`.

Gemini is invoked with LangChain `with_structured_output(..., method="json_mode")` (the method supported by `langchain-google-genai` 2.x). The result is validated with Pydantic. If the payload is malformed, the client retries, then tries a JSON parse of a raw completion. Invalid data is never silently accepted.

Severity is an enum: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFO`.

---

## 10. Gemini integration

- Package: `langchain-google-genai` (`ChatGoogleGenerativeAI`)
- API key: environment variable `GOOGLE_API_KEY` (loaded from `.env` via `python-dotenv`)
- Model: environment variable `GEMINI_MODEL`

**Not every Gemini model is enabled for every Google account.** If you see a 404 / “model not found” error, set `GEMINI_MODEL` to a model listed in [Google AI Studio](https://aistudio.google.com/).

All agents use `llm/client.py`. They do not construct their own unrelated clients.

The Gemini client is created **lazily**, so Streamlit can start even before a key is set. The key is required when you click **Review Code**.

---

## 11. Final review aggregation

The Final Review Agent is instructed to:

- drop duplicates
- merge related findings
- resolve conflicts when possible (prefer specific, high-confidence, higher-severity evidence)
- sort by severity
- keep line numbers and snippets
- emit suggested fixes and recommended tests
- compute `overall_score` (start at 100; deduct for remaining findings)

If that Gemini call fails after specialists succeeded, a **deterministic local merge** still fills the dashboard so the UI does not crash. That fallback only aggregates **real specialist outputs**; it does not invent Gemini findings.

---

## 12. Project structure

```
ai-code-review-system/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── agents/
│   ├── __init__.py
│   ├── base.py              # shared Gemini invoke + failure handling
│   ├── bug_agent.py
│   ├── security_agent.py
│   ├── performance_agent.py
│   ├── quality_agent.py
│   ├── testing_agent.py
│   └── final_review_agent.py
├── orchestration/
│   ├── __init__.py
│   ├── graph.py
│   └── state.py
├── models/
│   ├── __init__.py
│   └── review_models.py
├── llm/
│   ├── __init__.py
│   └── client.py
├── utils/
│   ├── __init__.py
│   ├── logging_config.py
│   └── file_utils.py
└── tests/
    ├── test_models.py
    └── test_utils.py
```

---

## 13. Environment setup

- Python **3.11+**
- A Google AI Studio (Gemini) API key
- Network access to the Gemini API when running a review

---

## 14. Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 15. How to run locally

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 16. How to configure the Gemini API

1. Copy the example env file:

   ```bash
   copy .env.example .env
   ```

   On macOS/Linux: `cp .env.example .env`

2. Edit `.env`:

   ```
   GOOGLE_API_KEY=your_real_key
   GEMINI_MODEL=gemini-3.5-flash-lite
   ```

3. If `gemini-3.5-flash-lite` is not available or its free-tier quota is exhausted (Google currently allows 20 generate requests/day per model on the free tier), set `GEMINI_MODEL` to another model listed for your key. Google may return a 404 naming a replacement, or a 429 naming the quota metric.

Never commit `.env`. Never put the key in source files.

---

## 17. Example usage

1. Start Streamlit.
2. Choose **Python** (or another supported language).
3. Paste a function, or upload `.py` / `.java` / `.js` / `.ts` / `.cpp` / `.c` / `.go` / `.sql`.
4. Click **Review Code**.
5. Watch agent statuses move from pending → completed (or failed).
6. Read the score, critical issues, and suggested tests.

**Input priority:** if both an upload and pasted text are present, **the uploaded file is reviewed** and the text area is ignored.

Supported languages: Python, Java, JavaScript, TypeScript, C++, C, Go, SQL.

---

## 18. Error handling

| Situation | Behavior |
| --- | --- |
| Missing `GOOGLE_API_KEY` | Clear configuration error in the UI; app still starts |
| Invalid key / unauthorized | User-safe message; no key logged |
| Rate limit / quota | Classified Gemini error; specialist marked failed |
| Model not available | Message tells you to change `GEMINI_MODEL` |
| Empty code | Blocked in the UI before the graph runs |
| Bad upload type/encoding/size | `UnsupportedFileTypeError` / `InvalidSourceFileError` |
| Malformed Gemini JSON | Retry, JSON fallback, then agent failure record |
| One specialist fails | Other specialists still run; final agent sees the error list |

Secrets and full source listings are not written to logs.

---

## 19. Testing

```bash
pytest
```

Tests cover Pydantic validation and file/redaction helpers. They do **not** call Gemini (no fake reviews, and no requirement for a key in CI).

---

## 20. Limitations

- Quality depends on the Gemini model and the size of the snippet.
- Very large files are rejected or truncated in the prompt.
- Agents can still overlap; the final agent reduces but may not eliminate all duplication.
- Line numbers are model-cited; they can be slightly off.
- Parallel Gemini calls can hit rate limits on free-tier keys.
- This is a review assistant, not a substitute for SAST, tests, or human review.

---

## 21. Possible future improvements

- Per-agent model routing (stronger model for security)
- Persistent review history
- Git diff / pull-request mode
- Optional RAG over internal coding standards
- Human-in-the-loop confirmation of auto-fixes
- Swap `llm/client.py` to another provider without changing agents

---

## Interview Explanation

Use this section as a walkthrough in a technical interview.

### Why multi-agent architecture?

Review is not one skill. Bugs, security, performance, quality, and testing pull the model in different directions. Separate agents with separate prompts produce cleaner, more complete coverage and isolate failures.

### Why LangGraph?

We need **parallel specialists** and a **join** before aggregation. LangGraph’s `StateGraph`, reducers, and explicit edges encode that workflow. Streamlit can `stream` node updates for live status.

### Why Gemini?

Gemini supports strong code reasoning and **native JSON structured output** (`json_mode` / `response_schema`) that maps onto Pydantic models. The model id is configurable because availability differs by account.

### Why structured outputs?

Downstream UI and the aggregator need fields (`severity`, `line_start`, `suggested_fix`), not paragraphs. Pydantic validation is the contract between the LLM and the rest of the system.

### How do agents communicate?

They do **not** chat with each other. They write `AgentReview` objects into shared graph state. The Final Review Agent reads that list as JSON.

### How are duplicate findings handled?

The Final Review Agent is prompted to drop duplicates and merge related items. If that call fails, a local fallback de-dupes by title, category, and snippet.

### How is severity determined?

Each specialist assigns `Severity` using rubric text in its system prompt. The aggregator keeps the more serious view when findings conflict and deducts from a 100-point score using fixed weights.

### How are failures handled?

Each specialist node catches API, validation, and unexpected errors, stores a failed `AgentReview`, and lets the graph continue. The UI shows errors without API keys.

### Why is this architecture scalable?

You can add a new node (for example “API design”) with the same `Finding` schema, attach it to the fan-out/fan-in, and the aggregator consumes one more `AgentReview`. Specialists can be moved to separate workers later because they are already independent.

### How could the LLM provider change?

Only `llm/client.py` talks to Gemini. Replace `ChatGoogleGenerativeAI` with another LangChain chat model that supports `with_structured_output`, keep the same Pydantic schemas, and the agents and graph stay the same.

---

## License

Interview / portfolio project. Use at your own risk on proprietary code — you are sending source to the Gemini API.
