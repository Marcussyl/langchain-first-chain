# langchain-first-chain

Hands-on starter for **LangChain LCEL**: Prompt → LLM → output parser (or structured Pydantic fields), running fully local with **Ollama** (no API key).

Learning notes (HTML, open in a browser): [`docs/study_notes.html`](docs/study_notes.html).  
Next topics / what to change next: [`docs/learning-roadmap.html`](docs/learning-roadmap.html).

## What you learn

**`first_chain.py` (one-shot explain):**

- `ChatPromptTemplate.from_messages` with a system role and a `{topic}` human message
- Passing a dict into LCEL: `chain.invoke({"topic": ...})`
- `ChatOllama` to call a local model
- `StrOutputParser` to get a plain string
- LCEL piping with `|`: `prompt | model | parser`

**`extract_paragraph.py` (structured output):**

- A Pydantic model (`ParagraphExtract`) with `summary`, `paragraph_count`, `author`, `published_at`
- `ChatOllama.with_structured_output(..., method='json_schema', include_raw=True)` so Ollama fills those fields
- LCEL: `prompt | structured_model` (no `StrOutputParser`); `invoke` waits for the parsed object
- Optional `author` / `published_at` stay empty when the text does not state them (do not invent)
- A Python paragraph count printed next to the model's `paragraph_count` so you can see miscounts
- Each paste is independent (no chat history)

**`chatbot.py` (multi-turn):**

- `HumanMessage` / `AIMessage` / `SystemMessage` as the real chat history
- `MessagesPlaceholder` to feed that list into the chain
- Short-term **memory**: later turns see earlier messages
- **Soft** compression: old turns become a running summary on the `SystemMessage`
- **Hard** compression: `trim_messages` with `token_counter='approximate'` (`strategy='last'`, `include_system=True`, `start_on='human'`) so a cut never starts on a dangling `ToolMessage`
- **One tool**: `get_time` via `@tool` + `bind_tools`; a hand-written round appends `ToolMessage` (not `create_agent`)
- **Replies**: `invoke` so we can see `tool_calls` before printing (`StrOutputParser` / `stream` would hide them)
- **Persist sessions**: `facts` + topic notes + recent human/ai/**tool** turns in `sessions/<session_id>.json` (system message is rebuilt on load)

**`graph_chatbot.py` (LangGraph CLI twin):**

- Same harvest / soft summary / hard trim / `get_time` as `chatbot.py` (via `graph_core.py`)
- `StateGraph` nodes `ingest` → `prepare` → `generate` instead of a `while True` body
- SQLite checkpointer (`sessions/langgraph.sqlite`); `thread_id` is the session id
- Does **not** use `create_agent`; does **not** share `sessions/*.json` with `chatbot.py`

**Dev server (`langgraph dev` + Studio):**

- [`langgraph.json`](langgraph.json) points at `graph_server.py:graph` (graph id **`tutor`**)
- Same `graph_core` pipeline; Studio persistence is provided by the LangGraph API (do **not** attach SqliteSaver in `graph_server.py`)
- Model name from `.env` / `OLLAMA_MODEL` (see `.env.example`)

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (Windows: `winget install astral-sh.uv`, or the [official installer](https://docs.astral.sh/uv/getting-started/installation/))
- [Ollama](https://ollama.com) installed and running

## Setup

```bash
# Pull the model (once)
ollama pull llama3.2

# Project env (from repo root)
uv sync

# Optional: local overrides (model name, LangSmith key)
copy .env.example .env   # Windows CMD; or Copy-Item .env.example .env in PowerShell
```

If your machine does not have the Python in `.python-version` yet:

```bash
uv python install 3.13
```

If Ollama is not already running:

```bash
ollama serve
```

## Run

Use `uv run` so you do not need to activate `.venv` first. After `uv sync`, you can also activate `.venv` and run `python` as usual.

One topic at a time (no memory between questions):

```bash
uv run python first_chain.py
```

Chatbot that remembers this session (old turns are summarized, then hard-trimmed if still long):

```bash
uv run python chatbot.py
```

Same chatbot as a LangGraph (`thread_id` = session id; sqlite, not JSON):

```bash
uv run python graph_chatbot.py
```

Paste a paragraph and print structured fields (`summary`, `paragraph_count`, `author`, `published_at`):

```bash
uv run python extract_paragraph.py
```

### Dev server (LangGraph Studio)

Requires Ollama running. Installs the CLI into the project env:

```bash
uv sync --group dev
uv run langgraph dev
```

- API default: `http://localhost:2024` (Studio usually opens in the browser)
- Graph id: **`tutor`**
- Checkpoints: LangGraph API built-in (separate from CLI `sessions/langgraph.sqlite`)
- Model: set `OLLAMA_MODEL` in `.env` (loaded via `langgraph.json` and `settings.py`)
- Windows note: `colorama` is in the `dev` group so Studio logging works

Optional flags: `uv run langgraph dev --no-browser`, `--port 2025`.

In Studio: new thread (= `thread_id`), say your name, then ask what your name is; try `What time is it?` for `get_time`. Memory debug lines still print in the **server terminal**.

Type `q`, `quit`, or `exit` to stop the CLI scripts. For the chatbot: at start, enter a session id (blank = `default`). Tell it your name, quit, run it again with the **same** id, then ask `What is my name?` Also try `What time is it?` — if the model calls `get_time` you will see `(tool: get_time -> ...)`. Facts, topic notes, and recent turns (including tool messages) live in `sessions/<id>.json` (gitignored). Known facts are also pinned in Python so they survive summarizer glitches. Replies print in one go (`invoke`) so tool calls are not lost. The memory line shows `~tokens/2048`; hard trim cuts by **tokens**, not by message count, and `start_on='human'` drops a leftover tool result. `(memory: soft-summarized N older messages; ...)` means soft compression just ran.

For `extract_paragraph.py`, each paste is a new request. Replies print in one go (not streamed). If the model omits author or date, those fields show `(none)`. Chatbot facts still use regex until a later slice.

## Later concepts (not implemented yet)

Keep these for the next learning slices. One concept at a time.

1. **Leave for another repo** — RAG / vector stores (that compression is about documents, not chat), Agent Chat UI / FastAPI as a product front end. Local **`langgraph dev`** + Studio are in this repo; `create_agent` is still unused.

Chatbot facts still use regex; applying Pydantic there is a later optional tweak. `chatbot.py` JSON is not a LangGraph checkpointer; CLI and Studio each use their own sqlite file.

## Commits (Conventional Commits)

Commit messages must follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(optional-scope): <description>
```

Examples: `feat: add get_time tool`, `docs: update study notes`, `chore: ignore .langgraph_api`.

Allowed types: `feat` `fix` `docs` `style` `refactor` `perf` `test` `build` `ci` `chore` `revert` `merge`.

### Local hook

```bash
uv sync --group dev
uv run pre-commit install --hook-type commit-msg
```

After that, non-conforming `git commit` messages are rejected. New clones need to run the install again. Prefer fixing the message over `git commit --no-verify`.

### CI

[`.github/workflows/conventional-commits.yml`](.github/workflows/conventional-commits.yml) runs on:

- **Pull requests** targeting `main` or `dev`
- **Pushes** to `main` or `dev` (including after a PR is merged)

Allowed types match the local hook. Default GitHub “Merge pull request …” merge-commit subjects are allowed on push so a normal UI merge does not fail CI.

## Optional tweaks

- Change the system prompt
- Set `OLLAMA_MODEL` in `.env` to another model from `ollama list`
