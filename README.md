# langchain-first-chain

Hands-on starter for **LangChain LCEL**: Prompt → LLM → output parser (or structured Pydantic fields), running fully local with **Ollama** (no API key).

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

**`graph_chatbot.py` (LangGraph twin):**

- Same harvest / soft summary / hard trim / `get_time` as `chatbot.py` (imported, not copied)
- `StateGraph` nodes `prepare` then `generate` instead of a `while True` body
- SQLite checkpointer (`sessions/langgraph.sqlite`); `thread_id` is the session id
- Does **not** use `create_agent`; does **not** share `sessions/*.json` with `chatbot.py`

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

## Setup

```bash
# Pull the model (once)
ollama pull llama3.2

# Project env
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
source .venv/Scripts/activate      # Windows Git Bash
.venv\Scripts\activate.bat         # Windows CMD
pip install -r requirements.txt
```

If Ollama is not already running:

```bash
ollama serve
```

## Run

One topic at a time (no memory between questions):

```bash
python first_chain.py
```

Chatbot that remembers this session (old turns are summarized, then hard-trimmed if still long):

```bash
python chatbot.py
```

Same chatbot as a LangGraph (`thread_id` = session id; sqlite, not JSON):

```bash
python graph_chatbot.py
```

Paste a paragraph and print structured fields (`summary`, `paragraph_count`, `author`, `published_at`):

```bash
python extract_paragraph.py
```

Type `q`, `quit`, or `exit` to stop. For the chatbot: at start, enter a session id (blank = `default`). Tell it your name, quit, run it again with the **same** id, then ask `What is my name?` Also try `What time is it?` — if the model calls `get_time` you will see `(tool: get_time -> ...)`. Facts, topic notes, and recent turns (including tool messages) live in `sessions/<id>.json` (gitignored). Known facts are also pinned in Python so they survive summarizer glitches. Replies print in one go (`invoke`) so tool calls are not lost. The memory line shows `~tokens/2048`; hard trim cuts by **tokens**, not by message count, and `start_on='human'` drops a leftover tool result. `(memory: soft-summarized N older messages; ...)` means soft compression just ran.

For `extract_paragraph.py`, each paste is a new request. Replies print in one go (not streamed). If the model omits author or date, those fields show `(none)`. Chatbot facts still use regex until a later slice.

## Later concepts (not implemented yet)

Keep these for the next learning slices. One concept at a time.

1. **Leave for another repo** — RAG / vector stores (that compression is about documents, not chat), LangGraph Studio / Agent Chat UI / FastAPI. `graph_chatbot.py` is only the CLI twin of `chatbot.py`, not a deployed Agent Server. `create_agent` is still unused.

Chatbot facts still use regex; applying Pydantic there is a later optional tweak. `chatbot.py` JSON is not a LangGraph checkpointer; `graph_chatbot.py` uses sqlite instead.

## Optional tweaks

- Change the system prompt
- Swap `llama3.2` for another model you have in `ollama list`
