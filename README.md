# langchain-first-chain

Hands-on starter for **LangChain LCEL**: Prompt → LLM → output parser, running fully local with **Ollama** (no API key).

## What you learn

**`first_chain.py` (one-shot explain):**

- `ChatPromptTemplate.from_messages` with a system role and a `{topic}` human message
- Passing a dict into LCEL: `chain.invoke({"topic": ...})`
- `ChatOllama` to call a local model
- `StrOutputParser` to get a plain string
- LCEL piping with `|`: `prompt | model | parser`

**`chatbot.py` (multi-turn):**

- `HumanMessage` / `AIMessage` / `SystemMessage` as the real chat history
- `MessagesPlaceholder` to feed that list into the chain
- Short-term **memory**: later turns see earlier messages
- **Soft** compression: old turns become a running summary on the `SystemMessage`
- **Hard** compression: `trim_messages` with `token_counter='approximate'` (`strategy='last'`, `include_system=True`, `start_on='human'`)
- **Streaming**: `chain.stream` prints tokens as they arrive (`invoke` waits for the full answer)

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

Type `q`, `quit`, or `exit` to stop. Tell it your name, chat for a while, then ask `What is my name?` Known facts are pinned in Python so they survive summarizer glitches. Replies **stream** token by token. The memory line shows `~tokens/2048`; hard trim cuts by **tokens**, not by message count. `(memory: soft-summarized N older messages; ...)` means soft compression just ran.

## Later concepts (not implemented yet)

Keep these for the next learning slices. One concept at a time; stay on this terminal chatbot.

1. **Structured output (Pydantic)** — parse the model into fields such as `answer`, `facts_mentioned`, instead of a free-form string. Safer way to update the facts dict than regex-only harvest. Also makes the summarizer less likely to return a refusal as “memory”.
2. **Persist sessions** — write `facts` + topic notes + recent messages to JSON keyed by `session_id`. Short-term memory is one request’s context; this is long-term memory on disk. LangGraph checkpointers can wait.
3. **One tool / tiny agent** — e.g. `get_time` or `ollama list`. Learn tool calls and `ToolMessage` (why trim uses `start_on='human'`). Stop at one tool.
4. **Leave for another repo** — RAG / vector stores (that compression is about documents, not chat), LangGraph multi-node graphs, FastAPI / a web UI.

Suggested order: Pydantic → JSON session → one tool.

## Optional tweaks

- Change the system prompt
- Swap `llama3.2` for another model you have in `ollama list`
