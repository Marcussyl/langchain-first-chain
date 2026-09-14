# Study notes — langchain-first-chain

Notes for what this repo actually uses so far. Not a full LangChain textbook.

Two scripts:

- `first_chain.py` — one prompt, one answer, no memory between questions.
- `chatbot.py` — multi-turn messages + `trim_messages`.

## Big picture

The app is one **LCEL chain**:

```text
user topic  →  prompt  →  model  →  parser  →  string on screen
```

- **Prompt**: turn a dict like `{"topic": "LCEL"}` into chat messages.
- **Model**: send those messages to a local LLM (Ollama).
- **Parser**: turn the model reply into a plain `str`.

LCEL is just composing these steps with `|`:

```python
chain = prompt | model | parser
```

Each piece is a **Runnable**. `invoke` on the chain runs them in order. The chain input is a **dict**, not a raw string, because the prompt template has named variables.

## Prompt: `ChatPromptTemplate`

Two styles:

| API | What it is |
|-----|------------|
| `from_template("...{topic}...")` | One human message. Fine for a first demo. |
| `from_messages([...])` | List of roles. This project uses this. |

Roles we use:

- **system**: standing instructions (be a tutor; do not mix up names). The model is more likely to follow this than a one-line human prompt.
- **human**: the actual user request. `{topic}` is filled at invoke time.

`"{topic}"` in the template is **not** an f-string. LangChain substitutes it when you call `chain.invoke({"topic": topic})`. If the key is missing, invoke fails.

## Model: `ChatOllama` vs Ollama

They are not the same program.

| Piece | What it is |
|-------|------------|
| **Ollama** | Native app / background service. Speaks HTTP (default `localhost:11434`). Holds models like `llama3.2`. Install globally; `ollama serve`, `ollama pull`. |
| **`langchain-ollama.ChatOllama`** | Python client inside the venv. Builds chat requests and talks to that service. |

If Ollama is down, the Python chain fails even if `pip install` succeeded.

`temperature=0` means more deterministic answers. It does **not** mean "always factually correct".

## Parser: `StrOutputParser`

Chat models return a **message object** (role + content + metadata). The parser pulls out `.content` as a `str`, so `print(result)` is readable.

Without it you would print something like an `AIMessage(...)`.

## Running the chain

```python
result = chain.invoke({'topic': topic})
```

- **`invoke`**: wait for the full answer, then return (`first_chain.py`, and the summarizer in `chatbot.py`).
- **`stream`**: yield chunks as they arrive (`chatbot.py` replies). Same chain, different consumption. The full string is still joined so we can append an `AIMessage`.
- **`batch`**: several inputs at once (not used yet).

In `first_chain.py`, the `while True` + `input()` loop is ordinary Python. It is **not** LangChain memory. Each `invoke` is a **new, independent** request.

`chatbot.py` is the version that **does** pass history (see below).

`if __name__ == '__main__'` keeps chain construction importable without starting the CLI.

## Hallucination (what we already saw)

The first run explained LangChain as a **blockchain** framework. The pipeline was correct; the **model guessed** from the word "Chain".

Takeaways:

- A working chain ≠ a true answer.
- Small local models hallucinate more, especially with short prompts.
- A **system** message that names the confusion (LLM framework, not blockchain) is a cheap mitigation, not a guarantee.

## Python env (setup, not LCEL)

For this project:

- **In the venv**: Python + `langchain` / `langchain-core` / `langchain-ollama` (PyPI).
- **Global**: Ollama app + pulled models; Git; editor.

`pip` + `venv` is enough because everything we import is on PyPI.

Windows **Git Bash** uses forward slashes:

```bash
source .venv/Scripts/activate
```

Backslash is an escape in bash (`.venv\Scripts\activate` becomes `.venvScriptsactivate`). Unix layouts use `.venv/bin/activate`; Windows venvs use `.venv/Scripts/`.

## Packages (what each one is for)

- `langchain-core`: prompts, parsers, LCEL primitives.
- `langchain-ollama`: `ChatOllama`.
- `langchain`: umbrella / extra integrations; this tiny script mostly needs the two above.

## Chatbot memory (`chatbot.py`)

Short-term memory is just a **list of messages** kept in the process:

```text
[system, human, ai, human, ai, ...]
```

Each turn we append a `HumanMessage`, `invoke` the chain, then append an `AIMessage`. The next call includes those earlier messages, so "My name is Ada" then "What is my name?" can work.

```mermaid
classDiagram
    class BaseMessage {
        +content str
    }
    BaseMessage <|-- SystemMessage : persona
    BaseMessage <|-- HumanMessage : user turn
    BaseMessage <|-- AIMessage : model turn

    class SessionMemory {
        +messages list
        +running_summary str
        +append()
    }
    SessionMemory "1" *-- "*" BaseMessage : system plus recent turns

    class SoftSummary {
        +fold older turns into running_summary
        +write summary onto SystemMessage
    }
    class HardTrim {
        +MAX_MESSAGES 8
        +strategy last
        +include_system True
        +start_on human
    }
    class LcelChain {
        +invoke()
    }
    SessionMemory ..> SoftSummary : when over MAX_MESSAGES
    SoftSummary ..> SessionMemory : replace old turns
    SessionMemory ..> HardTrim : backup
    HardTrim ..> LcelChain : invoke
```

One turn (summarize old turns if needed, then hard-trim as backup):

```mermaid
sequenceDiagram
    actor User
    participant CLI as chatbot.py
    participant Memory as messages list
    participant Soft as summary chain
    participant Hard as trim_messages
    participant Chain as LCEL chain
    participant Ollama as ChatOllama

    Note over Memory: starts with SystemMessage
    User->>CLI: user input
    CLI->>Memory: append HumanMessage
    alt list longer than MAX_MESSAGES
        CLI->>Soft: summarize older turns
        Soft-->>CLI: running summary
        CLI->>Memory: system plus summary plus recent turns
    end
    CLI->>Hard: trim_messages
    Hard-->>CLI: to_send
    CLI->>Chain: invoke to_send
    Chain->>Ollama: chat messages
    Ollama-->>Chain: AIMessage
    Chain-->>CLI: reply string
    CLI->>Memory: append AIMessage
    CLI-->>User: bot reply
```

`MessagesPlaceholder('messages')` means: do not template a single `{topic}`; inject this list as the prompt. The chain is still LCEL: `prompt | model | parser`. Input is `{"messages": ...}`.

This list dies when the process exits. That is still **short-term / session** memory, not a database.

## Compression: soft summary, then hard `trim_messages`

Unbounded history will blow the context window. Each `invoke` only sees what we send **this turn**.

Pipeline in `chatbot.py`:

```text
full messages
    -> if too many messages OR too many tokens: summarize older turns (soft)
       (facts stay in a Python dict on the SystemMessage; recent turns stay verbatim)
    -> trim_messages by approximate tokens (hard, backup)
    -> model
```

**Soft** (first): old `Human`/`AI` turns are folded into **topic notes** by a second LCEL chain. **Personal facts** (name, favorite color, …) are harvested with regex into a Python dict and always written onto the `SystemMessage`. The summarizer is not allowed to overwrite those facts. If the summarizer returns a safety refusal or echoes the prompt, we **keep the previous topic notes**.

That failure happened in testing: the first summary still had `Marcus`; the next summarizer call returned `I can't create content that sexualizes a child` and replaced the whole memory. Soft compression was running; the LLM summary was just a bad store for names.

Recent turns (`KEEP_RECENT_MESSAGES = 7`) stay as original messages, including the current human question. Soft compression also starts if approximate tokens go above `SUMMARY_TRIGGER_TOKENS` (1400), even when there are still few messages.

**Hard** (second): `trim_messages` caps the **token** budget (`MAX_CONTEXT_TOKENS = 2048`), not the message count. A long `SystemMessage` is still 1 message but can use hundreds of tokens. `token_counter='approximate'` is a fast estimate (not the Ollama tokenizer). `include_system=True` keeps persona + known facts.

```python
trim_messages(
    messages,
    max_tokens=MAX_CONTEXT_TOKENS,
    token_counter='approximate',
    strategy='last',
    include_system=True,
    start_on='human',
)
```

| Arg | Meaning |
|-----|---------|
| `strategy='last'` | Keep the **recent** tail; drop old turns first. `'first'` would keep the beginning and forget what you just said. |
| `include_system=True` | Always keep the `SystemMessage` at index 0 (persona **and** the running summary). |
| `start_on='human'` | After the cut, drop a leftover prefix until a `HumanMessage` (do not start on a dangling `AIMessage`). Does not strip the kept system message. |

`token_counter=len` (old) counted each message as 1. That was easier to demo, but it is not how the model window works.

When the CLI prints `~890/2048 tokens`, that is the approximate size of **this request**. If it also prints `was ~2100`, hard trim dropped tokens. `(known facts: ...)` is the Python-owned pin (names survive even if topic notes fail). `(topic notes: ...)` is the LLM summary.

Hard trim alone used to print `sending 8 of 32` and forget the name. Soft-then-hard is meant to avoid that — but only if the summarizer does not wipe the memory. That is why facts are pinned outside the LLM.

## Not in the code yet (next concepts)

See README “Later concepts”: Pydantic structured output, persist sessions, one tool. RAG / LangGraph / a web UI stay out of this repo for now.
