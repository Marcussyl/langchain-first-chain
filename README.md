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
- Context **compression** with `trim_messages` (`strategy='last'`, `include_system=True`, `start_on='human'`)

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

Chatbot that remembers this session (until the history is trimmed):

```bash
python chatbot.py
```

Type `q`, `quit`, or `exit` to stop. Tell it your name, then ask `What is my name?` to see memory working. When the printed `sending N of M messages` has `N < M`, trim has dropped older turns.

## Optional next steps

- Change the system prompt
- Try streaming: `for chunk in chain.stream(...): print(chunk, end="")`
- Swap `llama3.2` for another model you have in `ollama list`
- Summarize old turns instead of dropping them (soft compression)
