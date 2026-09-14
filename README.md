# langchain-first-chain

Hands-on starter for **LangChain LCEL**: Prompt → LLM → output parser, running fully local with **Ollama** (no API key).

## What you learn

- `ChatPromptTemplate.from_messages` with a system role and a `{topic}` human message
- Passing a dict into LCEL: `chain.invoke({"topic": ...})`
- `ChatOllama` to call a local model
- `StrOutputParser` to get a plain string
- LCEL piping with `|`: `prompt | model | parser`

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

```bash
python first_chain.py
```

The script asks for a topic, prints a 2-sentence explanation, then asks again. Type `q`, `quit`, or `exit` to stop.

## Optional next steps

- Change the system or human prompt in `first_chain.py`
- Try streaming: `for chunk in chain.stream({"topic": topic}): print(chunk, end="")`
- Swap `llama3.2` for another model you have in `ollama list`
