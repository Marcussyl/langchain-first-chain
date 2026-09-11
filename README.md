# langchain-first-chain

Hands-on starter for **LangChain LCEL**: Prompt → LLM → output parser, running fully local with **Ollama** (no API key).

## What you learn

- `ChatPromptTemplate` to shape inputs
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
source .venv/bin/activate   # Windows: .venv\Scripts\activate
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

You should see a short explanation of LangChain printed in the terminal.

## Optional next steps

- Change the prompt template or the `{topic}` value
- Try streaming: `for chunk in chain.stream({\"topic\": \"LCEL\"}): print(chunk, end=\"\")`
- Swap `llama3.2` for another model you have in `ollama list`
