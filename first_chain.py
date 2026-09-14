#!/usr/bin/env python3
"""First LangChain LCEL chain: Prompt → LLM → output parser (Ollama, no API key)."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# 1) Prompt template — system sets the tutor role; {topic} is filled in at invoke time
prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'You are a patient tutor. Explain clearly in plain language. '
            'Do not confuse similar-sounding names with unrelated fields '
            '(for example, LangChain is an LLM framework, not blockchain).',
        ),
        (
            'human',
            'Explain "{topic}" in 2 short sentences for a beginner.',
        ),
    ]
)

# 2) Local chat model via Ollama (must be running; model must be pulled)
model = ChatOllama(model='llama3.2', temperature=0)

# 3) Turn the model message into a plain string
parser = StrOutputParser()

# LCEL: pipe each step into the next with |
chain = prompt | model | parser

QUIT_COMMANDS = {'q', 'quit', 'exit'}


def read_topic() -> str | None:
    """Return a topic, or None when the user wants to stop."""
    topic = input('Topic (q to quit): ').strip()
    if topic.lower() in QUIT_COMMANDS:
        return None
    return topic


if __name__ == '__main__':
    print('Explain any topic. Empty input is ignored; type q to quit.')
    while True:
        topic = read_topic()
        if topic is None:
            print('Bye.')
            break
        if not topic:
            continue
        result = chain.invoke({'topic': topic})
        print(result)
        print()
