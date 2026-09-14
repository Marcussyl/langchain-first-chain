#!/usr/bin/env python3
"""Multi-turn chatbot: message history + trim_messages (Ollama, no API key)."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

SYSTEM_PROMPT = (
    'You are a patient tutor in a terminal chatbot. '
    'Answer clearly in plain language. '
    'Use the conversation history: remember names and topics the user already mentioned. '
    'Do not confuse similar-sounding names with unrelated fields '
    '(for example, LangChain is an LLM framework, not blockchain). '
    'Keep answers concise (a few short sentences) unless the user asks for more detail.'
)

# With token_counter=len, max_tokens is actually a max message count.
# 8 = system + a few recent turns, so trim is easy to notice while chatting.
MAX_MESSAGES = 8

# 1) The message list is the prompt. MessagesPlaceholder injects it as-is.
prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder('messages'),
    ]
)

# 2) Local chat model via Ollama (must be running; model must be pulled)
model = ChatOllama(model='llama3.2', temperature=0)

# 3) Turn the model message into a plain string
parser = StrOutputParser()

# LCEL: the list of messages goes straight into the model
chain = prompt | model | parser

QUIT_COMMANDS = {'q', 'quit', 'exit'}


def read_user_text() -> str | None:
    """Return the next user line, or None when the user wants to stop."""
    text = input('You: ').strip()
    if text.lower() in QUIT_COMMANDS:
        return None
    return text


def compact_history(messages: list) -> list:
    """Keep recent messages so the model context stays small and valid."""
    return trim_messages(
        messages,
        max_tokens=MAX_MESSAGES,
        token_counter=len,
        strategy='last',
        include_system=True,
        start_on='human',
        allow_partial=False,
    )


if __name__ == '__main__':
    # Full history in this process. Only a trimmed copy is sent to the model.
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    print('Chatbot with short-term memory. Empty input is ignored; type q to quit.')
    print('Try: tell it your name, then ask "What is my name?"')
    print()

    while True:
        user_text = read_user_text()
        if user_text is None:
            print('Bye.')
            break
        if not user_text:
            continue

        messages.append(HumanMessage(content=user_text))
        to_send = compact_history(messages)
        print(f'(memory: sending {len(to_send)} of {len(messages)} messages)')

        reply = chain.invoke({'messages': to_send})
        print(f'Bot: {reply}')
        print()

        messages.append(AIMessage(content=reply))
