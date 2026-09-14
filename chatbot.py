#!/usr/bin/env python3
"""Multi-turn chatbot: message history + soft summary + hard trim (Ollama)."""

import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

SYSTEM_PROMPT = (
    'You are a patient tutor in a continuing terminal chat (same session). '
    'You DO have short-term memory: the messages in this request, plus any '
    '"SESSION MEMORY" block below, are facts you already know. '
    'Use them. Never say you cannot remember this conversation, that each '
    'message is a new conversation, or that you have no way to recall facts. '
    'If the user just told you a fact, acknowledge it and keep using it. '
    'Answer clearly in plain language. '
    'Do not confuse similar-sounding names with unrelated fields '
    '(for example, LangChain is an LLM framework, not blockchain). '
    'Keep answers concise (a few short sentences) unless the user asks for more detail.'
)

# With token_counter=len, max_tokens is actually a max message count.
MAX_MESSAGES = 8
# Fill the budget: 1 SystemMessage + this many recent turns = MAX_MESSAGES.
# Odd on purpose: after we append the latest HumanMessage, the tail is
# H, A, H, ... H so start_on='human' stays valid.
KEEP_RECENT_MESSAGES = MAX_MESSAGES - 1

FACT_PATTERNS = (
    (r'(?i)\bmy name is\s+([A-Za-z]+)', 'user_name'),
    (r'(?i)\bmy name if\s+([A-Za-z]+)', 'user_name'),
    (r'(?i)\byou are\s+([A-Za-z]+)', 'assistant_name'),
    (r'(?i)\byour name is\s+([A-Za-z]+)', 'assistant_name'),
    (r'(?i)\bmy fav(?:orite)? color is\s+([A-Za-z]+)', 'favorite_color'),
    (r'(?i)\bmy fav(?:orite)? one is\s+([A-Za-z]+)', 'favorite_item'),
)

# Llama often replaces a good summary with a safety refusal or prompt echo.
UNUSABLE_SUMMARY_MARKERS = (
    "i can't fulfill",
    'i cannot fulfill',
    "i can't create content",
    'i cannot create content',
    'sexualizes a child',
    'new messages to merge',
    'existing summary:',
    'updated summary:',
)

FACT_LABELS = {
    'user_name': "The user's name is",
    'assistant_name': "The assistant's name is",
    'favorite_color': "The user's favorite color is",
    'favorite_item': "The user's favorite item/brand is",
}

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

# Separate chain: topical notes only. Personal facts are stored in Python.
summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'You write short topic notes for a tutoring chat. '
            'Output ONLY a few sentences. No headings. Never refuse. '
            'Do not mention children, safety policy, or that you lack memory. '
            'Do not repeat personal names; those are stored elsewhere.',
        ),
        (
            'human',
            'Previous topic notes:\n{existing_summary}\n\n'
            'New turns:\n{new_lines}\n\n'
            'Updated topic notes:',
        ),
    ]
)
summary_chain = summary_prompt | model | parser

QUIT_COMMANDS = {'q', 'quit', 'exit'}


def harvest_facts(user_text: str, facts: dict[str, str]) -> None:
    """Pin personal facts in Python so an LLM summary cannot overwrite them."""
    for pattern, key in FACT_PATTERNS:
        match = re.search(pattern, user_text)
        if match:
            facts[key] = match.group(1)


def format_facts(facts: dict[str, str]) -> str:
    lines = [
        f'- {FACT_LABELS[key]} {value}.'
        for key, value in facts.items()
        if key in FACT_LABELS
    ]
    return '\n'.join(lines) if lines else '(none yet)'


def is_unusable_summary(text: str) -> bool:
    """Reject refusals and prompt-echo so they cannot replace a good summary."""
    lowered = text.lower().strip()
    if not lowered:
        return True
    return any(marker in lowered for marker in UNUSABLE_SUMMARY_MARKERS)


def make_system(facts: dict[str, str], topic_summary: str | None) -> SystemMessage:
    """Persona + Python-owned facts + optional topical summary."""
    parts = [
        SYSTEM_PROMPT,
        '',
        'SESSION MEMORY (treat as true; do not ignore these facts):',
        'Known facts:',
        format_facts(facts),
    ]
    if topic_summary:
        parts.extend(['', 'Topic notes:', topic_summary])
    return SystemMessage(content='\n'.join(parts))


def format_messages_for_summary(messages: list[BaseMessage]) -> str:
    """Turn old turns into plain text for the summarizer."""
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f'Human: {message.content}')
        elif isinstance(message, AIMessage):
            lines.append(f'AI: {message.content}')
    return '\n'.join(lines) or '(none)'


def read_user_text() -> str | None:
    """Return the next user line, or None when the user wants to stop."""
    text = input('You: ').strip()
    if text.lower() in QUIT_COMMANDS:
        return None
    return text


def apply_soft_compression(
    messages: list[BaseMessage],
    running_summary: str | None,
    facts: dict[str, str],
) -> tuple[list[BaseMessage], str | None, int]:
    """Replace old turns with topic notes. Keep the previous notes if the LLM refuses."""
    if len(messages) <= MAX_MESSAGES:
        return messages, running_summary, 0
    if len(messages) <= 1 + KEEP_RECENT_MESSAGES:
        return messages, running_summary, 0

    older = messages[1:-KEEP_RECENT_MESSAGES]
    recent = messages[-KEEP_RECENT_MESSAGES:]
    if not older:
        return messages, running_summary, 0

    candidate = summary_chain.invoke(
        {
            'existing_summary': running_summary or '(none)',
            'new_lines': format_messages_for_summary(older),
        }
    )
    if is_unusable_summary(candidate):
        new_summary = running_summary
    else:
        new_summary = candidate.strip()

    compacted = [make_system(facts, new_summary), *recent]
    return compacted, new_summary, len(older)


def hard_trim(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drop leftover messages if the list is still over the cap."""
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
    running_summary: str | None = None
    facts: dict[str, str] = {}
    messages: list[BaseMessage] = [make_system(facts, None)]

    print('Chatbot with short-term memory. Empty input is ignored; type q to quit.')
    print('Try: tell it your name, chat for a while, then ask "What is my name?"')
    print()

    while True:
        user_text = read_user_text()
        if user_text is None:
            print('Bye.')
            break
        if not user_text:
            continue

        harvest_facts(user_text, facts)
        messages.append(HumanMessage(content=user_text))
        messages[0] = make_system(facts, running_summary)
        messages, running_summary, n_summarized = apply_soft_compression(
            messages,
            running_summary,
            facts,
        )
        to_send = hard_trim(messages)

        extra = f'soft-summarized {n_summarized} older messages; ' if n_summarized else ''
        print(f'(memory: {extra}sending {len(to_send)} of {len(messages)} messages)')
        print(f'(known facts: {format_facts(facts)})')
        if running_summary:
            print(f'(topic notes: {running_summary})')

        reply = chain.invoke({'messages': to_send})
        print(f'Bot: {reply}')
        print()

        messages.append(AIMessage(content=reply))
