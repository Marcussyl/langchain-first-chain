#!/usr/bin/env python3
"""Multi-turn chatbot: message history + soft summary + hard trim (Ollama)."""

import json
import re
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
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
    'Keep answers concise (a few short sentences) unless the user asks for more detail. '
    'You may call get_time when the user asks the current time; otherwise answer without it.'
)

# Soft compression still folds extra *turns* so the list does not grow forever.
# KEEP is a target; we walk left until the kept tail starts on HumanMessage
# so a ToolMessage is never the first recent turn.
MAX_MESSAGES_BEFORE_SUMMARY = 8
KEEP_RECENT_MESSAGES = 7

# Hard trim uses the real unit: tokens (approximate). Smaller than llama3.2's
# full window on purpose so the CLI budget is easy to see.
MAX_CONTEXT_TOKENS = 2048
SUMMARY_TRIGGER_TOKENS = 1400

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

# 2) Local chat model via Ollama (must be running; model must be pulled)
model = ChatOllama(model='llama3.2', temperature=0)


@tool
def get_time() -> str:
    """Return the current local date and time. Use when the user asks what time it is."""
    return datetime.now().astimezone().isoformat(timespec='seconds')


# Reply path keeps AIMessage.tool_calls (StrOutputParser would drop them).
tool_model = model.bind_tools([get_time])
prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder('messages'),
    ]
)
reply_chain = prompt | tool_model

# 3) Summarizer still wants a plain string. No tools.
parser = StrOutputParser()
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
DEFAULT_SESSION_ID = 'default'
SESSIONS_DIR = Path('sessions')
# Block path traversal; file name is sessions/<id>.json.
SESSION_ID_PATTERN = re.compile(r'^[A-Za-z0-9._-]+$')


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
            lines.append(f'AI: {message_text(message)}')
        elif isinstance(message, ToolMessage):
            lines.append(f'Tool: {message.content}')
    return '\n'.join(lines) or '(none)'


def message_text(message: BaseMessage) -> str:
    """Flatten message content to a string for printing and JSON."""
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def read_user_text() -> str | None:
    """Return the next user line, or None when the user wants to stop."""
    text = input('You: ').strip()
    if text.lower() in QUIT_COMMANDS:
        return None
    return text


def count_tokens(messages: list[BaseMessage]) -> int:
    """Approximate token count (not the model's exact tokenizer)."""
    return count_tokens_approximately(messages)


def recent_starting_on_human(
    messages: list[BaseMessage],
    keep: int,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Split after system so the kept tail starts on HumanMessage, not ToolMessage."""
    if len(messages) <= 1:
        return [], list(messages)
    start = max(1, len(messages) - keep)
    while start > 1 and not isinstance(messages[start], HumanMessage):
        start -= 1
    return messages[1:start], messages[start:]


def apply_soft_compression(
    messages: list[BaseMessage],
    running_summary: str | None,
    facts: dict[str, str],
) -> tuple[list[BaseMessage], str | None, int]:
    """Replace old turns with topic notes. Keep the previous notes if the LLM refuses."""
    over_messages = len(messages) > MAX_MESSAGES_BEFORE_SUMMARY
    over_tokens = count_tokens(messages) > SUMMARY_TRIGGER_TOKENS
    if not over_messages and not over_tokens:
        return messages, running_summary, 0
    if len(messages) <= 1 + KEEP_RECENT_MESSAGES:
        return messages, running_summary, 0

    older, recent = recent_starting_on_human(messages, KEEP_RECENT_MESSAGES)
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
    """Drop old messages until the approximate token budget fits."""
    return trim_messages(
        messages,
        max_tokens=MAX_CONTEXT_TOKENS,
        token_counter='approximate',
        strategy='last',
        include_system=True,
        start_on='human',
        allow_partial=False,
    )


def tool_call_parts(call) -> tuple[str, str, dict]:
    """Read name, id, args from a dict or ToolCall object."""
    if isinstance(call, dict):
        return str(call.get('name') or ''), str(call.get('id') or ''), call.get('args') or {}
    return (
        str(getattr(call, 'name', '') or ''),
        str(getattr(call, 'id', '') or ''),
        getattr(call, 'args', None) or {},
    )


def print_bot(message: AIMessage) -> None:
    """Print the visible reply. Tool-only AIMessages may have empty content."""
    print(f'Bot: {message_text(message)}')


def run_turn(to_send: list[BaseMessage]) -> list[BaseMessage]:
    """One model call, at most one tool round, then a final text message."""
    first = reply_chain.invoke({'messages': to_send})
    extra: list[BaseMessage] = [first]
    tool_calls = getattr(first, 'tool_calls', None) or []
    if not tool_calls:
        print_bot(first)
        return extra

    pending = list(to_send) + extra
    for call in tool_calls:
        name, call_id, args = tool_call_parts(call)
        if name == get_time.name:
            result = get_time.invoke(args)
        else:
            result = f'Unknown tool: {name}'
        print(f'(tool: {name} -> {result})')
        tool_msg = ToolMessage(
            content=str(result),
            tool_call_id=call_id,
            name=name,
        )
        extra.append(tool_msg)
        pending.append(tool_msg)

    # Cap at one tool round: do not loop even if the follow-up still has tool_calls.
    final = reply_chain.invoke({'messages': pending})
    extra.append(final)
    print_bot(final)
    return extra


def session_file(session_id: str) -> Path:
    """JSON path for one session id (never a user-supplied path)."""
    return SESSIONS_DIR / f'{session_id}.json'


def read_session_id() -> str:
    """Ask which disk session to use. Blank means default."""
    while True:
        raw = input('Session id (blank = default): ').strip()
        if not raw:
            return DEFAULT_SESSION_ID
        if SESSION_ID_PATTERN.fullmatch(raw):
            return raw
        print('Use only letters, digits, ".", "_", or "-".')


def messages_to_json(messages: list[BaseMessage]) -> list[dict]:
    """Serialize human/ai/tool turns. SystemMessage is rebuilt on load."""
    rows: list[dict] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            rows.append({'role': 'human', 'content': message_text(message)})
        elif isinstance(message, AIMessage):
            row: dict = {'role': 'ai', 'content': message_text(message)}
            tool_calls = getattr(message, 'tool_calls', None) or []
            if tool_calls:
                row['tool_calls'] = [
                    {
                        'name': name,
                        'args': args,
                        'id': call_id,
                        'type': 'tool_call',
                    }
                    for name, call_id, args in (tool_call_parts(call) for call in tool_calls)
                ]
            rows.append(row)
        elif isinstance(message, ToolMessage):
            rows.append(
                {
                    'role': 'tool',
                    'content': message_text(message),
                    'tool_call_id': message.tool_call_id,
                    'name': message.name or '',
                }
            )
    return rows


def messages_from_json(rows: list) -> list[BaseMessage]:
    """Rebuild human/ai/tool turns. Unknown roles fail the whole load."""
    history: list[BaseMessage] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('each message must be an object')
        role = row.get('role')
        content = row.get('content')
        if not isinstance(content, str):
            raise ValueError('message content must be a string')
        if role == 'human':
            history.append(HumanMessage(content=content))
        elif role == 'ai':
            tool_calls = row.get('tool_calls')
            if tool_calls:
                history.append(AIMessage(content=content, tool_calls=tool_calls))
            else:
                history.append(AIMessage(content=content))
        elif role == 'tool':
            tool_call_id = row.get('tool_call_id')
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ValueError('tool_call_id required')
            name = row.get('name') or get_time.name
            history.append(
                ToolMessage(content=content, tool_call_id=tool_call_id, name=name)
            )
        else:
            raise ValueError(f'unknown role: {role}')
    return history


def pinned_facts(raw: dict) -> dict[str, str]:
    """Keep only known fact keys as strings."""
    return {
        str(key): str(value)
        for key, value in raw.items()
        if key in FACT_LABELS and value is not None
    }


def load_session(
    path: Path,
) -> tuple[dict[str, str], str | None, list[BaseMessage], str]:
    """Load facts, topic notes, and human/ai/tool messages. status is new/loaded/failed."""
    if not path.exists():
        return {}, None, [], 'new'
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('session file must be an object')
        facts_raw = data.get('facts') or {}
        if not isinstance(facts_raw, dict):
            raise ValueError('facts must be an object')
        notes = data.get('topic_notes')
        if notes is not None and not isinstance(notes, str):
            raise ValueError('topic_notes must be a string or null')
        rows = data.get('messages') or []
        if not isinstance(rows, list):
            raise ValueError('messages must be a list')
        history = messages_from_json(rows)
        return pinned_facts(facts_raw), notes, history, 'loaded'
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}, None, [], 'failed'


def save_session(
    path: Path,
    session_id: str,
    facts: dict[str, str],
    notes: str | None,
    messages: list[BaseMessage],
) -> None:
    """Write JSON atomically so a crash keeps the last complete turn."""
    SESSIONS_DIR.mkdir(exist_ok=True)
    payload = {
        'session_id': session_id,
        'facts': pinned_facts(facts),
        'topic_notes': notes,
        'messages': messages_to_json(messages),
    }
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


if __name__ == '__main__':
    SESSIONS_DIR.mkdir(exist_ok=True)
    session_id = read_session_id()
    path = session_file(session_id)
    facts, running_summary, history, status = load_session(path)
    messages: list[BaseMessage] = [make_system(facts, running_summary), *history]

    print('Chatbot with short-term memory plus JSON on disk. Empty input is ignored; type q to quit.')
    print('Try: tell it your name, or ask "What time is it?" (get_time tool).')
    print('Same session id after quit still remembers facts and recent turns.')
    print(f'Hard-trim budget: ~{MAX_CONTEXT_TOKENS} tokens (approximate).')
    if status == 'loaded':
        print(f'(session: {session_id}  loaded {len(history)} messages)')
    elif status == 'failed':
        print(f'(session: {session_id} load failed, starting fresh)')
    else:
        print(f'(session: {session_id}  new)')
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
        tokens_before = count_tokens(messages)
        tokens_after = count_tokens(to_send)
        print(
            f'(memory: {extra}sending {len(to_send)} of {len(messages)} messages, '
            f'~{tokens_after}/{MAX_CONTEXT_TOKENS} tokens'
            f'{f", was ~{tokens_before}" if tokens_after != tokens_before else ""})'
        )
        print(f'(known facts: {format_facts(facts)})')
        if running_summary:
            print(f'(topic notes: {running_summary})')

        new_messages = run_turn(to_send)
        print()

        messages.extend(new_messages)
        save_session(path, session_id, facts, running_summary, messages)
