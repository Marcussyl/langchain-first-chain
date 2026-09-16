#!/usr/bin/env python3
"""LangGraph twin of chatbot.py: same harvest / compress / trim / get_time, graph + checkpointer."""

import sqlite3
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from chatbot import (
    MAX_CONTEXT_TOKENS,
    SESSIONS_DIR,
    apply_soft_compression,
    count_tokens,
    format_facts,
    hard_trim,
    harvest_facts,
    make_system,
    read_session_id,
    read_user_text,
    run_turn,
)

# One sqlite file for all thread_id values (session ids). Not chatbot.py JSON.
CHECKPOINT_PATH = SESSIONS_DIR / 'langgraph.sqlite'


class ChatState(TypedDict):
    """Full turn state. Nodes replace these keys (no add_messages reducer)."""

    messages: list
    facts: dict
    topic_notes: str | None
    n_summarized: int


def empty_state() -> ChatState:
    return {
        'messages': [make_system({}, None)],
        'facts': {},
        'topic_notes': None,
        'n_summarized': 0,
    }


def prepare_node(state: ChatState) -> dict:
    """Rebuild SystemMessage, then soft-compress like chatbot.py."""
    facts = dict(state.get('facts') or {})
    notes = state.get('topic_notes')
    messages = list(state.get('messages') or [])
    if not messages:
        messages = [make_system(facts, notes)]
    else:
        messages[0] = make_system(facts, notes)
    messages, notes, n_summarized = apply_soft_compression(messages, notes, facts)
    return {
        'messages': messages,
        'facts': facts,
        'topic_notes': notes,
        'n_summarized': n_summarized,
    }


def generate_node(state: ChatState) -> dict:
    """Hard-trim a copy, run one model/tool round, append new messages."""
    messages = list(state['messages'])
    to_send = hard_trim(messages)
    n_summarized = state.get('n_summarized') or 0
    extra = f'soft-summarized {n_summarized} older messages; ' if n_summarized else ''
    tokens_before = count_tokens(messages)
    tokens_after = count_tokens(to_send)
    print(
        f'(memory: {extra}sending {len(to_send)} of {len(messages)} messages, '
        f'~{tokens_after}/{MAX_CONTEXT_TOKENS} tokens'
        f'{f", was ~{tokens_before}" if tokens_after != tokens_before else ""})'
    )
    print(f'(known facts: {format_facts(state.get("facts") or {})})')
    notes = state.get('topic_notes')
    if notes:
        print(f'(topic notes: {notes})')

    new_messages = run_turn(to_send)
    print()
    return {'messages': messages + list(new_messages)}


def build_graph(checkpointer: SqliteSaver):
    """Two nodes: prepare (memory) then generate (model + one tool round)."""
    builder = StateGraph(ChatState)
    builder.add_node('prepare', prepare_node)
    builder.add_node('generate', generate_node)
    builder.add_edge(START, 'prepare')
    builder.add_edge('prepare', 'generate')
    builder.add_edge('generate', END)
    return builder.compile(checkpointer=checkpointer)


def history_len(messages: list) -> int:
    """Count stored turns, excluding the rebuilt SystemMessage."""
    return max(0, len(messages) - 1)


def read_checkpoint(graph, config: dict) -> tuple[ChatState, str]:
    """Load thread state, or a blank tutor session."""
    snapshot = graph.get_state(config)
    values = getattr(snapshot, 'values', None) or {}
    if not values.get('messages'):
        return empty_state(), 'new'
    return values, 'loaded'


if __name__ == '__main__':
    SESSIONS_DIR.mkdir(exist_ok=True)
    session_id = read_session_id()
    config = {'configurable': {'thread_id': session_id}}

    conn = sqlite3.connect(CHECKPOINT_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    graph = build_graph(checkpointer)

    state, status = read_checkpoint(graph, config)
    print('LangGraph chatbot (same logic as chatbot.py; sqlite checkpointer).')
    print('Empty input is ignored; type q to quit.')
    print('Try: tell it your name, or ask "What time is it?" (get_time tool).')
    print('This file does not share sessions/*.json with chatbot.py.')
    print(f'Hard-trim budget: ~{MAX_CONTEXT_TOKENS} tokens (approximate).')
    if status == 'loaded':
        print(f'(thread: {session_id}  loaded {history_len(state["messages"])} messages)')
    else:
        print(f'(thread: {session_id}  new)')
    print()

    while True:
        user_text = read_user_text()
        if user_text is None:
            print('Bye.')
            break
        if not user_text:
            continue

        state, _ = read_checkpoint(graph, config)
        facts = dict(state.get('facts') or {})
        harvest_facts(user_text, facts)
        messages = list(state.get('messages') or empty_state()['messages'])
        messages.append(HumanMessage(content=user_text))
        graph.invoke(
            {
                'messages': messages,
                'facts': facts,
                'topic_notes': state.get('topic_notes'),
                'n_summarized': 0,
            },
            config,
        )
