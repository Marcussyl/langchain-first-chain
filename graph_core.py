#!/usr/bin/env python3
"""Shared LangGraph tutor: ingest -> prepare -> generate (CLI and langgraph dev)."""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages

from chatbot import (
  MAX_CONTEXT_TOKENS,
  apply_soft_compression,
  count_tokens,
  format_facts,
  hard_trim,
  harvest_facts,
  make_system,
  run_turn,
)


class ChatState(TypedDict):
  """Turn state. messages use add_messages so Studio/CLI can send only new humans."""

  messages: Annotated[list, add_messages]
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


def _message_type(msg: Any) -> str | None:
  if isinstance(msg, HumanMessage):
    return 'human'
  msg_type = getattr(msg, 'type', None)
  if isinstance(msg_type, str):
    return msg_type
  if isinstance(msg, dict):
    return msg.get('type') or msg.get('role')
  return None


def _message_content(msg: Any) -> str | None:
  if isinstance(msg, dict):
    content = msg.get('content')
  else:
    content = getattr(msg, 'content', None)
  if content is None:
    return None
  if isinstance(content, str):
    return content
  return str(content)


def _last_human_text(messages: list) -> str | None:
  # Studio/API may pass dicts or LC messages; isinstance alone is not enough.
  for msg in reversed(messages):
    if _message_type(msg) in {'human', 'user'}:
      return _message_content(msg)
  return None


def _replace_messages(messages: list) -> list:
  """Replace the whole message channel (needed after soft-compress / system rebuild)."""
  return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]


def ingest_node(state: ChatState) -> dict:
  """Harvest facts from the latest human turn (CLI/Studio already appended it)."""
  facts = dict(state.get('facts') or {})
  messages = list(state.get('messages') or [])
  text = _last_human_text(messages)
  if text:
    harvest_facts(text, facts)
  return {
    'facts': facts,
    'n_summarized': 0,
  }


def prepare_node(state: ChatState) -> dict:
  """Rebuild SystemMessage, then soft-compress like chatbot.py.

  Studio/API may send only the new HumanMessage — never overwrite that with system.
  """
  facts = dict(state.get('facts') or {})
  notes = state.get('topic_notes')
  messages = list(state.get('messages') or [])
  system = make_system(facts, notes)
  if not messages:
    messages = [system]
  elif isinstance(messages[0], SystemMessage) or _message_type(messages[0]) == 'system':
    messages[0] = system
  else:
    messages = [system, *messages]
  messages, notes, n_summarized = apply_soft_compression(messages, notes, facts)
  return {
    'messages': _replace_messages(messages),
    'facts': facts,
    'topic_notes': notes,
    'n_summarized': n_summarized,
  }


def generate_node(state: ChatState) -> dict:
  """Hard-trim a copy, run one model/tool round, append only the new messages."""
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
  # add_messages: append model/tool turns only (do not resend full history).
  return {'messages': list(new_messages)}


def build_graph(checkpointer: Any | None = None):
  """Three nodes: ingest (facts) then prepare (memory) then generate (model + tool).

  Pass a checkpointer for the CLI. For langgraph.dev / Studio, omit it —
  the API injects its own persistence and rejects a custom SqliteSaver.
  """
  builder = StateGraph(ChatState)
  builder.add_node('ingest', ingest_node)
  builder.add_node('prepare', prepare_node)
  builder.add_node('generate', generate_node)
  builder.add_edge(START, 'ingest')
  builder.add_edge('ingest', 'prepare')
  builder.add_edge('prepare', 'generate')
  builder.add_edge('generate', END)
  if checkpointer is None:
    return builder.compile()
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
