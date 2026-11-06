#!/usr/bin/env python3
"""Shared LangGraph tutor: ingest -> prepare -> generate <-> tools (CLI and langgraph.dev)."""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from chatbot import (
  MAX_CONTEXT_TOKENS,
  apply_soft_compression,
  count_tokens,
  format_facts,
  get_time,
  hard_trim,
  harvest_facts,
  make_system,
  print_bot,
  reply_chain,
)

# ReAct-style tool execution for the graph (chatbot.py CLI still uses run_turn).
_tool_node = ToolNode([get_time], handle_tool_errors=True)


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
  """Hard-trim a copy, call the tool-bound model, append the AIMessage only.

  Tool rounds are handled by the tools node + conditional edges (not run_turn).
  """
  messages = list(state['messages'])
  to_send = hard_trim(messages)
  # Log memory once per user turn (first generate after a human / after prepare).
  if messages and _message_type(messages[-1]) in {'human', 'user'}:
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

  ai = reply_chain.invoke({'messages': to_send})
  if not isinstance(ai, AIMessage):
    ai = AIMessage(content=str(ai))
  tool_calls = getattr(ai, 'tool_calls', None) or []
  if not tool_calls:
    print_bot(ai)
    print()
  return {'messages': [ai]}


def tools_node(state: ChatState) -> dict:
  """Run tool calls from the latest AIMessage; print each result for the CLI."""
  result = _tool_node.invoke({'messages': list(state['messages'])})
  new_messages = list(result.get('messages') or [])
  for msg in new_messages:
    if isinstance(msg, ToolMessage):
      print(f'(tool: {msg.name} -> {msg.content})')
  return {'messages': new_messages}


def build_graph(checkpointer: Any | None = None):
  """ingest -> prepare -> generate <-> tools (conditional ReAct loop).

  Pass a checkpointer for the CLI. For langgraph.dev / Studio, omit it —
  the API injects its own persistence and rejects a custom SqliteSaver.
  """
  builder = StateGraph(ChatState)
  builder.add_node('ingest', ingest_node)
  builder.add_node('prepare', prepare_node)
  builder.add_node('generate', generate_node)
  builder.add_node('tools', tools_node)
  builder.add_edge(START, 'ingest')
  builder.add_edge('ingest', 'prepare')
  builder.add_edge('prepare', 'generate')
  builder.add_conditional_edges('generate', tools_condition)
  builder.add_edge('tools', 'generate')
  # recursion_limit covers generate<->tools loops (default 25 is usually enough)
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
