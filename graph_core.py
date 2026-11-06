#!/usr/bin/env python3
"""Shared LangGraph tutor: ingest -> prepare -> generate <-> tools (CLI and langgraph.dev)."""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

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


def _latest_tool_calls(messages: list) -> list[dict]:
  """Summarize tool_calls on the newest AIMessage (JSON-safe for interrupt())."""
  for msg in reversed(messages):
    if not isinstance(msg, AIMessage):
      continue
    calls = getattr(msg, 'tool_calls', None) or []
    if not calls:
      continue
    return [
      {
        'name': call.get('name'),
        'args': call.get('args') or {},
        'id': call.get('id'),
      }
      for call in calls
    ]
  return []


def _parse_tool_approval(decision: Any) -> bool:
  """Accept bool, yes/no strings, or {approved: bool} / {type: approve|reject}."""
  if isinstance(decision, bool):
    return decision
  if isinstance(decision, dict):
    if 'approved' in decision:
      return bool(decision['approved'])
    decision_type = str(decision.get('type') or '').strip().lower()
    if decision_type in {'approve', 'approved', 'yes', 'y'}:
      return True
    if decision_type in {'reject', 'rejected', 'no', 'n'}:
      return False
  if isinstance(decision, str):
    return decision.strip().lower() in {'y', 'yes', 'true', 'approve', 'approved', '1'}
  return False


def tools_node(state: ChatState) -> dict:
  """HITL: interrupt() for approval, then ToolNode (or rejected ToolMessages).

  On resume the node restarts from the top — keep interrupt() before side effects.
  Requires a checkpointer (CLI SqliteSaver or Studio platform persistence).
  """
  pending = _latest_tool_calls(list(state['messages']))
  decision = interrupt({
    'action': 'approve_tool_calls',
    'tool_calls': pending,
    'hint': 'Resume with true/false, yes/no, or {approved: bool}',
  })
  if not _parse_tool_approval(decision):
    print('(HITL: tools rejected by human)')
    rejected = [
      ToolMessage(
        content='Tool call rejected by human.',
        tool_call_id=call['id'],
        name=call.get('name') or 'tool',
      )
      for call in pending
      if call.get('id')
    ]
    return {'messages': rejected}

  result = _tool_node.invoke({'messages': list(state['messages'])})
  new_messages = list(result.get('messages') or [])
  for msg in new_messages:
    if isinstance(msg, ToolMessage):
      print(f'(tool: {msg.name} -> {msg.content})')
  return {'messages': new_messages}


def build_graph(checkpointer: Any | None = None):
  """ingest -> prepare -> generate <-> tools (HITL interrupt before tools).

  Pass a checkpointer for the CLI. For langgraph.dev / Studio, omit it —
  the API injects its own persistence and rejects a custom SqliteSaver.
  Interrupts need that persistence (CLI or Studio) — they will not work
  on a graph compiled with checkpointer=None outside the platform.
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


def pending_interrupts(result: Any) -> list:
  """Normalize __interrupt__ from invoke() (list of Interrupt or empty)."""
  if not isinstance(result, dict):
    return []
  items = result.get('__interrupt__') or []
  return list(items)


def print_checkpoint_history(graph, config: dict, limit: int = 8) -> None:
  """Show recent checkpoints for this thread (time-travel practice)."""
  print(f'(checkpoints for thread_id={config.get("configurable", {}).get("thread_id")})')
  for index, snap in enumerate(graph.get_state_history(config)):
    if index >= limit:
      print(f'  … truncated after {limit}')
      break
    cfg = (snap.config or {}).get('configurable') or {}
    checkpoint_id = cfg.get('checkpoint_id')
    next_nodes = list(snap.next or ())
    print(f'  [{index}] checkpoint_id={checkpoint_id}  next={next_nodes or ["(done)"]}')


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
