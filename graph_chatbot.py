#!/usr/bin/env python3
"""LangGraph twin of chatbot.py: CLI entry using graph_core + sqlite checkpointer."""

import sqlite3

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from chatbot import (
  MAX_CONTEXT_TOKENS,
  SESSIONS_DIR,
  read_session_id,
  read_user_text,
)
from graph_core import (
  build_graph,
  history_len,
  print_checkpoint_history,
  read_checkpoint,
  stream_with_hitl,
)
from settings import tracing_status_line

# One sqlite file for all thread_id values (session ids). Not chatbot.py JSON.
# langgraph.dev uses platform persistence (graph_server exports graph without SqliteSaver).
CHECKPOINT_PATH = SESSIONS_DIR / 'langgraph.sqlite'


def _read_tool_approval(_interrupts: list) -> bool:
  raw = input('HITL: approve tool call(s)? [y/N]: ').strip().lower()
  return raw in {'y', 'yes'}


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
  print('Streams node updates + Bot tokens; tool calls pause for HITL (y/N).')
  print('Type "history" for checkpoints.')
  print('This file does not share sessions/*.json with chatbot.py.')
  print(f'Hard-trim budget: ~{MAX_CONTEXT_TOKENS} tokens (approximate).')
  print(tracing_status_line())
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
    if user_text.strip().lower() == 'history':
      print_checkpoint_history(graph, config)
      print()
      continue

    state, _ = read_checkpoint(graph, config)
    # add_messages: pass only the new human turn (history lives in the checkpointer).
    stream_with_hitl(
      graph,
      {
        'messages': [HumanMessage(content=user_text)],
        'facts': dict(state.get('facts') or {}),
        'topic_notes': state.get('topic_notes'),
        'n_summarized': 0,
      },
      config,
      read_approval=_read_tool_approval,
    )
