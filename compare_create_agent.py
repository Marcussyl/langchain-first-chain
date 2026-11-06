#!/usr/bin/env python3
"""Phase 1 optional: compare create_agent with our hand-built graph_core.

Run:
  uv run python compare_create_agent.py

This is a learning sidecar — it does not replace graph_core / chatbot.py.
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from chatbot import SYSTEM_PROMPT, get_time, model
from graph_core import build_graph, pending_interrupts


def _node_names(compiled) -> list[str]:
  return [name for name in compiled.nodes if not str(name).startswith('__')]


def _print_messages(label: str, messages: list) -> None:
  print(f'--- {label} ---')
  for msg in messages[-6:]:
    kind = type(msg).__name__
    tool_calls = getattr(msg, 'tool_calls', None) or []
    extra = ''
    if tool_calls:
      extra = f' tool_calls={[c.get("name") for c in tool_calls]}'
    if isinstance(msg, ToolMessage):
      extra = f' name={msg.name}'
    content = str(getattr(msg, 'content', '') or '')[:100]
    print(f'  {kind}:{extra} {content!r}')


def main() -> None:
  print('Same model + get_time tool; two different agent graphs.\n')

  # --- A: create_agent (high-level ReAct) ---
  agent = create_agent(
    model=model,
    tools=[get_time],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
  )
  agent_config = {'configurable': {'thread_id': 'compare-create-agent'}}
  agent_out = agent.invoke(
    {'messages': [HumanMessage(content='What time is it right now? Use get_time.')]},
    agent_config,
  )

  # --- B: our tutor graph (custom nodes + HITL) ---
  # Approve tools automatically so this script is non-interactive.
  tutor = build_graph(InMemorySaver())
  tutor_config = {'configurable': {'thread_id': 'compare-graph-core'}}
  tutor_out = tutor.invoke(
    {
      'messages': [HumanMessage(content='What time is it right now? Use get_time.')],
      'facts': {},
      'topic_notes': None,
      'n_summarized': 0,
    },
    tutor_config,
  )
  while pending_interrupts(tutor_out):
    tutor_out = tutor.invoke(Command(resume=True), tutor_config)

  print('Nodes')
  print(f'  create_agent : {_node_names(agent)}')
  print(f'  graph_core   : {_node_names(tutor)}')
  print()
  print('Shape (mental model)')
  print('  create_agent : START -> model <-> tools -> END')
  print('  graph_core   : START -> ingest -> prepare -> generate <-> tools -> END')
  print('                 (tools node: interrupt HITL, then ToolNode)')
  print()
  print('What create_agent gives you for free')
  print('  - ReAct loop (model/tools conditional edges)')
  print('  - ToolNode + tool binding')
  print('  - Optional checkpointer / middleware (HITL, etc.)')
  print()
  print('What graph_core adds on purpose (learning + tutor memory)')
  print('  - ingest / prepare (facts harvest, soft compress, SystemMessage rebuild)')
  print('  - hard trim + memory debug prints')
  print('  - explicit HITL interrupt before tools')
  print('  - Studio/CLI share the same graph_core')
  print()
  _print_messages('create_agent messages (tail)', list(agent_out.get('messages') or []))
  print()
  _print_messages('graph_core messages (tail)', list(tutor_out.get('messages') or []))
  print()
  print('Takeaway: real apps usually start with create_agent.')
  print('graph_core here is the tutorial — open every step (memory, HITL, Studio).')


if __name__ == '__main__':
  main()
