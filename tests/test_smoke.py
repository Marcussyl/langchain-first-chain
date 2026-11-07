#!/usr/bin/env python3
"""Phase 5 smoke tests — no Ollama / network required."""

from langchain_core.messages import HumanMessage, SystemMessage

import settings
from chatbot import (
  format_facts,
  get_time,
  hard_trim,
  harvest_facts,
  make_system,
  pinned_facts,
  session_file,
)
from facts import SessionFacts
from graph_core import (
  _parse_tool_approval,
  build_graph,
  empty_state,
  pending_interrupts,
)


def test_harvest_and_format_facts() -> None:
  facts: dict[str, str] = {}
  harvest_facts('My name is Ada and my favorite color is teal.', facts)
  assert facts.get('user_name') == 'Ada'
  assert facts.get('favorite_color') == 'teal'
  text = format_facts(facts)
  assert 'Ada' in text
  assert 'teal' in text


def test_session_facts_rejects_invalid_and_unknown_keys() -> None:
  model = SessionFacts.from_mapping(
    {'user_name': 'Ada', 'favorite_color': 'not a color!!!', 'extra': 'drop-me'}
  )
  stored = model.stored_dict()
  assert stored == {'user_name': 'Ada'}
  assert pinned_facts({'user_name': 'Bob', 'nope': 1}) == {'user_name': 'Bob'}


def test_session_facts_harvest_merge() -> None:
  base = SessionFacts(user_name='Ada')
  merged = base.harvest_text('My favorite color is teal.')
  assert merged.user_name == 'Ada'
  assert merged.favorite_color == 'teal'


def test_make_system_pins_facts() -> None:
  msg = make_system({'user_name': 'Ada'}, 'We talked about graphs.')
  assert isinstance(msg, SystemMessage)
  assert 'Ada' in msg.content
  assert 'graphs' in msg.content


def test_hard_trim_keeps_system() -> None:
  messages = [make_system({}, None), HumanMessage(content='hi')]
  trimmed = hard_trim(messages)
  assert trimmed
  assert isinstance(trimmed[0], SystemMessage)


def test_get_time_tool_returns_timestamp() -> None:
  value = get_time.invoke({})
  assert isinstance(value, str)
  assert 'T' in value  # ISO-like


def test_session_file_stays_under_sessions() -> None:
  path = session_file('demo')
  assert path.name == 'demo.json'
  assert path.parent.name == 'sessions'


def test_parse_tool_approval_variants() -> None:
  assert _parse_tool_approval(True) is True
  assert _parse_tool_approval(False) is False
  assert _parse_tool_approval('yes') is True
  assert _parse_tool_approval('n') is False
  assert _parse_tool_approval({'approved': True}) is True
  assert _parse_tool_approval({'type': 'reject'}) is False


def test_build_graph_has_expected_nodes() -> None:
  graph = build_graph()
  names = {name for name in graph.nodes if not str(name).startswith('__')}
  assert names == {'ingest', 'prepare', 'generate', 'tools'}


def test_empty_state_has_system_message() -> None:
  state = empty_state()
  assert state['messages']
  assert isinstance(state['messages'][0], SystemMessage)
  assert state['facts'] == {}


def test_pending_interrupts_empty_on_plain_dict() -> None:
  assert pending_interrupts({'messages': []}) == []
  assert pending_interrupts('not-a-dict') == []


def test_memory_budget_settings_are_positive() -> None:
  assert settings.MAX_CONTEXT_TOKENS >= 256
  assert 0 < settings.SUMMARY_TRIGGER_TOKENS < settings.MAX_CONTEXT_TOKENS


def test_tracing_status_line_shapes(monkeypatch) -> None:
  monkeypatch.setattr(settings, 'LANGSMITH_TRACING', False)
  monkeypatch.setattr(settings, 'LANGSMITH_API_KEY', '')
  assert 'OFF' in settings.tracing_status_line()

  monkeypatch.setattr(settings, 'LANGSMITH_TRACING', True)
  monkeypatch.setattr(settings, 'LANGSMITH_API_KEY', 'lsv2_test')
  monkeypatch.setattr(settings, 'LANGSMITH_PROJECT', 'demo-project')
  line = settings.tracing_status_line()
  assert 'ON' in line
  assert 'demo-project' in line
  assert 'lsv2_test' not in line
