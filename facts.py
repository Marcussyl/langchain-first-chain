#!/usr/bin/env python3
"""Structured session facts (Pydantic) + regex harvest helpers."""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, field_validator


FACT_PATTERNS: tuple[tuple[str, str], ...] = (
  (r'(?i)\bmy name is\s+([A-Za-z]+)', 'user_name'),
  (r'(?i)\bmy name if\s+([A-Za-z]+)', 'user_name'),
  (r'(?i)\byou are\s+([A-Za-z]+)', 'assistant_name'),
  (r'(?i)\byour name is\s+([A-Za-z]+)', 'assistant_name'),
  (r'(?i)\bmy fav(?:orite)? color is\s+([A-Za-z]+)', 'favorite_color'),
  (r'(?i)\bmy fav(?:orite)? one is\s+([A-Za-z]+)', 'favorite_item'),
)

FACT_LABELS: dict[str, str] = {
  'user_name': "The user's name is",
  'assistant_name': "The assistant's name is",
  'favorite_color': "The user's favorite color is",
  'favorite_item': "The user's favorite item/brand is",
}


class SessionFacts(BaseModel):
  """Known personal facts pinned in Python (survive summarizer glitches).

  Regex still finds candidates in user text; Pydantic validates and stores them.
  Unknown keys from old JSON are ignored (extra='ignore').
  """

  model_config = ConfigDict(extra='ignore')

  user_name: str | None = None
  assistant_name: str | None = None
  favorite_color: str | None = None
  favorite_item: str | None = None

  _name_re: ClassVar[re.Pattern[str]] = re.compile(r'^[A-Za-z]+$')

  @field_validator(
    'user_name',
    'assistant_name',
    'favorite_color',
    'favorite_item',
    mode='before',
  )
  @classmethod
  def _normalize_token(cls, value: object) -> str | None:
    if value is None:
      return None
    text = str(value).strip()
    if not text:
      return None
    # Keep single alphabetic tokens (same constraint as the old regex groups).
    if not cls._name_re.fullmatch(text):
      return None
    return text

  def stored_dict(self) -> dict[str, str]:
    """JSON / checkpointer shape: only set keys as strings."""
    return {key: value for key, value in self.model_dump().items() if value}

  @classmethod
  def from_mapping(cls, raw: object) -> SessionFacts:
    """Load from session JSON or graph state; bad shapes become empty facts."""
    if not isinstance(raw, dict):
      return cls()
    return cls.model_validate(raw)

  def harvest_text(self, user_text: str) -> SessionFacts:
    """Return a copy with any new facts found in user_text merged in."""
    updates: dict[str, str] = {}
    for pattern, key in FACT_PATTERNS:
      match = re.search(pattern, user_text)
      if match:
        updates[key] = match.group(1)
    if not updates:
      return self
    return self.model_copy(update=updates)

  def format_lines(self) -> str:
    lines = [
      f'- {FACT_LABELS[key]} {value}.'
      for key, value in self.stored_dict().items()
      if key in FACT_LABELS
    ]
    return '\n'.join(lines) if lines else '(none yet)'
