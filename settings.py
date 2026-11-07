#!/usr/bin/env python3
"""Shared runtime settings (Ollama model name, memory budgets, LangSmith tracing)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path = ROOT / '.env') -> None:
  """Load KEY=VALUE lines from .env into os.environ if not already set."""
  if not path.is_file():
    return
  for raw in path.read_text(encoding='utf-8').splitlines():
    line = raw.strip()
    if not line or line.startswith('#') or '=' not in line:
      continue
    key, _, value = line.partition('=')
    key = key.strip()
    value = value.strip().strip("'").strip('"')
    if key and key not in os.environ:
      os.environ[key] = value


def _truthy(raw: str | None) -> bool:
  if raw is None:
    return False
  return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_float(name: str, default: float) -> float:
  raw = os.getenv(name)
  if raw is None or not raw.strip():
    return default
  try:
    return float(raw)
  except ValueError:
    return default


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
  raw = os.getenv(name)
  if raw is None or not raw.strip():
    return default
  try:
    value = int(raw)
  except ValueError:
    return default
  return max(minimum, value)


_load_dotenv()

OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2').strip() or 'llama3.2'
OLLAMA_TEMPERATURE = _env_float('OLLAMA_TEMPERATURE', 0.0)
# Soft/hard memory budgets (approximate tokens). Soft trigger should stay below hard max.
MAX_CONTEXT_TOKENS = _env_int('MAX_CONTEXT_TOKENS', 2048, minimum=256)
SUMMARY_TRIGGER_TOKENS = _env_int('SUMMARY_TRIGGER_TOKENS', 1400, minimum=128)
if SUMMARY_TRIGGER_TOKENS >= MAX_CONTEXT_TOKENS:
  SUMMARY_TRIGGER_TOKENS = max(128, MAX_CONTEXT_TOKENS - 1)

# LangSmith / LangChain tracing (optional). Prefer LANGSMITH_* names.
LANGSMITH_API_KEY = (
  os.getenv('LANGSMITH_API_KEY')
  or os.getenv('LANGCHAIN_API_KEY')
  or ''
).strip()
LANGSMITH_PROJECT = (
  os.getenv('LANGSMITH_PROJECT')
  or os.getenv('LANGCHAIN_PROJECT')
  or 'langchain-first-chain'
).strip()
# Explicit LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 wins; else on when a key is set.
_tracing_raw = os.getenv('LANGSMITH_TRACING')
if _tracing_raw is None:
  _tracing_raw = os.getenv('LANGCHAIN_TRACING_V2')
if _tracing_raw is None:
  LANGSMITH_TRACING = bool(LANGSMITH_API_KEY)
else:
  LANGSMITH_TRACING = _truthy(_tracing_raw)

# Mirror into the env names LangChain / LangSmith SDKs read.
if LANGSMITH_API_KEY and 'LANGSMITH_API_KEY' not in os.environ and 'LANGCHAIN_API_KEY' not in os.environ:
  os.environ['LANGSMITH_API_KEY'] = LANGSMITH_API_KEY
if LANGSMITH_TRACING:
  os.environ.setdefault('LANGSMITH_TRACING', 'true')
  os.environ.setdefault('LANGCHAIN_TRACING_V2', 'true')
os.environ.setdefault('LANGSMITH_PROJECT', LANGSMITH_PROJECT)


def tracing_status_line() -> str:
  """One-line CLI hint: whether cloud traces are expected (never prints the key)."""
  if LANGSMITH_TRACING and LANGSMITH_API_KEY:
    return f'(langsmith: tracing ON  project={LANGSMITH_PROJECT})'
  if LANGSMITH_API_KEY and not LANGSMITH_TRACING:
    return (
      '(langsmith: API key set but tracing OFF — set LANGSMITH_TRACING=true in .env)'
    )
  if LANGSMITH_TRACING and not LANGSMITH_API_KEY:
    return (
      '(langsmith: tracing flagged ON but no LANGSMITH_API_KEY — traces will not upload)'
    )
  return (
    '(langsmith: tracing OFF — optional LANGSMITH_API_KEY + LANGSMITH_TRACING=true in .env)'
  )
