#!/usr/bin/env python3
"""Shared runtime settings (Ollama model name, etc.)."""

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


_load_dotenv()

OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
OLLAMA_TEMPERATURE = float(os.getenv('OLLAMA_TEMPERATURE', '0'))
