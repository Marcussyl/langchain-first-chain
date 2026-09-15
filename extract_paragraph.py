#!/usr/bin/env python3
"""Extract paragraph metadata with Pydantic structured output (Ollama)."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# Flat schema so local Llama + Ollama json_schema stays reliable.
class ParagraphExtract(BaseModel):
  """Metadata extracted from pasted text. Do not invent missing fields."""

  summary: str = Field(description='A few sentences covering the pasted text.')
  paragraph_count: int = Field(description='How many paragraphs are in the input.')
  author: str | None = Field(
    default=None,
    description='Author only if the text states one. Otherwise null. Do not invent.',
  )
  published_at: str | None = Field(
    default=None,
    description=(
      'Publication date as YYYY-MM-DD only if the text states a date. '
      'Otherwise null. Do not invent.'
    ),
  )


def count_paragraphs(text: str) -> int:
  """Count non-empty blocks split on blank lines (Python, not the model)."""
  parts = [part.strip() for part in text.split('\n\n') if part.strip()]
  return len(parts)


# 1) Prompt template — {paragraph} is filled in at invoke time
prompt = ChatPromptTemplate.from_messages(
  [
    (
      'system',
      'You extract metadata from the pasted text. '
      'Fill summary and paragraph_count from the text. '
      'Set author or published_at only when the text states them; otherwise null. '
      'Never invent an author or a date.',
    ),
    (
      'human',
      '{paragraph}',
    ),
  ]
)

# 2) Local chat model via Ollama (must be running; model must be pulled)
model = ChatOllama(model='llama3.2', temperature=0)

# 3) Structured output instead of StrOutputParser.
# include_raw=True: a bad parse is a dict, not a crash.
structured_model = model.with_structured_output(
  ParagraphExtract,
  method='json_schema',
  include_raw=True,
)

# LCEL: prompt → model constrained to ParagraphExtract
chain = prompt | structured_model

QUIT_COMMANDS = {'q', 'quit', 'exit'}


def read_paragraph() -> str | None:
  """Return pasted text, or None when the user wants to stop."""
  text = input('Paste a paragraph (q to quit): ').strip()
  if text.lower() in QUIT_COMMANDS:
    return None
  return text


def format_optional(value: str | None) -> str:
  """Show missing optional fields as (none), not an invented string."""
  if value is None or not str(value).strip():
    return '(none)'
  return str(value).strip()


def print_extract(parsed: ParagraphExtract, python_count: int) -> None:
  """Print every schema field plus the Python paragraph count check."""
  print(f'summary: {parsed.summary}')
  print(f'paragraph_count: {parsed.paragraph_count}  (python count: {python_count})')
  print(f'author: {format_optional(parsed.author)}')
  print(f'published_at: {format_optional(parsed.published_at)}')


def print_parse_failure(result: dict) -> None:
  """Show that structured parse failed; do not fake fields."""
  print('(structured: parse failed)')
  raw = result.get('raw')
  content = getattr(raw, 'content', raw)
  print(content)
  error = result.get('parsing_error')
  if error:
    print(f'(parse error: {error})')


if __name__ == '__main__':
  print('Paste text to extract summary, paragraph_count, author, published_at.')
  print('Each turn is independent. Empty input is ignored; type q to quit.')
  print()

  while True:
    paragraph = read_paragraph()
    if paragraph is None:
      print('Bye.')
      break
    if not paragraph:
      continue

    python_count = count_paragraphs(paragraph)
    result = chain.invoke({'paragraph': paragraph})
    parsed = result.get('parsed') if isinstance(result, dict) else result
    if isinstance(parsed, ParagraphExtract):
      print_extract(parsed, python_count)
    else:
      print_parse_failure(result if isinstance(result, dict) else {'raw': result})
    print()
