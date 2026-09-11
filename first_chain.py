#!/usr/bin/env python3
"""First LangChain LCEL chain: Prompt → LLM → output parser (Ollama, no API key)."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# 1) Prompt template — {topic} is filled in at invoke time
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 2 short sentences for a beginner."
)

# 2) Local chat model via Ollama (must be running; model must be pulled)
model = ChatOllama(model="llama3.2", temperature=0)

# 3) Turn the model message into a plain string
parser = StrOutputParser()

# LCEL: pipe each step into the next with |
chain = prompt | model | parser

if __name__ == "__main__":
    result = chain.invoke({"topic": "LangChain"})
    print(result)
