#!/usr/bin/env python3
"""Streaming chat completion via Triton's OpenAI-compatible frontend.

Usage:
  python examples/openai/streaming.py "Explain quantum computing"
"""

import os
import sys

from openai import OpenAI

TRITON_OPENAI_BASE = os.environ.get("TRITON_OPENAI_BASE", "http://localhost:9000/v1")
TRITON_API_KEY = os.environ.get("TRITON_API_KEY", "EMPTY")
TRITON_MODEL = os.environ.get("TRITON_MODEL", "")

if not TRITON_MODEL:
    print("Error: TRITON_MODEL is not set.", file=sys.stderr)
    sys.exit(1)

prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello!"

client = OpenAI(base_url=TRITON_OPENAI_BASE, api_key=TRITON_API_KEY)

stream = client.chat.completions.create(
    model=TRITON_MODEL,
    messages=[{"role": "user", "content": prompt}],
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()
