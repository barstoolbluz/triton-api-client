#!/usr/bin/env python3
"""Streaming text generation via a Triton Inference Server generate_stream endpoint.

Usage:
  python examples/streaming.py "Explain quantum computing"
"""

import json
import os
import sys

import requests

TRITON_API_BASE = os.environ.get("TRITON_API_BASE", "http://localhost:8000")
TRITON_MODEL = os.environ.get("TRITON_MODEL", "")

if not TRITON_MODEL:
    print("Error: TRITON_MODEL is not set.", file=sys.stderr)
    sys.exit(1)

prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello!"

url = f"{TRITON_API_BASE}/v2/models/{TRITON_MODEL}/generate_stream"
payload = {
    "text_input": prompt,
    "parameters": {"max_tokens": 256},
}

with requests.post(url, json=payload, stream=True, timeout=120) as resp:
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = json.loads(line[len("data:"):].strip())
        text = data.get("text_output", "")
        if text:
            print(text, end="", flush=True)
print()
