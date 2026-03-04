#!/usr/bin/env python3
"""Single text generation via a Triton Inference Server generate endpoint.

Usage:
  python examples/generate/single.py "What is the capital of France?"
"""

import os
import sys

import requests

TRITON_API_BASE = os.environ.get("TRITON_API_BASE", "http://localhost:8000")
TRITON_MODEL = os.environ.get("TRITON_MODEL", "")

if not TRITON_MODEL:
    print("Error: TRITON_MODEL is not set.", file=sys.stderr)
    sys.exit(1)

prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello!"

url = f"{TRITON_API_BASE}/v2/models/{TRITON_MODEL}/generate"
payload = {
    "text_input": prompt,
    "parameters": {"max_tokens": 256},
}

response = requests.post(url, json=payload, timeout=60)
response.raise_for_status()

print(response.json()["text_output"])
