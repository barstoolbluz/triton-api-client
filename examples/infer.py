#!/usr/bin/env python3
"""KServe v2 tensor inference using tritonclient.http.

Demonstrates creating InferInput/InferRequestedOutput tensors and calling
client.infer(). The default example sends a string input suitable for text
models (e.g. identity, tokenizer, or text-classification models).

Adapt the INPUT_NAME, OUTPUT_NAME, shape, and datatype for your model --
check them with: python examples/metadata.py

Usage:
  python examples/infer.py                          # uses "hello world"
  python examples/infer.py "classify this text"
"""

import os
import sys

import numpy as np
import tritonclient.http as httpclient

TRITON_API_BASE = os.environ.get("TRITON_API_BASE", "http://localhost:8000")
TRITON_MODEL = os.environ.get("TRITON_MODEL", "")

# ── Adapt these to match your model's metadata ──────────────────────
INPUT_NAME = "text_input"       # name of the model's input tensor
OUTPUT_NAME = "text_output"     # name of the model's output tensor
DATATYPE = "BYTES"              # BYTES for strings, FP32 for floats, etc.
# ────────────────────────────────────────────────────────────────────

if not TRITON_MODEL:
    print("Error: TRITON_MODEL is not set.", file=sys.stderr)
    sys.exit(1)

# Strip http(s):// -- tritonclient wants host:port only
server_url = TRITON_API_BASE.replace("http://", "").replace("https://", "")

client = httpclient.InferenceServerClient(url=server_url)

# Build input tensor
text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "hello world"
input_data = np.array([text], dtype=object)  # shape [1], BYTES

infer_input = httpclient.InferInput(INPUT_NAME, input_data.shape, DATATYPE)
infer_input.set_data_from_numpy(input_data)

infer_output = httpclient.InferRequestedOutput(OUTPUT_NAME)

# Run inference
result = client.infer(
    model_name=TRITON_MODEL,
    inputs=[infer_input],
    outputs=[infer_output],
)

# Read output
output = result.as_numpy(OUTPUT_NAME)
for item in output.flat:
    value = item.decode("utf-8") if isinstance(item, bytes) else str(item)
    print(value)
