#!/usr/bin/env python3
"""KServe v2 async tensor inference using tritonclient.grpc.aio.

Demonstrates parallel inference with asyncio.gather() -- sends 3 requests
concurrently and collects results.

Adapt the INPUT_NAME, OUTPUT_NAME, shape, and datatype for your model --
check them with: python examples/kserve/metadata.py

Usage:
  python examples/kserve/infer_async.py
  python examples/kserve/infer_async.py "first prompt" "second prompt" "third prompt"
"""

import asyncio
import os
import sys

import numpy as np
import tritonclient.grpc.aio as grpcclient

TRITON_API_BASE = os.environ.get("TRITON_API_BASE", "http://localhost:8000")
TRITON_GRPC_PORT = os.environ.get("TRITON_GRPC_PORT", "8001")
TRITON_MODEL = os.environ.get("TRITON_MODEL", "")

# ── Adapt these to match your model's metadata ──────────────────────
INPUT_NAME = "text_input"       # name of the model's input tensor
OUTPUT_NAME = "text_output"     # name of the model's output tensor
DATATYPE = "BYTES"              # BYTES for strings, FP32 for floats, etc.
# ────────────────────────────────────────────────────────────────────

DEFAULT_PROMPTS = ["hello world", "what is AI?", "explain gravity"]


async def infer_one(client, model, text):
    """Run a single inference request and return the output."""
    input_data = np.array([text], dtype=object)

    infer_input = grpcclient.InferInput(INPUT_NAME, input_data.shape, DATATYPE)
    infer_input.set_data_from_numpy(input_data)

    infer_output = grpcclient.InferRequestedOutput(OUTPUT_NAME)

    result = await client.infer(
        model_name=model,
        inputs=[infer_input],
        outputs=[infer_output],
    )

    output = result.as_numpy(OUTPUT_NAME)
    values = []
    for item in output.flat:
        values.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return values


async def main():
    if not TRITON_MODEL:
        print("Error: TRITON_MODEL is not set.", file=sys.stderr)
        sys.exit(1)

    prompts = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PROMPTS

    # Derive gRPC URL: strip scheme, replace port with gRPC port
    host_port = TRITON_API_BASE.replace("http://", "").replace("https://", "")
    grpc_host = host_port.split(":")[0] or "localhost"
    grpc_url = f"{grpc_host}:{TRITON_GRPC_PORT}"

    client = grpcclient.InferenceServerClient(url=grpc_url)

    # Send all requests concurrently
    tasks = [infer_one(client, TRITON_MODEL, p) for p in prompts]
    results = await asyncio.gather(*tasks)

    for prompt, output_values in zip(prompts, results):
        print(f"Input:  {prompt}")
        for v in output_values:
            print(f"Output: {v}")
        print()

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
