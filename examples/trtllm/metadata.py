#!/usr/bin/env python3
"""TRT-LLM model metadata and config parameter viewer.

Extends the standard KServe metadata view with TRT-LLM-specific config
parameters like gpt_model_type, tokenizer_dir, and batch scheduler policy.

Usage:
  python examples/trtllm/metadata.py                # uses TRITON_MODEL
  python examples/trtllm/metadata.py qwen2_5_05b_trtllm
"""

import json
import os
import sys

import tritonclient.http as httpclient

TRITON_API_BASE = os.environ.get("TRITON_API_BASE", "http://localhost:8000")
TRITON_MODEL = os.environ.get("TRITON_MODEL", "")

# TRT-LLM config parameters of interest
TRTLLM_PARAMS = [
    "gpt_model_type",
    "gpt_model_path",
    "tokenizer_dir",
    "executor_worker_path",
    "decoupled_mode",
    "batch_scheduler_policy",
    "kv_cache_free_gpu_mem_fraction",
    "max_beam_width",
    "batching_type",
    "enable_chunked_context",
]

# Strip http(s):// -- tritonclient wants host:port only
server_url = TRITON_API_BASE.replace("http://", "").replace("https://", "")

client = httpclient.InferenceServerClient(url=server_url)

# ── Server health ───────────────────────────────────────────────────
print("=== Server Health ===")
try:
    print(f"  Live:  {client.is_server_live()}")
    print(f"  Ready: {client.is_server_ready()}")
except Exception as e:
    print(f"  Could not reach server at {TRITON_API_BASE}: {e}")
    sys.exit(1)

# ── Model metadata ──────────────────────────────────────────────────
model = sys.argv[1] if len(sys.argv) > 1 else TRITON_MODEL

if not model:
    print("\nNo model specified. Set TRITON_MODEL or pass a model name as argument.")
    sys.exit(0)

print(f"\n=== Model: {model} ===")

try:
    ready = client.is_model_ready(model)
    print(f"  Ready: {ready}")
except Exception as e:
    print(f"  Error checking model readiness: {e}")

try:
    meta = client.get_model_metadata(model)
    print(f"\n  Metadata:")
    print(f"    Name:     {meta['name']}")
    print(f"    Versions: {meta.get('versions', 'n/a')}")
    print(f"    Platform: {meta.get('platform', 'n/a')}")

    if "inputs" in meta:
        print(f"\n  Inputs:")
        for inp in meta["inputs"]:
            print(f"    - {inp['name']:20s}  shape={inp['shape']}  dtype={inp['datatype']}")

    if "outputs" in meta:
        print(f"\n  Outputs:")
        for out in meta["outputs"]:
            print(f"    - {out['name']:20s}  shape={out['shape']}  dtype={out['datatype']}")
except Exception as e:
    print(f"  Error fetching metadata: {e}")

# ── TRT-LLM config parameters ───────────────────────────────────────
try:
    config = client.get_model_config(model)
    params = config.get("parameters", {})

    trtllm_values = {}
    for key in TRTLLM_PARAMS:
        entry = params.get(key, {})
        value = entry.get("string_value", "") if isinstance(entry, dict) else ""
        if value:
            trtllm_values[key] = value

    if trtllm_values:
        print(f"\n  TRT-LLM Parameters:")
        for key, value in trtllm_values.items():
            print(f"    {key:40s}  {value}")

    print(f"\n  Full Config:")
    print(f"    {json.dumps(config, indent=4)}")
except Exception as e:
    print(f"  Error fetching config: {e}")
