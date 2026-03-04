#!/usr/bin/env python3
"""Server health and model metadata via tritonclient.http.

Usage:
  python examples/kserve/metadata.py                # show server status + TRITON_MODEL metadata
  python examples/kserve/metadata.py <model_name>   # show metadata for a specific model
"""

import json
import os
import sys

import tritonclient.http as httpclient

TRITON_API_BASE = os.environ.get("TRITON_API_BASE", "http://localhost:8000")
TRITON_MODEL = os.environ.get("TRITON_MODEL", "")

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

try:
    config = client.get_model_config(model)
    print(f"\n  Config:")
    print(f"    {json.dumps(config, indent=4)}")
except Exception as e:
    print(f"  Error fetching config: {e}")
