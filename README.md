# triton-api-client

Lightweight client environment for [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) covering all three client interfaces: the **OpenAI-compatible frontend**, the **generate extension**, and the **KServe v2 tensor inference protocol** (HTTP and gRPC). Powered by [Flox](https://flox.dev).

Provides an interactive chat REPL (`triton-chat`), a health/smoke/benchmark tool (`triton-test`), and example scripts for every interface -- everything needed to develop against a Triton server without installing dependencies globally.

## What's in the environment

| Component | Description |
|-----------|-------------|
| `triton-chat` | Interactive multi-turn chat REPL via OpenAI-compatible frontend |
| `triton-test` | Health check, smoke test, and benchmark tool |
| `examples/openai/` | Chat, streaming, and batch completions via OpenAI SDK |
| `examples/generate/` | Text generation via Triton's generate extension |
| `examples/kserve/` | Tensor inference (HTTP and gRPC) and server metadata |

## Three Triton interfaces

| Interface | Port | Protocol | Use case | Client library |
|-----------|------|----------|----------|----------------|
| OpenAI-compatible frontend | 9000 | HTTP | LLM chat/completions (dominant pattern) | `openai` Python SDK |
| Generate extension | 8000 | HTTP | Triton-specific LLM text generation | `requests` |
| KServe v2 inference | 8000 HTTP / 8001 gRPC | HTTP + gRPC | Standard tensor inference for any model type | `tritonclient.http`, `tritonclient.grpc` |

## Quick start

```bash
cd ~/dev/triton-api-client
flox activate

# ── OpenAI-compatible frontend (port 9000) ──────────────────────────

# Interactive chat
TRITON_MODEL=my-llm triton-chat

# Health check + smoke test
TRITON_MODEL=my-llm triton-test

# Benchmark
TRITON_MODEL=my-llm triton-test bench -n 50 --concurrent 5

# Single completion
TRITON_MODEL=my-llm python examples/openai/chat.py "What is the capital of France?"

# Streaming completion
TRITON_MODEL=my-llm python examples/openai/streaming.py "Explain quantum computing"

# Batch completions from JSON
echo '["Hello", "What is 2+2?"]' | TRITON_MODEL=my-llm python examples/openai/batch.py -

# ── Generate extension (port 8000) ──────────────────────────────────

TRITON_MODEL=my-llm python examples/generate/single.py "Hello!"
TRITON_MODEL=my-llm python examples/generate/streaming.py "Write a haiku"

# ── KServe v2 tensor inference (port 8000 HTTP / 8001 gRPC) ────────

# HTTP inference
TRITON_MODEL=my-model python examples/kserve/infer_http.py "hello world"

# gRPC inference
TRITON_MODEL=my-model python examples/kserve/infer_grpc.py "hello world"

# Async gRPC (3 concurrent requests)
TRITON_MODEL=my-model python examples/kserve/infer_async.py

# Server health and model metadata
python examples/kserve/metadata.py
TRITON_MODEL=my-model python examples/kserve/metadata.py
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_OPENAI_BASE` | `http://localhost:9000/v1` | OpenAI-compatible frontend URL |
| `TRITON_API_KEY` | `EMPTY` | API key for OpenAI frontend auth |
| `TRITON_API_BASE` | `http://localhost:8000` | KServe v2 + generate extension URL |
| `TRITON_MODEL` | _(none)_ | Model name (used across all interfaces) |
| `TRITON_SYSTEM_PROMPT` | `You are a helpful assistant.` | System prompt for `triton-chat` |
| `TRITON_GRPC_PORT` | `8001` | gRPC port for KServe tensor inference |

All variables are set with `${VAR:-default}` fallbacks in the Flox `on-activate` hook, so they can be overridden at activation time or per-command:

```bash
# Override at activation time (persists for session)
TRITON_OPENAI_BASE=http://gpu-server:9000/v1 TRITON_MODEL=llama flox activate

# Override per-command
TRITON_MODEL=llama python examples/openai/chat.py "Hello"
```

## Chat CLI

`triton-chat` is an interactive REPL that uses Triton's OpenAI-compatible frontend (port 9000) to stream chat completions and renders output as markdown using [Rich](https://github.com/Textualize/rich).

### Commands

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation history and start fresh |
| `/model [name]` | Show current model, or switch to a different model |
| `/system [prompt]` | Show current system prompt, or set a new one |
| `/help` | Show available commands |
| `/quit` | Exit the chat (also: `/exit`, Ctrl+C, Ctrl+D) |

### Example session

```
$ TRITON_MODEL=llama triton-chat
triton-chat connected to http://localhost:9000/v1
Model: llama
Type /help for commands, /quit to exit.

you> What is 2 + 2?

2 + 2 = 4.

you> And if you multiply that by 3?

4 multiplied by 3 is 12.

you> /quit
Bye!
```

## Test CLI

`triton-test` checks server connectivity, runs smoke tests, and benchmarks throughput via the OpenAI-compatible frontend.

### Usage

```bash
# Health check + smoke test (exit 0 on success, 1 on failure)
triton-test

# Benchmark with defaults (10 requests, concurrency 1)
triton-test bench

# Heavier load test
triton-test bench -n 50 --concurrent 5 --max-tokens 256

# Custom prompt
triton-test bench --prompt "Summarize the theory of relativity"
```

### What it tests

| Check | Description |
|-------|-------------|
| **Health** | Connects to server, lists available models |
| **Smoke (non-streaming)** | Single completion, reports latency and token count |
| **Smoke (streaming)** | Streaming completion, reports TTFT and chunk count |
| **Benchmark** | Concurrent streaming requests with p50/p90/p99 latency, TTFT, ITL, tokens/sec |

## Example scripts

### OpenAI-compatible frontend (`examples/openai/`)

| Script | Description |
|--------|-------------|
| `chat.py` | Single chat completion (non-streaming) |
| `streaming.py` | Streaming chat completion, prints tokens as they arrive |
| `batch.py` | Batch completions from a JSON array, outputs JSONL with usage stats |

### Generate extension (`examples/generate/`)

| Script | Description |
|--------|-------------|
| `single.py` | Single text generation via `POST /v2/models/{model}/generate` |
| `streaming.py` | Streaming text generation via SSE from `generate_stream` |

### KServe v2 inference (`examples/kserve/`)

| Script | Description |
|--------|-------------|
| `infer_http.py` | Synchronous tensor inference via `tritonclient.http` |
| `infer_grpc.py` | Synchronous tensor inference via `tritonclient.grpc` (port 8001) |
| `infer_async.py` | Async parallel inference via `tritonclient.grpc.aio` with `asyncio.gather()` |
| `metadata.py` | Server health, model metadata, and config introspection |

## Triton API reference

### OpenAI-compatible frontend (port 9000)

The [OpenAI-compatible frontend](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/openai.md) provides standard OpenAI API endpoints. Uses the `openai` Python SDK.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completion (streaming and non-streaming) |

### Generate extension (port 8000)

The [generate extension](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md) is Triton's native LLM text generation interface. Not part of the KServe v2 spec.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/models/{model}/generate` | POST | Single text generation (blocking) |
| `/v2/models/{model}/generate_stream` | POST | Streaming text generation (SSE) |

**Request format**:

```json
{
  "text_input": "What is the capital of France?",
  "parameters": { "max_tokens": 256 }
}
```

### KServe v2 inference protocol (port 8000 HTTP / 8001 gRPC)

The [KServe v2 inference protocol](https://kserve.github.io/website/latest/modelserving/data_plane/v2_protocol/) is the standard tensor-in/tensor-out interface for any model type.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/health/live` | GET | Server liveness probe |
| `/v2/health/ready` | GET | Server readiness probe |
| `/v2/models/{model}` | GET | Model metadata (inputs, outputs, datatypes) |
| `/v2/models/{model}/config` | GET | Model configuration |
| `/v2/models/{model}/infer` | POST | Tensor inference |

The `infer` endpoint uses structured tensor payloads with typed inputs/outputs, accessed via `tritonclient.http` or `tritonclient.grpc`.

## Flox environment details

### Installed packages

| Package | Purpose |
|---------|---------|
| `python312` | Python 3.12 interpreter |
| `uv` | Fast Python package installer (creates venv, installs pip packages) |
| `gcc-unwrapped` | Provides `libstdc++.so.6` needed by numpy/grpcio C extensions |
| `zlib` | Provides `libz.so.1` needed by numpy/grpcio C extensions |

### Pip packages (installed in venv on first activation)

| Package | Purpose |
|---------|---------|
| `tritonclient[all]` | Official Triton client (HTTP + gRPC, includes numpy) |
| `openai` | OpenAI Python SDK for the OpenAI-compatible frontend |
| `rich` | Terminal markdown rendering for `triton-chat` and `triton-test` |
| `requests` | HTTP client for generate extension endpoints |

### Activation behavior

On `flox activate`:

1. Sets `TRITON_OPENAI_BASE`, `TRITON_API_KEY`, `TRITON_API_BASE`, `TRITON_MODEL`, and `TRITON_SYSTEM_PROMPT` with fallback defaults
2. Adds `$FLOX_ENV/lib` to `LD_LIBRARY_PATH` (numpy and grpcio need native libs from Nix packages)
3. Creates a Python venv in `$FLOX_ENV_CACHE/venv` (if it doesn't exist)
4. Installs pip packages on first activation (skips if `$VENV/.installed` marker exists)
5. Adds the project root and venv `bin/` to `PATH` so `triton-chat` and `triton-test` are available as commands

To force a clean reinstall of pip packages:

```bash
rm -rf .flox/cache/venv
flox activate
```

## Troubleshooting

### `TRITON_MODEL is not set`

Most scripts require `TRITON_MODEL`. Set it before running:

```bash
export TRITON_MODEL=my-model
# or per-command
TRITON_MODEL=my-model triton-chat
```

### Connection refused on port 9000 (OpenAI frontend)

The OpenAI-compatible frontend runs on port 9000 by default. Verify it's enabled in your Triton server configuration and reachable:

```bash
curl http://localhost:9000/v1/models
```

If on a different host/port:

```bash
export TRITON_OPENAI_BASE=http://gpu-server:9000/v1
```

### Connection refused on port 8000 (generate/KServe)

The Triton server is not running at `TRITON_API_BASE` (default `http://localhost:8000`):

```bash
curl http://localhost:8000/v2/health/ready
```

### Connection refused on port 8001 (gRPC)

gRPC inference uses port 8001 by default. Verify Triton's gRPC endpoint is enabled:

```bash
# Set a custom gRPC port if needed
export TRITON_GRPC_PORT=8001
```

### `404 Not Found` on generate endpoint

The model may not support the generate extension. Only text-generation backends (vLLM, TensorRT-LLM) expose `/v2/models/{model}/generate`. Use `examples/kserve/metadata.py` to check the model's interface.

### `ImportError: libstdc++.so.6` or `libz.so.1`

The Flox environment should handle this automatically via `LD_LIBRARY_PATH`. If it occurs, force a clean venv:

```bash
rm -rf .flox/cache/venv
flox activate
```

### `tritonclient` host format

`tritonclient.http.InferenceServerClient` expects `host:port` without a scheme prefix (not `http://host:port`). The example scripts strip the scheme automatically.

For gRPC, `tritonclient.grpc.InferenceServerClient` also expects `host:port` without a scheme.

## File structure

```
triton-api-client/
  .flox/
    env/manifest.toml           # Flox environment config
  triton-chat                   # Interactive chat REPL (OpenAI SDK)
  triton-test                   # Health/smoke/benchmark tool (OpenAI SDK)
  examples/
    openai/
      chat.py                   # Single chat completion
      streaming.py              # Streaming chat completion
      batch.py                  # Batch completions from JSON
    generate/
      single.py                 # Single text generation
      streaming.py              # Streaming text generation (SSE)
    kserve/
      infer_http.py             # HTTP tensor inference
      infer_grpc.py             # gRPC tensor inference
      infer_async.py            # Async gRPC parallel inference
      metadata.py               # Server health & model metadata
  README.md
```

## Related documentation

- [Triton Inference Server](https://github.com/triton-inference-server/server) -- The server this client targets
- [Triton OpenAI-Compatible Frontend](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/openai.md) -- OpenAI API compatibility layer
- [Triton Generate Extension](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md) -- Text generation endpoint spec
- [KServe v2 Inference Protocol](https://kserve.github.io/website/latest/modelserving/data_plane/v2_protocol/) -- Standard tensor inference protocol
- [tritonclient Python API](https://github.com/triton-inference-server/client) -- Official client library documentation
- [OpenAI Python SDK](https://github.com/openai/openai-python) -- OpenAI client used for the frontend
- [Flox](https://flox.dev) -- Environment manager powering this project
