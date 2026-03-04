# triton-api-client

Lightweight client environment for interacting with [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) via the [KServe v2 protocol](https://kserve.github.io/website/latest/modelserving/data_plane/v2_protocol/) and Triton's generate extension for LLM text generation. Powered by [Flox](https://flox.dev).

Provides an interactive chat REPL (`triton-chat`), example scripts for text generation, streaming, tensor inference, and server introspection -- everything needed to develop against a Triton server without installing dependencies globally.

## What's in the environment

| Component | Description |
|-----------|-------------|
| `triton-chat` | Interactive multi-turn chat REPL with streaming and markdown rendering |
| `examples/generate.py` | Single text generation via the generate endpoint |
| `examples/streaming.py` | Streaming text generation via SSE |
| `examples/infer.py` | KServe v2 tensor inference using `tritonclient.http` |
| `examples/metadata.py` | Server health checks and model metadata/config introspection |

## Triton API endpoints used

This client targets two distinct Triton interfaces:

### Generate extension (LLM text generation)

The [generate extension](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md) is Triton's interface for text-generation models (e.g. vLLM, TensorRT-LLM backends). It is **not** part of the core KServe v2 spec -- it is a Triton-specific extension.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/models/{model}/generate` | POST | Single text generation (blocking) |
| `/v2/models/{model}/generate_stream` | POST | Streaming text generation (SSE) |

**Request format** (both endpoints):

```json
{
  "text_input": "What is the capital of France?",
  "parameters": {
    "max_tokens": 256
  }
}
```

**Response format** (generate):

```json
{
  "model_name": "my-llm",
  "model_version": "1",
  "text_output": "The capital of France is Paris."
}
```

**Response format** (generate_stream): Server-Sent Events, one per token:

```
data: {"text_output": "The"}
data: {"text_output": " capital"}
data: {"text_output": " of"}
...
```

### KServe v2 inference protocol

The [KServe v2 inference protocol](https://kserve.github.io/website/latest/modelserving/data_plane/v2_protocol/) is the standard tensor-in/tensor-out interface for any model type (classification, detection, embedding, etc.).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/health/live` | GET | Server liveness probe |
| `/v2/health/ready` | GET | Server readiness probe |
| `/v2/models/{model}` | GET | Model metadata (inputs, outputs, datatypes) |
| `/v2/models/{model}/config` | GET | Model configuration |
| `/v2/models/{model}/infer` | POST | Tensor inference |

The `infer` endpoint uses structured tensor payloads with typed inputs/outputs, accessed via the `tritonclient.http` Python library rather than raw HTTP.

### Key difference from OpenAI-compatible APIs

Triton does **not** use the OpenAI chat completions format. There are no `/v1/chat/completions` or `/v1/models` endpoints and no API key authentication. Multi-turn conversation state must be managed client-side by building the full prompt from history (which `triton-chat` handles automatically).

## Quick start

```bash
cd ~/dev/triton-api-client
flox activate

# Interactive chat (requires a running Triton server with a text-generation model)
TRITON_MODEL=my-llm triton-chat

# Single text generation
TRITON_MODEL=my-llm python examples/generate.py "What is the capital of France?"

# Streaming text generation
TRITON_MODEL=my-llm python examples/streaming.py "Explain quantum computing"

# KServe v2 tensor inference
TRITON_MODEL=my-model python examples/infer.py "hello world"

# Server health and model metadata
python examples/metadata.py
TRITON_MODEL=my-model python examples/metadata.py
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_API_BASE` | `http://localhost:8000` | Triton server URL (scheme + host + port, no path suffix) |
| `TRITON_MODEL` | _(none)_ | Model name as registered in Triton's model repository |
| `TRITON_SYSTEM_PROMPT` | `You are a helpful assistant.` | System prompt prepended to conversations in `triton-chat` |

All variables are set with `${VAR:-default}` fallbacks in the Flox `on-activate` hook, so they can be overridden at activation time or per-command:

```bash
# Override at activation time (persists for session)
TRITON_API_BASE=http://gpu-server:8000 TRITON_MODEL=ensemble flox activate

# Override per-command
TRITON_MODEL=llama python examples/generate.py "Hello"
```

**Note**: Unlike vLLM's OpenAI-compatible API, Triton has no built-in authentication -- there is no API key variable. If your Triton deployment is behind a reverse proxy with auth, handle that separately.

## Chat CLI

`triton-chat` is an interactive REPL that streams responses from Triton's `generate_stream` endpoint and renders output as markdown using [Rich](https://github.com/Textualize/rich).

### Commands

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation history and start fresh |
| `/model [name]` | Show current model, or switch to a different model |
| `/system [prompt]` | Show current system prompt, or set a new one |
| `/help` | Show available commands |
| `/quit` | Exit the chat (also: `/exit`, Ctrl+C, Ctrl+D) |

### Multi-turn conversation

Since Triton's generate endpoint accepts a raw text prompt (not structured chat messages), `triton-chat` builds multi-turn context client-side using a simple template:

```
<system prompt>

User: <first message>
Assistant: <first response>
User: <second message>
Assistant:
```

The full prompt is sent on each turn, so the model sees the complete conversation history. Use `/clear` to reset when the context gets too long.

### Example session

```
$ TRITON_MODEL=llama triton-chat
triton-chat connected to http://localhost:8000
Model: llama
Type /help for commands, /quit to exit.

you> What is 2 + 2?

2 + 2 = 4.

you> And if you multiply that by 3?

4 multiplied by 3 is 12.

you> /model mixtral
Model set to mixtral
you> /clear
Conversation cleared.
you> /quit
Bye!
```

## Example scripts

### `examples/generate.py` -- Single text generation

Sends a one-shot prompt to `POST /v2/models/{model}/generate` and prints the response. Analogous to a non-streaming chat completion.

```bash
python examples/generate.py "What is the capital of France?"
python examples/generate.py                                    # defaults to "Hello!"
```

Uses `requests` directly -- no `tritonclient` dependency needed for the generate extension.

### `examples/streaming.py` -- Streaming text generation

Sends a prompt to `POST /v2/models/{model}/generate_stream` and prints tokens as they arrive via Server-Sent Events. Analogous to a streaming chat completion.

```bash
python examples/streaming.py "Write a haiku about Triton"
```

Parses SSE `data:` lines and extracts `text_output` from each JSON payload. Tokens are printed with `flush=True` for real-time output.

### `examples/infer.py` -- KServe v2 tensor inference

Demonstrates the full `tritonclient.http` workflow for structured tensor inference:

1. Create an `InferenceServerClient` connection
2. Build an `InferInput` tensor from a numpy array
3. Declare `InferRequestedOutput` tensors
4. Call `client.infer()` to run the model
5. Read results with `result.as_numpy()`

```bash
python examples/infer.py "classify this text"
python examples/infer.py                        # defaults to "hello world"
```

**Adapting for your model**: The script has three constants at the top that you should change to match your model's metadata (use `examples/metadata.py` to discover them):

```python
INPUT_NAME = "text_input"       # name of the model's input tensor
OUTPUT_NAME = "text_output"     # name of the model's output tensor
DATATYPE = "BYTES"              # BYTES for strings, FP32 for floats, etc.
```

**Common datatype mappings**:

| Triton Datatype | numpy dtype | Use case |
|-----------------|-------------|----------|
| `BYTES` | `object` | String inputs/outputs |
| `FP32` | `np.float32` | Floating-point tensors |
| `FP16` | `np.float16` | Half-precision tensors |
| `INT32` | `np.int32` | Integer inputs (e.g. token IDs) |
| `INT64` | `np.int64` | Long integer inputs |
| `BOOL` | `np.bool_` | Boolean flags |

**Example: adapting for a float-input model**:

```python
INPUT_NAME = "input"
OUTPUT_NAME = "output"
DATATYPE = "FP32"

# Build input tensor -- shape [1, 3] with 3 float features
input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
```

### `examples/metadata.py` -- Server health and model metadata

Queries server health and model introspection endpoints via `tritonclient.http`. Useful for discovering model input/output names, shapes, and datatypes before writing inference code.

```bash
# Server health only (no TRITON_MODEL needed)
python examples/metadata.py

# Server health + specific model metadata
python examples/metadata.py my-model
TRITON_MODEL=my-model python examples/metadata.py
```

**Output includes**:

- **Server health**: `is_server_live()`, `is_server_ready()`
- **Model readiness**: `is_model_ready(model)`
- **Model metadata**: Name, versions, platform, input tensors (name, shape, datatype), output tensors (name, shape, datatype)
- **Model config**: Full JSON configuration from `get_model_config()`

**Example output**:

```
=== Server Health ===
  Live:  True
  Ready: True

=== Model: my-llm ===
  Ready: True

  Metadata:
    Name:     my-llm
    Versions: ['1']
    Platform: python

  Inputs:
    - text_input            shape=[-1]  dtype=BYTES

  Outputs:
    - text_output           shape=[-1]  dtype=BYTES

  Config:
    { ... }
```

## Flox environment details

### Installed packages

| Package | Purpose |
|---------|---------|
| `python312` | Python 3.12 interpreter |
| `uv` | Fast Python package installer (creates venv, installs pip packages) |
| `gcc-unwrapped` | Provides `libstdc++.so.6` needed by numpy's C extensions |
| `zlib` | Provides `libz.so.1` needed by numpy's C extensions |

### Pip packages (installed in venv on first activation)

| Package | Purpose |
|---------|---------|
| `tritonclient[http]` | Official Triton HTTP client library (includes numpy) |
| `rich` | Terminal markdown rendering for `triton-chat` |
| `requests` | HTTP client for generate/generate_stream endpoints |

### Activation behavior

On `flox activate`:

1. Sets `TRITON_API_BASE`, `TRITON_MODEL`, and `TRITON_SYSTEM_PROMPT` with fallback defaults
2. Adds `$FLOX_ENV/lib` to `LD_LIBRARY_PATH` (numpy needs `libstdc++.so.6` and `libz.so.1` from Nix packages)
3. Creates a Python venv in `$FLOX_ENV_CACHE/venv` (if it doesn't exist)
4. Installs pip packages on first activation (skips if `$VENV/.installed` marker exists)
5. Adds the project root and venv `bin/` to `PATH` so `triton-chat` is available as a command

To force a clean reinstall of pip packages:

```bash
rm -rf .flox/cache/venv
flox activate
```

## Troubleshooting

### `TRITON_MODEL is not set`

All scripts except `metadata.py` require `TRITON_MODEL`. Set it before running:

```bash
export TRITON_MODEL=my-model
# or per-command
TRITON_MODEL=my-model python examples/generate.py "hello"
```

### Connection refused errors

The Triton server is not running at `TRITON_API_BASE` (default `http://localhost:8000`). Verify:

```bash
curl http://localhost:8000/v2/health/ready
```

If your server is on a different host/port:

```bash
export TRITON_API_BASE=http://gpu-server:8001
```

### `404 Not Found` on generate endpoint

The model may not support the generate extension. Only text-generation backends (vLLM, TensorRT-LLM, Python backend with generate support) expose `/v2/models/{model}/generate`. Use `examples/metadata.py` to check the model's inputs/outputs and switch to `examples/infer.py` for standard tensor inference.

### `ImportError: libstdc++.so.6` or `libz.so.1`

The Flox environment should handle this automatically via `LD_LIBRARY_PATH`. If it occurs, force a clean venv:

```bash
rm -rf .flox/cache/venv
flox activate
```

### `tritonclient` host format

`tritonclient.http.InferenceServerClient` expects `host:port` without a scheme prefix (not `http://host:port`). The example scripts strip the scheme automatically, but if you use the library directly, pass just `localhost:8000`.

### Wrong input/output tensor names in `infer.py`

Every model has different tensor names, shapes, and datatypes. Run `examples/metadata.py` first to discover them, then update the constants at the top of `infer.py`:

```bash
# Discover model interface
TRITON_MODEL=my-model python examples/metadata.py

# Then edit infer.py constants to match
```

## File structure

```
triton-api-client/
  .flox/
    env/manifest.toml           # Flox environment (python312, uv, gcc-unwrapped, zlib, venv setup)
  triton-chat                   # Interactive chat REPL (Python, executable)
  examples/
    generate.py                 # Single text generation via generate endpoint
    streaming.py                # Streaming text generation via generate_stream SSE
    infer.py                    # KServe v2 tensor inference via tritonclient.http
    metadata.py                 # Server health & model metadata introspection
  README.md
```

## Related documentation

- [Triton Inference Server](https://github.com/triton-inference-server/server) -- The server this client targets
- [KServe v2 Inference Protocol](https://kserve.github.io/website/latest/modelserving/data_plane/v2_protocol/) -- Standard tensor inference protocol
- [Triton Generate Extension](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md) -- Text generation endpoint spec
- [tritonclient Python API](https://github.com/triton-inference-server/client) -- Official client library documentation
- [Flox](https://flox.dev) -- Environment manager powering this project
