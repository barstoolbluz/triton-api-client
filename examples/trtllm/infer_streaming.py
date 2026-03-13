#!/usr/bin/env python3
"""TRT-LLM streaming inference using tritonclient.grpc with HuggingFace tokenization.

Uses gRPC streaming (start_stream + async_stream_infer) with a callback to
receive partial results. Detokenizes incrementally, printing tokens as they arrive.

Requires model config `decoupled: true` for actual streaming. If the model is
non-decoupled, a single final response is returned (graceful degradation).

Usage:
  python examples/trtllm/infer_streaming.py "The capital of France is"
  python examples/trtllm/infer_streaming.py --max-tokens 128 "Explain quantum computing"
"""

import argparse
import os
import queue
import sys
import threading

import numpy as np
import tritonclient.grpc as grpcclient

TRITON_API_BASE = os.environ.get("TRITON_API_BASE", "http://localhost:8000")
TRITON_GRPC_PORT = os.environ.get("TRITON_GRPC_PORT", "8001")
TRITON_MODEL = os.environ.get("TRITON_MODEL", "")
TRITON_TOKENIZER = os.environ.get("TRITON_TOKENIZER", "")


def resolve_tokenizer(client, model, cli_tokenizer):
    """Resolve tokenizer: CLI arg > env var > model config > error."""
    if cli_tokenizer:
        return cli_tokenizer
    if TRITON_TOKENIZER:
        return TRITON_TOKENIZER
    try:
        config = client.get_model_config(model)
        params = config.config.parameters
        tokenizer_dir = params.get("tokenizer_dir")
        if tokenizer_dir and tokenizer_dir.string_value:
            return tokenizer_dir.string_value
    except Exception:
        pass
    print("Error: No tokenizer specified.", file=sys.stderr)
    print("  Set TRITON_TOKENIZER, pass --tokenizer, or configure tokenizer_dir in model config.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="TRT-LLM streaming gRPC inference")
    parser.add_argument("prompt", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens (default: 256)")
    parser.add_argument("--tokenizer", default="", help="HuggingFace tokenizer name or path")
    args = parser.parse_args()

    if not TRITON_MODEL:
        print("Error: TRITON_MODEL is not set.", file=sys.stderr)
        sys.exit(1)

    # Derive gRPC URL
    host_port = TRITON_API_BASE.replace("http://", "").replace("https://", "")
    grpc_host = host_port.split(":")[0] or "localhost"
    grpc_url = f"{grpc_host}:{TRITON_GRPC_PORT}"

    client = grpcclient.InferenceServerClient(url=grpc_url)

    # Check if model supports decoupled mode (required for streaming)
    is_decoupled = False
    try:
        config = client.get_model_config(TRITON_MODEL)
        is_decoupled = config.config.model_transaction_policy.decoupled
    except Exception:
        pass

    if not is_decoupled:
        print("(Note: model is not in decoupled mode -- falling back to non-streaming inference)", file=sys.stderr)

    # Resolve and load tokenizer
    tokenizer_id = resolve_tokenizer(client, TRITON_MODEL, args.tokenizer)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    # Tokenize
    token_ids = tokenizer.encode(args.prompt)
    seq_len = len(token_ids)
    end_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end_id

    # Build input tensors
    input_ids = grpcclient.InferInput("input_ids", [1, seq_len], "INT32")
    input_ids.set_data_from_numpy(np.array([token_ids], dtype=np.int32))

    input_lengths = grpcclient.InferInput("input_lengths", [1, 1], "INT32")
    input_lengths.set_data_from_numpy(np.array([[seq_len]], dtype=np.int32))

    request_output_len = grpcclient.InferInput("request_output_len", [1, 1], "INT32")
    request_output_len.set_data_from_numpy(np.array([[args.max_tokens]], dtype=np.int32))

    end_id_tensor = grpcclient.InferInput("end_id", [1, 1], "INT32")
    end_id_tensor.set_data_from_numpy(np.array([[end_id]], dtype=np.int32))

    pad_id_tensor = grpcclient.InferInput("pad_id", [1, 1], "INT32")
    pad_id_tensor.set_data_from_numpy(np.array([[pad_id]], dtype=np.int32))

    inputs = [input_ids, input_lengths, request_output_len, end_id_tensor, pad_id_tensor]

    # Only add streaming tensor if model supports decoupled mode
    if is_decoupled:
        streaming_tensor = grpcclient.InferInput("streaming", [1, 1], "BOOL")
        streaming_tensor.set_data_from_numpy(np.array([[True]], dtype=bool))
        inputs.append(streaming_tensor)

    # Set up streaming with queue and sentinel
    result_queue = queue.Queue()

    def callback(result, error):
        result_queue.put((result, error))

    client.start_stream(callback=callback)

    client.async_stream_infer(
        model_name=TRITON_MODEL,
        inputs=inputs,
        outputs=[
            grpcclient.InferRequestedOutput("output_ids"),
            grpcclient.InferRequestedOutput("sequence_length"),
        ],
    )

    # stop_stream blocks until all responses are received, then sends sentinel
    def stop_and_signal():
        client.stop_stream()
        result_queue.put((None, None))

    stop_thread = threading.Thread(target=stop_and_signal, daemon=True)
    stop_thread.start()

    # Consume streaming responses
    prev_text = ""
    response_count = 0

    while True:
        result, error = result_queue.get()

        if result is None and error is None:
            break

        if error:
            print(f"\nStream error: {error}", file=sys.stderr)
            break

        response_count += 1
        out_ids = result.as_numpy("output_ids")       # [1, beam, max_seq_len]
        seq_lengths = result.as_numpy("sequence_length")  # [1, beam]
        output_len = int(seq_lengths[0][0])
        generated_ids = out_ids[0][0][:output_len].tolist()

        # Incremental detokenization
        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        new_text = full_text[len(prev_text):]
        if new_text:
            print(new_text, end="", flush=True)
        prev_text = full_text

    stop_thread.join()
    print()  # final newline

    if response_count == 1:
        print("(Note: received a single response -- model may not be in decoupled mode for streaming)", file=sys.stderr)


if __name__ == "__main__":
    main()
