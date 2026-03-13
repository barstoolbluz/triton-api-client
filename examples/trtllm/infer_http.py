#!/usr/bin/env python3
"""TRT-LLM tensor inference using tritonclient.http with HuggingFace tokenization.

TRT-LLM models use raw INT32 tensor I/O (input_ids, input_lengths,
request_output_len) instead of text_input/text_output BYTES tensors.
This script handles client-side tokenization and detokenization.

Usage:
  python examples/trtllm/infer_http.py "The capital of France is"
  python examples/trtllm/infer_http.py --max-tokens 128 "Hello, how are you?"
  python examples/trtllm/infer_http.py --tokenizer Qwen/Qwen2.5-0.5B "Hello"
"""

import argparse
import os
import sys

import numpy as np
import tritonclient.http as httpclient

TRITON_API_BASE = os.environ.get("TRITON_API_BASE", "http://localhost:8000")
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
        params = config.get("parameters", {})
        tokenizer_dir = params.get("tokenizer_dir", {})
        value = tokenizer_dir.get("string_value", "") if isinstance(tokenizer_dir, dict) else ""
        if value:
            return value
    except Exception:
        pass
    print("Error: No tokenizer specified.", file=sys.stderr)
    print("  Set TRITON_TOKENIZER, pass --tokenizer, or configure tokenizer_dir in model config.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="TRT-LLM HTTP tensor inference")
    parser.add_argument("prompt", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens (default: 256)")
    parser.add_argument("--tokenizer", default="", help="HuggingFace tokenizer name or path")
    args = parser.parse_args()

    if not TRITON_MODEL:
        print("Error: TRITON_MODEL is not set.", file=sys.stderr)
        sys.exit(1)

    # Strip http(s):// -- tritonclient wants host:port only
    server_url = TRITON_API_BASE.replace("http://", "").replace("https://", "")
    client = httpclient.InferenceServerClient(url=server_url)

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
    input_ids = httpclient.InferInput("input_ids", [1, seq_len], "INT32")
    input_ids.set_data_from_numpy(np.array([token_ids], dtype=np.int32))

    input_lengths = httpclient.InferInput("input_lengths", [1, 1], "INT32")
    input_lengths.set_data_from_numpy(np.array([[seq_len]], dtype=np.int32))

    request_output_len = httpclient.InferInput("request_output_len", [1, 1], "INT32")
    request_output_len.set_data_from_numpy(np.array([[args.max_tokens]], dtype=np.int32))

    end_id_tensor = httpclient.InferInput("end_id", [1, 1], "INT32")
    end_id_tensor.set_data_from_numpy(np.array([[end_id]], dtype=np.int32))

    pad_id_tensor = httpclient.InferInput("pad_id", [1, 1], "INT32")
    pad_id_tensor.set_data_from_numpy(np.array([[pad_id]], dtype=np.int32))

    # Request outputs
    output_ids_req = httpclient.InferRequestedOutput("output_ids")
    sequence_length_req = httpclient.InferRequestedOutput("sequence_length")

    # Run inference
    result = client.infer(
        model_name=TRITON_MODEL,
        inputs=[input_ids, input_lengths, request_output_len, end_id_tensor, pad_id_tensor],
        outputs=[output_ids_req, sequence_length_req],
    )

    # Parse output
    out_ids = result.as_numpy("output_ids")       # [1, beam, max_seq_len]
    seq_lengths = result.as_numpy("sequence_length")  # [1, beam]
    output_len = int(seq_lengths[0][0])
    generated_ids = out_ids[0][0][:output_len].tolist()

    # Detokenize
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
