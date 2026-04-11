#!/usr/bin/env python3
"""Capture real model generations for the demo video."""

import json
import re
import subprocess
import time
from pathlib import Path


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks and trailing special tokens from output."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    # Strip everything after first special token (model finished, rest is echo)
    for marker in ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]:
        if marker in text:
            text = text[:text.index(marker)]
    return text.strip()

PROMPT = (
    "/no_think\n"
    "The function $f$ satisfies the functional equation "
    "f(x) + f(y) = f(x + y) - xy - 1 "
    "for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ "
    "such that $f(n) = n$. Enter all such integers, separated by commas.\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# llama.cpp uses --reasoning off instead of /no_think, so strip it from the prompt
PROMPT_LLAMA = PROMPT.replace("/no_think\n", "")

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "captures"
OUT_DIR.mkdir(exist_ok=True)
MAX_TOKENS = 2048


def capture_llama_cpp():
    """Capture llama.cpp generation."""
    print("=== Capturing llama.cpp ===")

    import glob
    # Explicitly use BF16 GGUF for fair comparison
    gguf_paths = glob.glob(str(Path.home() / ".cache/huggingface/**/*Qwen3.5*4B*BF16*.gguf"), recursive=True)
    if not gguf_paths:
        # Fallback to any GGUF
        gguf_paths = glob.glob(str(Path.home() / ".cache/huggingface/**/*Qwen3.5*4B*.gguf"), recursive=True)

    if not gguf_paths:
        print("  ERROR: No GGUF model found. Skipping.")
        result = {
            "framework": "llama.cpp",
            "text": "",
            "elapsed": 0,
            "tps": 0,
            "max_tokens": MAX_TOKENS,
            "error": "GGUF model not found",
        }
        (OUT_DIR / "llama_cpp.json").write_text(json.dumps(result, indent=2))
        return

    gguf_path = gguf_paths[0]
    print(f"  Using: {gguf_path}")

    start = time.perf_counter()
    proc = subprocess.run(
        [
            "llama-simple",
            "-m", gguf_path,
            "-n", str(MAX_TOKENS),
            "-c", "4096",
            "--temp", "0",
            PROMPT_LLAMA,
        ],
        capture_output=True, text=True, timeout=300,
    )
    elapsed = time.perf_counter() - start

    text = proc.stdout.strip()
    stderr = proc.stderr.strip()
    # llama-simple echoes the prompt, strip it
    if PROMPT_LLAMA in text:
        text = text[text.index(PROMPT_LLAMA) + len(PROMPT_LLAMA):].strip()
    text = strip_think_tags(text)

    # Parse TPS from llama.cpp stderr (looks for "eval time = ... tok/s")
    tps = 0.0
    for line in stderr.split("\n"):
        if "eval time" in line and "tok/s" in line:
            match = re.search(r"([\d.]+)\s*tok/s", line)
            if match:
                tps = float(match.group(1))

    # Fallback: estimate from token count / elapsed
    if tps == 0 and elapsed > 0:
        # Rough estimate: ~4 chars per token
        tps = (len(text) / 4) / elapsed

    print(f"  TPS: {tps:.1f}")
    result = {
        "framework": "llama.cpp",
        "text": text,
        "elapsed": elapsed,
        "tps": tps,
        "max_tokens": MAX_TOKENS,
    }

    out_path = OUT_DIR / "llama_cpp.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved to {out_path} ({len(text)} chars, {elapsed:.1f}s)")


def capture_mlx_plain():
    """Capture plain MLX-LM bf16 generation."""
    print("=== Capturing MLX-LM bf16 ===")
    import mlx.core as mx
    from mlx_lm import load, generate

    model, tokenizer = load("mlx-community/Qwen3.5-4B-MLX-bf16")
    mx.eval(model.parameters())

    # Warmup
    generate(model, tokenizer, prompt="Hello", max_tokens=10)

    start = time.perf_counter()
    response = generate(
        model, tokenizer,
        prompt=PROMPT,
        max_tokens=MAX_TOKENS,
    )
    elapsed = time.perf_counter() - start

    response = strip_think_tags(response)
    # mlx_lm.generate returns text; estimate tokens from tokenizer
    token_count = len(tokenizer.encode(response))
    tps = token_count / elapsed if elapsed > 0 else 0
    print(f"  TPS: {tps:.1f} ({token_count} tokens)")
    result = {
        "framework": "MLX-LM",
        "text": response,
        "elapsed": elapsed,
        "tps": tps,
        "max_tokens": MAX_TOKENS,
    }

    out_path = OUT_DIR / "mlx_bf16.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved to {out_path} ({len(response)} chars, {elapsed:.1f}s)")

    del model, tokenizer


def capture_dflash():
    """Capture DFlash + MLX generation."""
    print("=== Capturing DFlash + MLX bf16 ===")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import mlx.core as mx
    from dflash_mlx.adapters import load_target_model
    from dflash_mlx.draft import load_draft_model
    from dflash_mlx.runtime import dflash_generate

    loaded = load_target_model("mlx-community/Qwen3.5-4B-MLX-bf16")
    target_model = loaded
    tokenizer = loaded.tokenizer
    draft_model, _ = load_draft_model("z-lab/Qwen3.5-4B-DFlash")
    mx.eval(loaded.model.parameters())
    mx.eval(draft_model.parameters())

    stop_ids = target_model.stop_token_ids
    if callable(stop_ids):
        stop_ids = stop_ids()
    stop_token_ids = stop_ids
    layer_ids = draft_model.target_layer_ids
    prompt_tokens = mx.array(tokenizer.encode(PROMPT))

    # Warmup
    _ = dflash_generate(
        target=target_model, draft=draft_model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=10, temperature=0.0,
        stop_token_ids=stop_token_ids, layer_ids=layer_ids,
        speculative_tokens=None, verify_mode="exact", verify_chunk_size=1,
    )

    # Real generation
    prompt_tokens = mx.array(tokenizer.encode(PROMPT))
    start = time.perf_counter()
    output_tokens, stats = dflash_generate(
        target=target_model, draft=draft_model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=MAX_TOKENS, temperature=0.0,
        stop_token_ids=stop_token_ids, layer_ids=layer_ids,
        speculative_tokens=None, verify_mode="exact", verify_chunk_size=1,
    )
    elapsed = time.perf_counter() - start

    generated_tokens = output_tokens[stats["num_input_tokens"]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    text = strip_think_tags(text)
    token_count = len(generated_tokens)
    tps = token_count / elapsed if elapsed > 0 else 0
    print(f"  TPS: {tps:.1f} ({token_count} tokens)")
    result = {
        "framework": "DFlash + MLX",
        "text": text,
        "elapsed": elapsed,
        "tps": tps,
        "max_tokens": MAX_TOKENS,
    }

    out_path = OUT_DIR / "dflash_bf16.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved to {out_path} ({len(text)} chars, {elapsed:.1f}s)")

    del target_model, draft_model, tokenizer


if __name__ == "__main__":
    capture_llama_cpp()
    capture_mlx_plain()
    capture_dflash()
    print("\nAll captures done!")
