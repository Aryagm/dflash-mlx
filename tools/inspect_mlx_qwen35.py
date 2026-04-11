#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import qwen3_5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect MLX Qwen3.5 target internals needed for a DFlash port."
    )
    parser.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-bf16")
    parser.add_argument("--prompt-file", type=Path, default=Path("prompts/functional_equation.txt"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--print-prompt-shape", action="store_true")
    return parser.parse_args()


def build_prompt(tokenizer, prompt_text: str) -> mx.array:
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tokens = tokenizer.encode(prompt)
    return mx.array(tokens, dtype=mx.uint32)


def forward_with_hidden_states(model, inputs: mx.array, layer_ids: list[int]):
    text_wrapper = model.language_model
    text_model = text_wrapper.model

    hidden_states = text_model.embed_tokens(inputs)
    cache = text_wrapper.make_cache()

    fa_mask = qwen3_5.create_attention_mask(hidden_states, cache[text_model.fa_idx])
    ssm_mask = qwen3_5.create_ssm_mask(hidden_states, cache[text_model.ssm_idx])

    selected_hidden_states = {}
    for idx, (layer, layer_cache) in enumerate(zip(text_model.layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
        if idx in layer_ids:
            selected_hidden_states[idx] = hidden_states

    hidden_states = text_model.norm(hidden_states)
    if text_wrapper.args.tie_word_embeddings:
        logits = text_model.embed_tokens.as_linear(hidden_states)
    else:
        logits = text_wrapper.lm_head(hidden_states)

    return logits, selected_hidden_states, cache


def cache_summary(cache) -> list[str]:
    rows: list[str] = []
    for idx, layer_cache in enumerate(cache):
        if hasattr(layer_cache, "keys"):
            rows.append(
                f"layer {idx:02d} KVCache size={layer_cache.size()}"
            )
        else:
            conv = None if layer_cache[0] is None else tuple(layer_cache[0].shape)
            recurrent = None if layer_cache[1] is None else tuple(layer_cache[1].shape)
            rows.append(
                f"layer {idx:02d} ArraysCache conv={conv} recurrent={recurrent}"
            )
    return rows


def main() -> None:
    args = parse_args()
    prompt_text = args.prompt_file.read_text()
    target_layer_ids = [1, 8, 15, 22, 29]

    load_start = time.perf_counter()
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": args.trust_remote_code})
    load_elapsed = time.perf_counter() - load_start

    prompt = build_prompt(tokenizer, prompt_text)
    prompt_batch = prompt[None]

    run_start = time.perf_counter()
    logits, selected_hidden_states, cache = forward_with_hidden_states(model, prompt_batch, target_layer_ids)
    mx.eval(logits, *selected_hidden_states.values())
    run_elapsed = time.perf_counter() - run_start

    print(f"model:                    {args.model}")
    print(f"load_time_s:              {load_elapsed:.2f}")
    print(f"prompt_tokens:            {prompt.shape[0]}")
    print(f"prefill_time_s:           {run_elapsed:.2f}")
    print(f"prompt_tps:               {prompt.shape[0] / max(run_elapsed, 1e-9):.2f}")
    print(f"logits_shape:             {tuple(logits.shape)}")
    print("selected_hidden_shapes:")
    for layer_id in target_layer_ids:
        shape = tuple(selected_hidden_states[layer_id].shape)
        print(f"  layer_{layer_id}:             {shape}")
    if args.print_prompt_shape:
        print(f"prompt_array_shape:       {tuple(prompt_batch.shape)}")
    print("cache_summary:")
    for row in cache_summary(cache):
        print(f"  {row}")


if __name__ == "__main__":
    main()
