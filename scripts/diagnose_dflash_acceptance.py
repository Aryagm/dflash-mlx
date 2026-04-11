#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm.generate import wired_limit

from dflash_mlx.adapters import load_target_model
from dflash_mlx.draft import load_draft_model
from dflash_mlx.runtime import longest_prefix_match, sample_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose exact acceptance for one DFlash draft block."
    )
    parser.add_argument(
        "--target-model",
        default="mlx-community/Qwen3.5-4B-MLX-4bit",
        help="MLX target model repo or local path.",
    )
    parser.add_argument(
        "--draft-model",
        default="z-lab/Qwen3.5-4B-DFlash",
        help="Hugging Face repo or local path for the DFlash draft weights.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("prompts/functional_equation.txt"),
    )
    parser.add_argument("--speculative-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--draft-attention-mask",
        choices=["auto", "none", "causal"],
        default="auto",
    )
    parser.add_argument("--print-prefix", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mx.random.seed(0)

    prompt_text = args.prompt_file.read_text()
    target = load_target_model(args.target_model)
    draft, draft_path = load_draft_model(args.draft_model)
    draft_attention_mask = args.draft_attention_mask
    if draft_attention_mask == "auto":
        draft_attention_mask = "causal" if target.adapter.family == "qwen3_5" else "none"
    draft.attention_mask_mode = draft_attention_mask

    prompt_tokens = target.build_prompt(prompt_text)
    layer_ids = draft.target_layer_ids
    block_size = max(1, args.speculative_tokens)

    with wired_limit(target.model):
        target_cache = target.make_cache()
        draft_cache = draft.make_cache()

        prefill_start = time.perf_counter()
        logits, target_hidden = target.forward_with_hidden_states(
            prompt_tokens[None],
            target_cache,
            layer_ids,
        )
        first_token_tensor = sample_tokens(logits[:, -1:, :], args.temperature)
        mx.eval(first_token_tensor, target_hidden)
        first_token = int(first_token_tensor.item())
        prefill_time = time.perf_counter() - prefill_start

        block_tokens = [first_token] + [draft.mask_token_id] * (block_size - 1)
        block_input = mx.array(block_tokens, dtype=mx.uint32)[None]
        noise_embedding = target.embed_tokens(block_input)

        draft_start = time.perf_counter()
        draft_hidden = draft(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
        )
        draft_logits = target.lm_head_logits(draft_hidden[:, 1:, :])
        drafted_tokens = sample_tokens(draft_logits, args.temperature)
        mx.eval(drafted_tokens)
        draft_time = time.perf_counter() - draft_start
        block_tokens[1:] = drafted_tokens[0].tolist()

        verify_start = time.perf_counter()
        verifier_logits, verifier_hidden, _ = target.forward_with_hidden_states(
            mx.array(block_tokens, dtype=mx.uint32)[None],
            target_cache,
            layer_ids,
            return_rollback_records=True,
        )
        posterior_tokens = sample_tokens(verifier_logits, args.temperature)
        mx.eval(posterior_tokens, verifier_hidden)
        full_verify_time = time.perf_counter() - verify_start
        posterior = posterior_tokens[0].tolist()
        accepted_inputs = longest_prefix_match(block_tokens[1:], posterior[:-1]) + 1

        unsafe_cache = target.make_cache()
        unsafe_logits, unsafe_hidden = target.forward_with_hidden_states(
            prompt_tokens[None],
            unsafe_cache,
            layer_ids,
        )
        mx.eval(sample_tokens(unsafe_logits[:, -1:, :], args.temperature), unsafe_hidden)
        unsafe_verify_start = time.perf_counter()
        last_logits, last_hidden = target.forward_accept_all_block(
            mx.array(block_tokens, dtype=mx.uint32)[None],
            unsafe_cache,
            layer_ids,
        )
        last_token = sample_tokens(last_logits[:, -1:, :], args.temperature)
        mx.eval(last_token, last_hidden)
        unsafe_verify_time = time.perf_counter() - unsafe_verify_start

    prefix_len = max(0, args.print_prefix)
    print(f"Target model:             {target.resolved_model_path}")
    print(f"Target adapter:           {target.adapter.family}")
    print(f"Draft model:              {draft_path}")
    print(f"Draft configured block:   {draft.block_size}")
    print(f"Requested block:          {block_size}")
    print(f"Draft attention mask:     {draft_attention_mask}")
    print(f"Prompt tokens:            {int(prompt_tokens.shape[0])}")
    print(f"Prefill time:             {prefill_time:.4f}s")
    print(f"Draft block time:         {draft_time:.4f}s")
    print(f"Exact verify time:        {full_verify_time:.4f}s")
    print(f"Accept-all verify time:   {unsafe_verify_time:.4f}s")
    print(f"Exact accepted inputs:    {accepted_inputs}/{block_size}")
    print(f"Exact drafted suffix:     {max(accepted_inputs - 1, 0)}/{block_size - 1}")
    print(
        "Exact one-block TPS:      "
        f"{accepted_inputs / max(draft_time + full_verify_time, 1e-9):.2f}"
    )
    print(
        "Accept-all one-block TPS: "
        f"{block_size / max(draft_time + unsafe_verify_time, 1e-9):.2f}"
    )
    if block_size > draft.block_size:
        print(
            "Oversized note:           requested block exceeds the trained draft "
            "block size; this is expected to fail exact prefix verification."
        )
    if prefix_len:
        print(f"Draft prefix:             {block_tokens[:prefix_len]}")
        print(f"Target posterior prefix:  {posterior[:prefix_len]}")


if __name__ == "__main__":
    main()
