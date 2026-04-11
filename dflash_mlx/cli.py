#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import wired_limit

from .history import (
    DEFAULT_HISTORY_PATH,
    append_rows,
    prompt_sha256,
    run_metadata,
)
from .adapters import load_target_model
from .draft import load_draft_model, maybe_quantize_draft_model
from .runtime import dflash_generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLX-first DFlash runner for Apple Silicon."
    )
    parser.add_argument(
        "--target-model",
        default="mlx-community/Qwen3.5-4B-MLX-bf16",
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
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument(
        "--speculative-tokens",
        type=int,
        default=None,
        help=(
            "Number of draft tokens per step. Exact verifier modes clamp this to "
            "the draft model block size; inexact accept-all mode allows larger "
            "oversized blocks for speed experiments."
        ),
    )
    parser.add_argument(
        "--verify-mode",
        choices=[
            "stream",
            "chunked",
            "parallel-replay",
            "parallel-lazy-logits",
            "accept-all",
        ],
        default="parallel-replay",
        help=(
            "Verifier strategy. 'parallel-lazy-logits' keeps exact prefix checks "
            "but computes verifier logits in chunks. 'accept-all' is experimental "
            "and inexact: it trusts the full drafted block instead of checking "
            "the accepted prefix."
        ),
    )
    parser.add_argument("--verify-chunk-size", type=int, default=4)
    parser.add_argument("--draft-quant-bits", type=int, default=None)
    parser.add_argument("--draft-quant-group-size", type=int, default=64)
    parser.add_argument(
        "--draft-attention-mask",
        choices=["auto", "none", "causal"],
        default="auto",
        help=(
            "Attention mask used inside the DFlash drafter. 'auto' uses the "
            "fastest measured exact-safe mask for the target family."
        ),
    )
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument(
        "--history-file",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="CSV file that accumulates benchmark history.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not append this run to the benchmark history CSV.",
    )
    parser.add_argument(
        "--experiment-tag",
        type=str,
        default="",
        help="Optional label for grouping benchmark runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mx.random.seed(args.seed)
    history_meta = (
        {}
        if args.no_history
        else run_metadata("dflash-mlx", experiment_tag=args.experiment_tag)
    )
    prompt_text = args.prompt_file.read_text()

    target = load_target_model(args.target_model)
    print(f"[load target] {target.resolved_model_path}")

    print(f"[load draft] {args.draft_model}")
    draft, draft_path = load_draft_model(args.draft_model)
    draft_attention_mask = args.draft_attention_mask
    if draft_attention_mask == "auto":
        draft_attention_mask = "causal" if target.adapter.family == "qwen3_5" else "none"
    draft.attention_mask_mode = draft_attention_mask
    draft_quantization = maybe_quantize_draft_model(
        draft,
        bits=args.draft_quant_bits,
        group_size=args.draft_quant_group_size,
    )

    prompt_tokens = target.build_prompt(prompt_text)
    stop_token_ids = target.stop_token_ids()
    requested_speculative_tokens = (
        draft.block_size if args.speculative_tokens is None else args.speculative_tokens
    )
    effective_speculative_tokens = (
        max(1, requested_speculative_tokens)
        if args.verify_mode == "accept-all"
        else max(1, min(requested_speculative_tokens, draft.block_size))
    )

    print(
        f"[run] prompt_tokens={prompt_tokens.shape[0]} "
        f"draft_block_size={draft.block_size} "
        f"speculative_tokens={effective_speculative_tokens} "
        f"max_new_tokens={args.max_new_tokens} temperature={args.temperature}"
    )
    if args.verify_mode == "accept-all":
        print(
            "[warning] accept-all is inexact: it trusts drafted blocks and may "
            "diverge from target-model output."
        )
        if effective_speculative_tokens > draft.block_size:
            print(
                "[warning] oversized accept-all block: this draft checkpoint is "
                f"configured for block_size={draft.block_size}, but "
                f"speculative_tokens={effective_speculative_tokens}. This is a "
                "raw throughput probe, not lossless DFlash."
            )
    with wired_limit(target.model):
        for warmup_idx in range(args.warmup_runs):
            _, warm_metrics = dflash_generate(
                target=target,
                draft=draft,
                prompt_tokens=prompt_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                stop_token_ids=stop_token_ids,
                layer_ids=draft.target_layer_ids,
                speculative_tokens=args.speculative_tokens,
                verify_mode=args.verify_mode,
                verify_chunk_size=args.verify_chunk_size,
            )
            print(
                f"[warmup {warmup_idx + 1}/{args.warmup_runs}] "
                f"gen_tps={warm_metrics['generation_tps']:.2f} "
                f"accept={warm_metrics['avg_acceptance_length']:.2f}"
            )

        mx.reset_peak_memory()
        output_tokens, metrics = dflash_generate(
            target=target,
            draft=draft,
            prompt_tokens=prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stop_token_ids=stop_token_ids,
            layer_ids=draft.target_layer_ids,
            speculative_tokens=args.speculative_tokens,
            verify_mode=args.verify_mode,
            verify_chunk_size=args.verify_chunk_size,
        )

    generated_tokens = output_tokens[metrics["num_input_tokens"] :]
    output_text = target.tokenizer.decode(generated_tokens, skip_special_tokens=False)

    print("\n" + "=" * 60)
    print(f"Target model:             {target.resolved_model_path}")
    print(f"Target adapter:           {target.adapter.family}")
    print(f"Draft model:              {draft_path}")
    print(f"Draft attention mask:     {draft_attention_mask}")
    if draft_quantization is not None:
        print(
            "Draft quantization:       "
            f"{draft_quantization.get('bits')}bit g{draft_quantization.get('group_size')}"
        )
    print(f"Speculative tokens:       {metrics['speculative_tokens']}")
    print(f"Verify mode:              {args.verify_mode}")
    if args.verify_mode in {"chunked", "parallel-lazy-logits"}:
        print(f"Verify chunk size:        {args.verify_chunk_size}")
    print(f"Prompt tokens:            {metrics['num_input_tokens']}")
    print(f"Generated tokens:         {metrics['num_output_tokens']}")
    print(f"Prefill time:             {metrics['prefill_time_s']:.2f}s")
    print(f"Decode time:              {metrics['decode_time_s']:.2f}s")
    print(f"Total time:               {metrics['total_time_s']:.2f}s")
    print(f"Prompt TPS:               {metrics['prompt_tps']:.2f}")
    print(f"Generation TPS:           {metrics['generation_tps']:.2f}")
    print(f"End-to-end TPS:           {metrics['end_to_end_tps']:.2f}")
    print(f"Average acceptance:       {metrics['avg_acceptance_length']:.2f}")
    print(f"Acceptance lengths:       {metrics['acceptance_lengths']}")
    print(f"Peak memory:              {metrics['peak_memory_gb']:.2f} GB")
    print("=" * 60)

    if not args.no_history:
        history_row = {
            **history_meta,
            "record_type": "run",
            "target_model": args.target_model,
            "resolved_target_model": str(target.resolved_model_path),
            "target_adapter_family": target.adapter.family,
            "draft_model": args.draft_model,
            "resolved_draft_path": str(draft_path),
            "draft_quant_bits": args.draft_quant_bits,
            "draft_quant_group_size": (
                args.draft_quant_group_size if args.draft_quant_bits is not None else ""
            ),
            "draft_attention_mask": draft_attention_mask,
            "prompt_file": str(args.prompt_file),
            "prompt_sha256": prompt_sha256(prompt_text),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "warmup_runs": args.warmup_runs,
            "verify_mode": args.verify_mode,
            "verify_chunk_size": args.verify_chunk_size,
            "speculative_tokens_arg": args.speculative_tokens,
            **metrics,
        }
        append_rows(args.history_file, [history_row])
        print(f"[history] appended 1 row to {args.history_file}")

    if args.print_output:
        print(output_text)


if __name__ == "__main__":
    main()
