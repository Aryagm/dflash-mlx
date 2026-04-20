#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub.utils import disable_progress_bars

from .api import DEFAULT_DRAFT_MODEL, DEFAULT_TARGET_MODEL, DFlashGenerator, DFlashResult
from .history import (
    DEFAULT_HISTORY_PATH,
    append_rows,
    prompt_sha256,
    run_metadata,
)


DEFAULT_PROMPT = (
    "The function $f$ satisfies the functional equation \\[f(x) + f(y) = "
    "f(x + y - xy)\\] for all real numbers $x$ and $y$. If $f(1) = 1$, "
    "then find all integers $n$ such that $f(n) = n$. Enter all your "
    "integers, separated by commas.\n\nPlease reason step by step, and put "
    "your final answer within \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLX-first DFlash runner for Apple Silicon."
    )
    parser.add_argument(
        "--target-model",
        default=DEFAULT_TARGET_MODEL,
        help="MLX target model repo or local path.",
    )
    parser.add_argument(
        "--draft-model",
        default=DEFAULT_DRAFT_MODEL,
        help="Hugging Face repo or local path for the DFlash draft weights.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text to run. Uses the built-in benchmark prompt if omitted.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Read prompt text from a file.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Warmup runs before the measured run. Benchmark rows use warm runs.",
    )
    parser.add_argument(
        "--warmup-max-new-tokens",
        type=int,
        default=None,
        help=(
            "Generated tokens per warmup run. Defaults to --max-new-tokens; set "
            "this lower for long generations so you warm kernels without doing "
            "a full-length warmup."
        ),
    )
    parser.add_argument(
        "--speculative-tokens",
        type=int,
        default=None,
        help=(
            "Number of draft tokens per step. Clamped to the draft model block size."
        ),
    )
    parser.add_argument(
        "--verify-mode",
        choices=[
            "stream",
            "chunked",
            "parallel-replay",
            "parallel-lazy-logits",
            "parallel-greedy-argmax",
        ],
        default="parallel-replay",
        help=(
            "Verifier strategy. All options are exact. 'parallel-greedy-argmax' "
            "only supports temperature=0. 'parallel-lazy-logits' keeps exact "
            "prefix checks but computes verifier logits in chunks."
        ),
    )
    parser.add_argument("--verify-chunk-size", type=int, default=4)
    parser.add_argument("--draft-quant-bits", type=int, default=None)
    parser.add_argument("--draft-quant-group-size", type=int, default=64)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Include coarse decode-phase timings in metrics.",
    )
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
        "--stream",
        action="store_true",
        help="Print generated text as soon as verified tokens are committed.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a single JSON result object and suppress progress logs.",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help=(
            "Append run metrics to this CSV file. If omitted, no history is "
            "written unless --history is passed."
        ),
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Append run metrics to the default benchmark history CSV.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not append this run to benchmark history. This is the default.",
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
    if args.prompt is not None and args.prompt_file is not None:
        raise SystemExit("Use either --prompt or --prompt-file, not both.")
    if args.json and args.stream:
        raise SystemExit("Use either --json or --stream, not both.")
    if args.json:
        disable_progress_bars()
    log = (lambda *items: None) if args.json else print
    history_path = args.history_file or DEFAULT_HISTORY_PATH
    record_history = (
        args.history or args.history_file is not None
    ) and not args.no_history
    history_meta = (
        run_metadata("dflash-mlx", experiment_tag=args.experiment_tag)
        if record_history
        else {}
    )
    prompt_text = (
        args.prompt_file.read_text()
        if args.prompt_file is not None
        else (args.prompt or DEFAULT_PROMPT)
    )

    log(f"[load target] {args.target_model}")
    log(f"[load draft] {args.draft_model}")
    runner = DFlashGenerator(
        target_model=args.target_model,
        draft_model=args.draft_model,
        draft_attention_mask=args.draft_attention_mask,
        draft_quant_bits=args.draft_quant_bits,
        draft_quant_group_size=args.draft_quant_group_size,
        seed=args.seed,
    )
    log(f"[target path] {runner.target_model_path}")
    log(f"[draft path] {runner.draft_path}")

    prompt_tokens = runner.encode_prompt(prompt_text)
    requested_speculative_tokens = (
        runner.draft.block_size
        if args.speculative_tokens is None
        else args.speculative_tokens
    )
    effective_speculative_tokens = max(
        1, min(requested_speculative_tokens, runner.draft.block_size)
    )

    log(
        f"[run] prompt_tokens={prompt_tokens.shape[0]} "
        f"draft_block_size={runner.draft.block_size} "
        f"speculative_tokens={effective_speculative_tokens} "
        f"max_new_tokens={args.max_new_tokens} temperature={args.temperature}"
    )
    if args.warmup_runs == 0:
        log(
            "[note] cold run: first-run compile/prefill overhead is included. "
            "Use --warmup-runs 1 for benchmark-style numbers; add "
            "--warmup-max-new-tokens for long generations."
        )
    warmup_max_new_tokens = (
        args.max_new_tokens
        if args.warmup_max_new_tokens is None
        else args.warmup_max_new_tokens
    )
    if args.warmup_runs > 0:
        log(
            f"[warmup] runs={args.warmup_runs} "
            f"max_new_tokens={warmup_max_new_tokens}"
        )
    for warmup_idx in range(args.warmup_runs):
        warm_result = runner.generate_from_tokens(
            prompt_tokens=prompt_tokens,
            max_new_tokens=warmup_max_new_tokens,
            temperature=args.temperature,
            speculative_tokens=args.speculative_tokens,
            verify_mode=args.verify_mode,
            verify_chunk_size=args.verify_chunk_size,
            reset_peak_memory=False,
        )
        log(
            f"[warmup {warmup_idx + 1}/{args.warmup_runs}] "
            f"gen_tps={warm_result.metrics['generation_tps']:.2f} "
            f"accept={warm_result.metrics['avg_acceptance_length']:.2f}"
        )

    if args.stream:
        final_event = None
        for event in runner.stream_from_tokens(
            prompt_tokens=prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            speculative_tokens=args.speculative_tokens,
            verify_mode=args.verify_mode,
            verify_chunk_size=args.verify_chunk_size,
            profile=args.profile,
        ):
            if event.finished:
                if event.delta:
                    print(event.delta, end="", flush=True)
                final_event = event
            elif event.delta:
                print(event.delta, end="", flush=True)
        print()
        if final_event is None or final_event.metrics is None:
            raise RuntimeError("Streaming generation did not produce a final event.")
        result = DFlashResult(
            text=final_event.text,
            output_tokens=final_event.output_tokens,
            generated_tokens=final_event.generated_tokens,
            metrics=final_event.metrics,
        )
    else:
        result = runner.generate_from_tokens(
            prompt_tokens=prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            speculative_tokens=args.speculative_tokens,
            verify_mode=args.verify_mode,
            verify_chunk_size=args.verify_chunk_size,
            profile=args.profile,
        )
    metrics = result.metrics

    log("\n" + "=" * 60)
    log(f"Target model:             {runner.target_model_path}")
    log(f"Target adapter:           {runner.target.adapter.family}")
    log(f"Draft model:              {runner.draft_path}")
    log(f"Draft attention mask:     {runner.draft_attention_mask}")
    if runner.draft_quantization is not None:
        log(
            "Draft quantization:       "
            f"{runner.draft_quantization.get('bits')}bit "
            f"g{runner.draft_quantization.get('group_size')}"
        )
    log(f"Speculative tokens:       {metrics['speculative_tokens']}")
    log(f"Verify mode:              {args.verify_mode}")
    if args.verify_mode in {"chunked", "parallel-lazy-logits"}:
        log(f"Verify chunk size:        {args.verify_chunk_size}")
    log(f"Prompt tokens:            {metrics['num_input_tokens']}")
    log(f"Generated tokens:         {metrics['num_output_tokens']}")
    log(f"Prefill time:             {metrics['prefill_time_s']:.2f}s")
    log(f"Decode time:              {metrics['decode_time_s']:.2f}s")
    log(f"Total time:               {metrics['total_time_s']:.2f}s")
    log(f"Prompt TPS:               {metrics['prompt_tps']:.2f}")
    log(f"Generation TPS:           {metrics['generation_tps']:.2f}")
    log(f"End-to-end TPS:           {metrics['end_to_end_tps']:.2f}")
    log(f"Average acceptance:       {metrics['avg_acceptance_length']:.2f}")
    log(f"Acceptance lengths:       {metrics['acceptance_lengths']}")
    if args.profile and "profile" in metrics:
        log(f"Profile:                  {metrics['profile']}")
    log(f"Peak memory:              {metrics['peak_memory_gb']:.2f} GB")
    log("=" * 60)

    if record_history:
        history_row = {
            **history_meta,
            "record_type": "run",
            "target_model": args.target_model,
            "resolved_target_model": str(runner.target_model_path),
            "target_adapter_family": runner.target.adapter.family,
            "draft_model": args.draft_model,
            "resolved_draft_path": str(runner.draft_path),
            "draft_quant_bits": args.draft_quant_bits,
            "draft_quant_group_size": (
                args.draft_quant_group_size if args.draft_quant_bits is not None else ""
            ),
            "draft_attention_mask": runner.draft_attention_mask,
            "prompt_file": str(args.prompt_file or ""),
            "prompt_source": "prompt_file" if args.prompt_file is not None else "prompt",
            "prompt_sha256": prompt_sha256(prompt_text),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "warmup_runs": args.warmup_runs,
            "warmup_max_new_tokens": warmup_max_new_tokens,
            "verify_mode": args.verify_mode,
            "verify_chunk_size": args.verify_chunk_size,
            "speculative_tokens_arg": args.speculative_tokens,
            **metrics,
        }
        append_rows(history_path, [history_row])
        log(f"[history] appended 1 row to {history_path}")

    result_payload = {
        "text": result.text,
        "metrics": metrics,
        "target_model": args.target_model,
        "resolved_target_model": str(runner.target_model_path),
        "target_adapter_family": runner.target.adapter.family,
        "draft_model": args.draft_model,
        "resolved_draft_path": str(runner.draft_path),
        "draft_attention_mask": runner.draft_attention_mask,
        "draft_quantization": runner.draft_quantization or {},
        "warmup_max_new_tokens": warmup_max_new_tokens,
        "warmup_runs": args.warmup_runs,
        "verify_mode": args.verify_mode,
    }

    if args.json:
        print(json.dumps(result_payload, indent=2, sort_keys=True))
    elif args.print_output and not args.stream:
        print(result.text)


if __name__ == "__main__":
    main()
