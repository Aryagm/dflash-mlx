#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

import mlx.core as mx
from datasets import load_dataset
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "cache"

DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(
            **x
        ),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(
            **x
        ),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```".format(
            **x
        ),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: x["prompt"],
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: x["prompt"],
        "multi_turn": True,
    },
}


@dataclass
class PromptResult:
    prompt_tokens: int
    output_tokens: int
    prompt_tps: float
    generation_tps: float
    wall_time_s: float
    peak_memory_gb: float
    finish_reason: str
    output_text: str


def _prepare_dataset(name: str) -> Path:
    cfg = DATASETS[name]
    CACHE_DIR.mkdir(exist_ok=True)
    out_path = CACHE_DIR / f"{name}.jsonl"

    print(f"[download] {name} ...")
    dataset = load_dataset(*cfg["load_args"], **cfg["load_kwargs"])

    with out_path.open("w") as f:
        for row in dataset:
            turns = cfg["format"](row) if cfg.get("multi_turn") else [cfg["format"](row)]
            f.write(json.dumps({"turns": turns}) + "\n")

    sample_count = sum(1 for _ in out_path.open())
    print(f"[cached] {out_path} ({sample_count} samples)")
    return out_path


def load_and_process_dataset(name: str) -> list[dict]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}")

    path = CACHE_DIR / f"{name}.jsonl"
    if not path.exists():
        _prepare_dataset(name)

    with path.open() as f:
        return [json.loads(line) for line in f]


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt is not None:
        return [args.prompt]
    if args.prompt_file is not None:
        return [Path(args.prompt_file).read_text()]

    dataset = load_and_process_dataset(args.dataset)
    prompts = [dataset[i % len(dataset)]["turns"][0] for i in range(args.num_prompts + args.warmup_prompts)]

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(prompts)

    return prompts


def build_prompt_tokens(tokenizer, user_prompt: str, enable_thinking: bool) -> list[int]:
    if getattr(tokenizer, "has_chat_template", False):
        messages = [{"role": "user", "content": user_prompt}]
        kwargs = {"enable_thinking": enable_thinking}
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return tokenizer.encode(prompt_text, add_special_tokens=False)

    return tokenizer.encode(user_prompt)


def build_sampler(tokenizer, args: argparse.Namespace) -> Callable:
    newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
    return make_sampler(
        temp=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        min_tokens_to_keep=args.min_tokens_to_keep,
        top_k=args.top_k,
        xtc_probability=0.0,
        xtc_threshold=0.0,
        xtc_special_tokens=list(newline_tokens) + list(tokenizer.eos_token_ids),
    )


def run_one_prompt(model, tokenizer, prompt_tokens: list[int], args: argparse.Namespace) -> PromptResult:
    sampler = build_sampler(tokenizer, args)
    mx.reset_peak_memory()

    output_chunks: list[str] = []
    final_response = None
    wall_start = time.perf_counter()

    for response in stream_generate(
        model,
        tokenizer,
        prompt_tokens,
        max_tokens=args.max_new_tokens,
        sampler=sampler,
    ):
        output_chunks.append(response.text)
        final_response = response

    wall_time_s = time.perf_counter() - wall_start
    if final_response is None:
        raise RuntimeError("No response was produced by MLX-LM generation.")

    return PromptResult(
        prompt_tokens=final_response.prompt_tokens,
        output_tokens=final_response.generation_tokens,
        prompt_tps=final_response.prompt_tps,
        generation_tps=final_response.generation_tps,
        wall_time_s=wall_time_s,
        peak_memory_gb=final_response.peak_memory,
        finish_reason=final_response.finish_reason or "unknown",
        output_text="".join(output_chunks),
    )


def summarize(results: list[PromptResult]) -> None:
    if not results:
        raise ValueError("No benchmark results to summarize.")

    prompt_time_s = sum(r.prompt_tokens / r.prompt_tps for r in results)
    generation_time_s = sum(r.output_tokens / r.generation_tps for r in results)
    total_time_s = sum(r.wall_time_s for r in results)
    total_prompt_tokens = sum(r.prompt_tokens for r in results)
    total_output_tokens = sum(r.output_tokens for r in results)

    aggregate_prompt_tps = total_prompt_tokens / max(prompt_time_s, 1e-9)
    aggregate_generation_tps = total_output_tokens / max(generation_time_s, 1e-9)
    end_to_end_output_tps = total_output_tokens / max(total_time_s, 1e-9)

    print("\n" + "=" * 60)
    print(f"Prompts benchmarked:      {len(results)}")
    print(f"Prompt tokens:            {total_prompt_tokens}")
    print(f"Generated tokens:         {total_output_tokens}")
    print(f"Aggregate prompt TPS:     {aggregate_prompt_tps:,.2f}")
    print(f"Aggregate generation TPS: {aggregate_generation_tps:,.2f}")
    print(f"End-to-end output TPS:    {end_to_end_output_tps:,.2f}")
    print(f"Mean prompt TPS:          {mean(r.prompt_tps for r in results):,.2f}")
    print(f"Mean generation TPS:      {mean(r.generation_tps for r in results):,.2f}")
    print(f"Mean wall time:           {mean(r.wall_time_s for r in results):.2f}s")
    print(f"Peak memory (max):        {max(r.peak_memory_gb for r in results):.2f} GB")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark plain MLX-LM generation on Apple Silicon.")
    parser.add_argument("--model", required=True, help="MLX model repo or local path.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="gsm8k",
        help="Dataset to benchmark when no explicit prompt is provided.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Run a single benchmark prompt.")
    parser.add_argument("--prompt-file", type=Path, default=None, help="Read a single benchmark prompt from a file.")
    parser.add_argument("--num-prompts", type=int, default=1, help="Number of prompts to benchmark.")
    parser.add_argument("--warmup-prompts", type=int, default=1, help="Warmup prompts to run before measuring.")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum generated tokens per prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling.")
    parser.add_argument("--min-p", type=float, default=0.0, help="Min-p sampling.")
    parser.add_argument("--min-tokens-to-keep", type=int, default=1, help="Min tokens to keep for min-p sampling.")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking in chat templates that support it.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for prompt shuffling.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle prompts before warmup/benchmark.")
    parser.add_argument("--print-output", action="store_true", help="Print the generated output for each measured prompt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    prompts = load_prompts(args)
    print(f"[load] {args.model}")
    model, tokenizer = load(args.model)

    warmup = min(args.warmup_prompts, len(prompts) if args.prompt is None and args.prompt_file is None else 0)
    warmup_prompts = prompts[:warmup]
    benchmark_prompts = prompts[warmup:] if warmup else prompts

    if not benchmark_prompts:
        raise ValueError("No prompts left to benchmark after warmup.")

    for index, prompt in enumerate(warmup_prompts, start=1):
        prompt_tokens = build_prompt_tokens(tokenizer, prompt, args.enable_thinking)
        result = run_one_prompt(model, tokenizer, prompt_tokens, args)
        print(
            f"[warmup {index}/{len(warmup_prompts)}] "
            f"prompt={result.prompt_tokens} out={result.output_tokens} "
            f"gen_tps={result.generation_tps:,.2f}"
        )
        mx.clear_cache()

    results: list[PromptResult] = []
    for index, prompt in enumerate(benchmark_prompts, start=1):
        prompt_tokens = build_prompt_tokens(tokenizer, prompt, args.enable_thinking)
        result = run_one_prompt(model, tokenizer, prompt_tokens, args)
        results.append(result)
        print(
            f"[run {index}/{len(benchmark_prompts)}] "
            f"prompt={result.prompt_tokens} out={result.output_tokens} "
            f"prompt_tps={result.prompt_tps:,.2f} gen_tps={result.generation_tps:,.2f} "
            f"wall={result.wall_time_s:.2f}s peak_mem={result.peak_memory_gb:.2f}GB "
            f"finish={result.finish_reason}"
        )
        if args.print_output:
            print(result.output_text)
        mx.clear_cache()

    summarize(results)


if __name__ == "__main__":
    main()
