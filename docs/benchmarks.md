# Benchmarks

README benchmark numbers use the functional-equation prompt in `prompts/functional_equation.txt`, temperature 0, and 128 generated tokens on a MacBook Pro M4 Max with 36 GB memory.

The public benchmark table comes from the curated run log in `benchmarks/metrics_history.csv`. Chat-backfilled reconstruction rows are intentionally excluded.

## Metrics

**Generation tok/s** measures output-token throughput after prefill. This is the clearest decode-loop metric because it focuses on the repeated autoregressive or speculative generation phase.

**End-to-end tok/s** measures output tokens divided by total wall time, including prefill and decode. This is more sensitive to prompt length, warmup policy, and model loading choices. Use it when comparing user-visible latency for a fixed workload.

For short prompts, generation tok/s and end-to-end tok/s can diverge because prefill is a larger fraction of total time. For long generations, they usually move closer together.

## BF16 vs 4-bit

BF16 is the cleanest speedup story because the target model and DFlash path are both unquantized. On the current M4 Max benchmark, DFlash + MLX reaches 100.5 tok/s vs 35.6 tok/s for llama.cpp and 40.6 tok/s for MLX-LM.

4-bit is the fastest absolute throughput story. On the same benchmark, DFlash + MLX reaches 161.9 tok/s vs 76.4 tok/s for llama.cpp Q4_K_M and 119.4 tok/s for MLX-LM.

The tradeoff is that quantization changes the target model. Exactness still means exact relative to the quantized target model being used.

## Lossless vs Raw Speed Mode

Default DFlash verifier modes are lossless: the target model verifies the drafted block and accepts only the matching prefix.

`--verify-mode accept-all` is not lossless. It trusts the draft block instead of checking the accepted prefix, so it can diverge from the target model's greedy output. Use it only to probe the hardware ceiling, never for quality or exact-speedup claims.

You can inspect a single draft block with:

```bash
uv run python scripts/diagnose_dflash_acceptance.py \
  --target-model mlx-community/Qwen3.5-4B-MLX-4bit \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --speculative-tokens 512
```

## Reproducing README Numbers

```bash
uv run dflash-mlx \
  --target-model mlx-community/Qwen3.5-4B-MLX-4bit \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --prompt-file prompts/functional_equation.txt \
  --max-new-tokens 128 \
  --temperature 0 \
  --warmup-runs 1
```

```bash
uv run dflash-mlx-bench \
  --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --prompt-file prompts/functional_equation.txt \
  --max-new-tokens 128 \
  --temperature 0 \
  --warmup-prompts 0
```

Results append to `benchmarks/metrics_history.csv` unless `--no-history` is passed. Keep noisy scratch or reconstruction rows out of that file.
