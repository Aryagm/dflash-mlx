# Contributing

Use `uv` for local setup and checks.

```bash
uv sync --extra dev
uv run --extra dev pytest
uv run --extra dev python -m py_compile dflash_mlx/*.py scripts/*.py tools/*.py experiments/*.py
```

## Smoke Test

Run a short exact decode. This downloads the target and draft model on first use.

```bash
uv run dflash-mlx \
  --target-model mlx-community/Qwen3.5-4B-MLX-4bit \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --max-new-tokens 32 \
  --warmup-runs 0
```

The plain MLX-LM baseline is:

```bash
uv run dflash-mlx-bench \
  --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --prompt-file prompts/functional_equation.txt \
  --max-new-tokens 32 \
  --warmup-prompts 0
```

## Exactness

Default verifier modes are lossless: the target model verifies a drafted block and accepts only the longest prefix whose next-token predictions match the target. Output must match the target model's greedy output for the same prompt and sampling settings.

`--verify-mode accept-all` is not lossless. It is a raw speed probe that trusts drafted tokens instead of checking the accepted prefix, so do not use it for quality, benchmark, or correctness claims.

## Adding A Model Adapter

Add model-family support in `dflash_mlx/adapters.py`.

An adapter must define:

- Prompt formatting and stop token extraction.
- Target embedding and LM-head access.
- Hidden-state extraction for the target layers expected by the DFlash draft.
- KV and linear-cache rollback behavior for rejected draft suffixes.
- Cache summaries that make benchmark logs debuggable.

Register the adapter in `ADAPTERS`, then add a no-download unit test for `adapter_for_model_type`.

## Benchmark Standards

Use the same prompt, output length, precision, and warmup policy when comparing methods. Report generation tok/s and end-to-end tok/s separately when possible.

For README numbers, use the functional-equation prompt in `prompts/functional_equation.txt`, 128 generated tokens, temperature 0, and Mac hardware details. Do not commit raw `benchmarks/metrics_history.csv`; it is a local run log. Promote clean public numbers into `benchmarks/summary.csv`.
