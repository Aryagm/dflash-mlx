# DFlash for Apple Silicon
[**Paper**](https://arxiv.org/abs/2602.06036) | [**Blog**](https://z-lab.ai/projects/dflash/) | [**Models**](https://huggingface.co/collections/z-lab/dflash)

This fork brings **exact DFlash inference to Apple Silicon** with an **MLX-native runtime**.

Today, the flagship path is:
- target: `mlx-community/Qwen3.5-4B-MLX-4bit`
- draft: `z-lab/Qwen3.5-4B-DFlash`
- backend: `MLX / Metal`
- mode: exact speculative decoding

On this Mac, that path is the **best exact local result we have measured** for this model:
- `161.9 tok/s` generation
- `140.7 tok/s` end-to-end

## Why This Fork Exists

Upstream DFlash is built around CUDA-first serving and research workflows. This fork adds a **local Apple Silicon runtime** that makes DFlash usable on Mac without treating MLX as an afterthought.

What that means in practice:
- exact local inference on Apple Silicon
- a reusable MLX runtime instead of a one-off script
- adapter-based model support
- benchmark history tracked in CSV

## Current Result

Steady-state numbers on [functional_equation.txt](prompts/functional_equation.txt):

| Path | Generation TPS | End-to-end TPS |
|---|---:|---:|
| Plain MLX BF16 | `40.6` | `37.6` |
| DFlash MLX BF16 | `100.5` | `92.3` |
| Plain MLX 4-bit | `119.4` | `98.8` |
| **DFlash MLX 4-bit** | **`161.9`** | **`140.7`** |

Current uplift:
- about `2.47x` over plain MLX BF16 on decode speed
- about `1.36x` over plain MLX 4-bit on decode speed

Benchmark history lives in [benchmarks/metrics_history.csv](benchmarks/metrics_history.csv).

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install mlx mlx-lm
python3 scripts/run_dflash_mlx.py \
  --target-model mlx-community/Qwen3.5-4B-MLX-4bit \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --max-new-tokens 128 \
  --warmup-runs 1
```

The script reports:
- prompt TPS
- generation TPS
- end-to-end TPS
- average acceptance
- peak memory

## What Works Today

Stable and validated:
- exact `Qwen3.5-4B + z-lab/Qwen3.5-4B-DFlash`
- MLX/Metal runtime on Apple Silicon
- benchmark logging and regression tracking

Important qualifier:
- this is currently a **research-grade Mac runtime**, not a polished serving product

## Runtime Layout

The Apple Silicon path is split into four pieces:

- [scripts/run_dflash_mlx.py](scripts/run_dflash_mlx.py)
  Thin CLI for local benchmarking and generation.
- [scripts/mlx_dflash_runtime.py](scripts/mlx_dflash_runtime.py)
  Generic MLX DFlash loop.
- [scripts/mlx_dflash_adapters.py](scripts/mlx_dflash_adapters.py)
  Family-specific target adapters. The working adapter today is `qwen3_5`.
- [scripts/mlx_dflash_draft.py](scripts/mlx_dflash_draft.py)
  MLX implementation of the DFlash draft model.

`Qwen3.5` also uses:
- [scripts/custom_qwen35_dflash_model.py](scripts/custom_qwen35_dflash_model.py)
- [scripts/prepare_custom_mlx_model.py](scripts/prepare_custom_mlx_model.py)

Those files exist because DFlash on `Qwen3.5` needs hidden-state hooks and correct rollback for its hybrid cache.

## Benchmarking

Plain MLX baseline:

```bash
python3 scripts/benchmark_mlx.py
```

DFlash MLX benchmark:

```bash
python3 scripts/run_dflash_mlx.py \
  --target-model mlx-community/Qwen3.5-4B-MLX-4bit \
  --max-new-tokens 128 \
  --warmup-runs 1
```

Metric definitions:
- `Prompt TPS`: prompt tokens / prefill time
- `Generation TPS`: generated tokens / decode time
- `End-to-end TPS`: generated tokens / total request time after model load

If you care about raw decode speed, use `Generation TPS`. If you care about what a local user actually feels, use `End-to-end TPS`.

## Model Coverage

The MLX runtime is built to support more than one checkpoint family, but the current production-quality Mac path is `Qwen3.5-4B`.

Upstream DFlash drafts already exist for additional families, including:
- Qwen3 / Qwen3.5
- Qwen3 Coder / MoE variants
- Llama 3.1
- gpt-oss
- Kimi-K2.5

See the full collection here:
- [Hugging Face collection](https://huggingface.co/collections/z-lab/dflash)

## Status

What is true right now:
- the MLX runtime is real
- the current path is exact
- the best stable setup is also the fastest stable setup we have found so far

What is not true yet:
- this is not a general-purpose packaged library
- this is not full adapter coverage for every published DFlash checkpoint
- this is not the final performance ceiling for Apple Silicon

## Roadmap

Near-term work:
- add more MLX adapters
- make the runtime easier to reuse outside this repo
- push verifier internals deeper into the model path

## Citation

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```
