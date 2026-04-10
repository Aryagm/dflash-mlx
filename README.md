# DFlash on Apple Silicon
[**Paper**](https://arxiv.org/abs/2602.06036) | [**Blog**](https://z-lab.ai/projects/dflash/) | [**Models**](https://huggingface.co/collections/z-lab/dflash)

This repository started as the public reference code for **DFlash: Block Diffusion for Flash Speculative Decoding**. It now also contains a working **Apple Silicon / MLX** implementation for local DFlash inference on Mac.

The main addition in this fork is an **MLX-first runtime** that:
- runs DFlash locally on Apple Silicon
- uses a small adapter layer per target-model family
- currently supports an exact `Qwen3.5-4B + z-lab/Qwen3.5-4B-DFlash` path on Mac
- keeps benchmark history in CSV so performance changes are tracked over time

## What Works Today

### Stable local Mac path

The current safe path is:
- target: `mlx-community/Qwen3.5-4B-MLX-4bit`
- draft: `z-lab/Qwen3.5-4B-DFlash`
- backend: `MLX`
- verify mode: `parallel-replay`
- speculative block size: `16`

That path is implemented in:
- [run_dflash_mlx.py](scripts/run_dflash_mlx.py)
- [mlx_dflash_runtime.py](scripts/mlx_dflash_runtime.py)
- [mlx_dflash_adapters.py](scripts/mlx_dflash_adapters.py)
- [mlx_dflash_draft.py](scripts/mlx_dflash_draft.py)

### Current exact result on this Mac

On the benchmark prompt in [functional_equation.txt](prompts/functional_equation.txt), the current exact MLX result is roughly:
- `161.9 tok/s` generation TPS
- `140.7 tok/s` end-to-end TPS

The benchmark history is recorded in:
- [metrics_history.csv](benchmarks/metrics_history.csv)

### Current benchmark snapshot

Steady-state numbers on the same prompt and hardware:

| Path | Generation TPS | End-to-end TPS |
|---|---:|---:|
| Plain MLX BF16 | `40.6` | `37.6` |
| DFlash MLX BF16 | `100.5` | `92.3` |
| Plain MLX 4-bit | `119.4` | `98.8` |
| DFlash MLX 4-bit | `161.9` | `140.7` |

These are the current exact, local results to beat.

## Status

### MLX runtime

Implemented:
- generic MLX DFlash loop
- adapter-based target abstraction
- local draft loading and optional draft quantization
- benchmark logging
- `qwen3_5` target adapter

Verified:
- exact greedy decoding for the current `Qwen3.5-4B` MLX path
- parity preserved after the adapter/runtime refactor

Not yet implemented:
- MLX adapters for `qwen3`, `llama`, `gpt_oss`, `qwen3_next`, `qwen3_5_moe`, `qwen3_moe`, `kimi`
- packaged library API for the MLX runtime
- production serving layer

Experimental but not enabled:
- a more aggressive compiled verifier path exists in the custom Qwen3.5 model fork, but it is disabled because it is not exact over longer runs

## Supported DFlash Draft Checkpoints

Upstream DFlash currently publishes drafts for the following targets:

| Model | DFlash Draft |
|---|---|
| Kimi-K2.5 (Preview) | [z-lab/Kimi-K2.5-DFlash](https://huggingface.co/z-lab/Kimi-K2.5-DFlash) |
| Qwen3.5-4B | [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) |
| Qwen3.5-9B | [z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) |
| Qwen3.5-27B | [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash) |
| Qwen3.5-35B-A3B | [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) |
| Qwen3-Coder-Next | [z-lab/Qwen3-Coder-Next-DFlash](https://huggingface.co/z-lab/Qwen3-Coder-Next-DFlash) |
| Qwen3-Coder-30B-A3B | [z-lab/Qwen3-Coder-30B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3-Coder-30B-A3B-DFlash) |
| gpt-oss-20b | [z-lab/gpt-oss-20b-DFlash](https://huggingface.co/z-lab/gpt-oss-20b-DFlash) |
| gpt-oss-120b | [z-lab/gpt-oss-120b-DFlash](https://huggingface.co/z-lab/gpt-oss-120b-DFlash) |
| Qwen3-4B | [z-lab/Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16) |
| Qwen3-8B | [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) |
| Llama-3.1-8B-Instruct | [z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat](https://huggingface.co/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat) |

The MLX runtime is structured so these map to a small number of **family adapters**, not one implementation per checkpoint.

## Installation

### Base repository

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Apple Silicon / MLX path

The MLX runtime is separate from the CUDA-first upstream paths. Install MLX support explicitly:

```bash
pip install mlx mlx-lm
```

Recommended environment:
- Apple Silicon Mac
- Python `>= 3.10`
- a fresh virtual environment

## Quick Start

### Best current local Mac command

```bash
python3 scripts/run_dflash_mlx.py \
  --target-model mlx-community/Qwen3.5-4B-MLX-4bit \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --max-new-tokens 128 \
  --warmup-runs 1
```

Useful flags:
- `--verify-mode {stream,chunked,parallel-replay}`
- `--speculative-tokens N`
- `--draft-quant-bits {4,8}`
- `--print-output`
- `--experiment-tag TAG`

### Example output

The script reports:
- prompt TPS
- generation TPS
- end-to-end TPS
- average speculative acceptance
- acceptance lengths
- peak memory

## MLX Runtime Architecture

The Apple Silicon path is split into four pieces:

### 1. Draft model

[mlx_dflash_draft.py](scripts/mlx_dflash_draft.py)
- loads DFlash draft checkpoints from Hugging Face or local paths
- implements the MLX draft model
- optionally quantizes the draft model

### 2. Target adapters

[mlx_dflash_adapters.py](scripts/mlx_dflash_adapters.py)
- defines the target-model adapter interface
- contains the current `qwen3_5` adapter
- handles prompt construction, hidden-state extraction, cache rollback, and model resolution

### 3. Generic DFlash runtime

[mlx_dflash_runtime.py](scripts/mlx_dflash_runtime.py)
- implements the draft/verify loop
- does not hard-code Qwen3.5-specific logic
- only depends on the adapter interface

### 4. CLI

[run_dflash_mlx.py](scripts/run_dflash_mlx.py)
- thin runner around the adapter + runtime stack
- prints metrics
- appends benchmark rows to CSV

## Custom Qwen3.5 MLX Fork

The current `qwen3_5` path uses a local custom MLX model fork:
- [custom_qwen35_dflash_model.py](scripts/custom_qwen35_dflash_model.py)
- [prepare_custom_mlx_model.py](scripts/prepare_custom_mlx_model.py)

This is needed because DFlash on `Qwen3.5` requires:
- selected intermediate hidden states during verification
- correct rollback of Qwen3.5 hybrid caches after partial speculative acceptance

The adapter resolves stock `qwen3_5` MLX models into a prepared local custom model directory automatically.

## Benchmarks

### MLX benchmark history

Every benchmark run can append a row to:
- [metrics_history.csv](benchmarks/metrics_history.csv)

The shared CSV helpers are:
- [benchmark_history.py](scripts/benchmark_history.py)
- [backfill_chat_history.py](scripts/backfill_chat_history.py)

### Plain MLX baseline

Use:

```bash
python3 scripts/benchmark_mlx.py
```

### DFlash MLX benchmark

Use:

```bash
python3 scripts/run_dflash_mlx.py \
  --target-model mlx-community/Qwen3.5-4B-MLX-4bit \
  --max-new-tokens 128 \
  --warmup-runs 1
```

### Metric definitions

- `Prompt TPS`: prompt tokens divided by prefill time
- `Generation TPS`: generated tokens divided by decode time
- `End-to-end TPS`: generated tokens divided by total request time after model load

`Generation TPS` is the cleanest decode-speed metric. `End-to-end TPS` is usually the most useful single-request metric.

## Upstream Backends

The original repository also keeps the upstream DFlash paths:
- Transformers
- SGLang
- vLLM

These are still documented by the original project and remain useful on supported non-Mac stacks.

### Transformers

```bash
uv pip install -e .
```

### SGLang

```bash
uv pip install -e ".[sglang]"
```

### vLLM

```bash
uv pip install -e ".[vllm]"
uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

## Roadmap

The MLX-first runtime is now structured to support additional families through adapters. The next adapters are likely:
- `qwen3`
- `llama`
- `gpt_oss`
- `qwen3_next`
- `qwen3_5_moe`
- `qwen3_moe`
- `kimi`

## Acknowledgement

Huge thanks to [@dcw02](https://github.com/dcw02), [@gongy](https://github.com/gongy), and the team at [@modal-labs](https://github.com/modal-labs) for their fast, high-quality support in bringing DFlash to SGLang. And huge thanks as well to [@benchislett](https://github.com/benchislett) at NVIDIA for his work in bringing DFlash to vLLM and helping make it available to the broader serving community.

## Citation

If you find DFlash useful, please cite the paper:

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```
