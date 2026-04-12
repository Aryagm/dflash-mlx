# dflash-mlx

Lossless speculative decoding on Apple Silicon. **Same output as the target model, faster when the draft is accepted.**

![Benchmarks](assets/benchmark-chart.png)

https://github.com/user-attachments/assets/e7a78bca-1a62-42eb-ba75-da32b3b3ad40

Logged warm Qwen3-4B long-generation benchmark on MacBook Pro M4 Max, 36 GB:

| Max new tokens | MLX-LM BF16 tok/s | dflash-mlx BF16 tok/s | vs MLX-LM | Avg acceptance |
|---:|---:|---:|---:|---:|
| 512 | 42.3 | 133.1 | 3.1x | 8.81 |
| 1024 | 42.0 | 144.6 | 3.4x | 9.66 |
| 2048 | 41.3 | 174.4 | 4.2x | 11.97 |
| 4028 | 40.6 | 186.4 | 4.6x | 13.55 |

4028-token runtime comparison on the same prompt:

| Target | Runtime | tok/s | vs plain MLX | Detail |
|---|---|---:|---:|---|
| BF16 | llama.cpp | 41.1 | 1.0x | `Qwen3-4B-BF16.gguf`, `-ngl all -fa on` |
| BF16 | MLX-LM | 40.6 | 1.0x | `mlx-community/Qwen3-4B-bf16` |
| BF16 | dflash-mlx | 186.4 | 4.6x | Avg acceptance 13.55 |
| 4-bit / Q4_K_M | llama.cpp | 97.8 | 0.9x | `Qwen3-4B-Q4_K_M.gguf`, `-ngl all -fa on` |
| 4-bit | MLX-LM | 110.5 | 1.0x | `mlx-community/Qwen3-4B-4bit` |
| 4-bit | dflash-mlx | 159.2 | 1.4x | Avg acceptance 8.92 |

> These are single-prompt warm generation tok/s numbers on the built-in functional-equation prompt. The 4-bit llama.cpp row uses Q4_K_M GGUF, while the MLX rows use the MLX 4-bit checkpoint, so that quantized comparison is runtime-level rather than byte-identical weights. Cold first runs include MLX compilation overhead, and long continuations depend on acceptance length, so benchmark your exact workload.

Detailed run notes are saved in [benchmarks/qwen3-results.md](benchmarks/qwen3-results.md).

## How it works

[DFlash](https://arxiv.org/abs/2602.06036) trains a small block-diffusion model to propose multiple tokens at once. The target model verifies them in a single forward pass and accepts the longest correct prefix. You get identical output, fewer forward passes, higher throughput.

The original DFlash targets CUDA. This repo is a native MLX port for Apple Silicon.

## Quick start

```bash
git clone https://github.com/aryagm/dflash-mlx.git && cd dflash-mlx
uv sync

uv run dflash-mlx \
  --target-model mlx-community/Qwen3-4B-bf16 \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --no-history
```

If you omit `--warmup-runs`, the first run is a cold smoke test and will be lower than the chart because MLX kernel compilation is included. For long generations, warm the kernels without doing a full-length warmup:

```bash
uv run dflash-mlx \
  --target-model mlx-community/Qwen3-4B-bf16 \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --max-new-tokens 4096 \
  --warmup-runs 1 \
  --warmup-max-new-tokens 128
```

Machine-readable output:

```bash
uv run dflash-mlx \
  --target-model mlx-community/Qwen3-4B-bf16 \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompt "Write a quicksort in Python." \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --json \
  --no-history
```

Interactive chat:

```bash
uv run dflash-mlx-chat
```

Check model support before loading full weights:

```bash
uv run dflash-mlx-inspect \
  --target-model mlx-community/Qwen3-4B-bf16 \
  --draft-model z-lab/Qwen3-4B-DFlash-b16
```

Python API:

```python
from dflash_mlx import DFlashGenerator

runner = DFlashGenerator()
result = runner.generate("Write a quicksort in Python.", max_new_tokens=128)
print(result.text)
```

See [examples/python_api.py](examples/python_api.py) for a minimal script.

## Supported models

Today this repo is focused on Qwen3-4B BF16, which is the closest supported path to the DFlash paper's main Qwen3 long-generation setup. Qwen3.5 remains supported and archived for follow-up work.

| Target | Draft | Status |
|---|---|---|
| `mlx-community/Qwen3-4B-bf16` | `z-lab/Qwen3-4B-DFlash-b16` | Supported, headline path |
| `mlx-community/Qwen3-4B-4bit` | `z-lab/Qwen3-4B-DFlash-b16` | Supported, measured quantized target |
| `mlx-community/Qwen3-4B-8bit` | `z-lab/Qwen3-4B-DFlash-b16` | Supported, quantized target |
| `mlx-community/Qwen3.5-4B-MLX-bf16` | `z-lab/Qwen3.5-4B-DFlash` | Supported, archived optimization path |
| `mlx-community/Qwen3.5-4B-MLX-4bit` | `z-lab/Qwen3.5-4B-DFlash` | Supported, archived quantized target |

Qwen3.5 results are saved in [benchmarks/qwen35-results.md](benchmarks/qwen35-results.md). Its hybrid attention plus linear-attention cache path is still useful for adapter work, but current long-generation acceptance is weaker than Qwen3 on the built-in prompt.

Upstream DFlash checkpoints exist for Llama 3.1, Qwen3 Coder, Kimi-K2.5, GPT-OSS, and more ([HF collection](https://huggingface.co/collections/z-lab/dflash)). Adding a new family starts with an adapter and may need a custom MLX model shim if cache rollback is architecture-specific; see [ADDING_MODELS.md](ADDING_MODELS.md).

## What we built

MLX has no speculative decoding primitives. Everything below was written from scratch:

- **Draft-then-verify loop** running entirely on Metal: proposal generation, batched verification, token acceptance, and KV cache management in one tight loop
- **Hidden-state extraction** from target model intermediate layers &mdash; DFlash's drafter needs internal representations, not just logits
- **Cache rollback** when the target rejects a proposed token (Qwen3.5's hybrid attention + linear-attention state needs per-layer rollback logic)
- **Pluggable model adapters** so adding a new architecture doesn't touch the core decode loop
- **Reproducible chart generation** via `uv run --extra charts python scripts/generate_benchmark_chart.py`

## Adding new models

Each model family needs an adapter in `dflash_mlx/adapters.py`. The Qwen3 adapter is the default path; Qwen3.5 is the reference for custom recurrent cache rollback. See [ADDING_MODELS.md](ADDING_MODELS.md) for the full checklist.

## Citation

```bibtex
@article{chen2026dflash,
  title   = {DFlash: Block Diffusion for Flash Speculative Decoding},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```

## License

MIT
