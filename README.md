# dflash-mlx

Lossless speculative decoding on Apple Silicon. **Same output as the target model, faster when the draft is accepted.**

![Benchmarks](assets/benchmark-chart.png)

https://github.com/user-attachments/assets/13411079-7ffd-4f3f-a3cd-fdf3dd44a537

Warm 128-token Qwen3.5-4B benchmark on MacBook Pro M4 Max, 36 GB:

| | tok/s | vs llama.cpp |
|---|---:|---:|
| **bf16** | | |
| llama.cpp | 35.6 | 1.0x |
| MLX-LM | 40.6 | 1.1x |
| **DFlash + MLX** | **100.5** | **2.8x** |
| **4-bit** | | |
| llama.cpp (Q4_K_M) | 76.4 | 1.0x |
| MLX-LM | 119.4 | 1.6x |
| **DFlash + MLX** | **161.9** | **2.1x** |

> These are warm generation tok/s numbers on the built-in short prompt. Cold first runs include MLX compilation overhead, and long continuations depend on acceptance length, so benchmark your exact workload.

## How it works

[DFlash](https://arxiv.org/abs/2602.06036) trains a small block-diffusion model to propose multiple tokens at once. The target model verifies them in a single forward pass and accepts the longest correct prefix. You get identical output, fewer forward passes, higher throughput.

The original DFlash targets CUDA. This repo is a native MLX port for Apple Silicon.

## Quick start

```bash
git clone https://github.com/aryagm/dflash-mlx.git && cd dflash-mlx
uv sync

uv run dflash-mlx \
  --target-model mlx-community/Qwen3.5-4B-MLX-4bit \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --no-history
```

If you omit `--warmup-runs`, the first run is a cold smoke test and will be lower than the chart because MLX kernel compilation is included. For long generations, warm the kernels without doing a full-length warmup:

```bash
uv run dflash-mlx \
  --target-model mlx-community/Qwen3.5-4B-MLX-4bit \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --max-new-tokens 4096 \
  --warmup-runs 1 \
  --warmup-max-new-tokens 128
```

Machine-readable output:

```bash
uv run dflash-mlx \
  --target-model mlx-community/Qwen3.5-4B-MLX-4bit \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
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
  --target-model mlx-community/Qwen3.5-4B-MLX-4bit \
  --draft-model z-lab/Qwen3.5-4B-DFlash
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

Today this repo is focused on Qwen3.5-4B. Other upstream DFlash checkpoints need MLX target adapters before they can be exact on Mac.

| Target | Draft | Status |
|---|---|---|
| `mlx-community/Qwen3.5-4B-MLX-4bit` | `z-lab/Qwen3.5-4B-DFlash` | Supported |
| `mlx-community/Qwen3.5-4B-MLX-bf16` | `z-lab/Qwen3.5-4B-DFlash` | Supported |
| `mlx-community/Qwen3-4B-{bf16,8bit,4bit}` | `z-lab/Qwen3-4B-DFlash-b16` | Experimental adapter |

Upstream DFlash checkpoints exist for Llama 3.1, Qwen3 Coder, Kimi-K2.5, GPT-OSS, and more ([HF collection](https://huggingface.co/collections/z-lab/dflash)). Adding a new family starts with an adapter and may need a custom MLX model shim if cache rollback is architecture-specific; see [ADDING_MODELS.md](ADDING_MODELS.md).

## What we built

MLX has no speculative decoding primitives. Everything below was written from scratch:

- **Draft-then-verify loop** running entirely on Metal: proposal generation, batched verification, token acceptance, and KV cache management in one tight loop
- **Hidden-state extraction** from target model intermediate layers &mdash; DFlash's drafter needs internal representations, not just logits
- **Cache rollback** when the target rejects a proposed token (Qwen3.5's hybrid attention + linear-attention state needs per-layer rollback logic)
- **Pluggable model adapters** so adding a new architecture doesn't touch the core decode loop

## Adding new models

Each model family needs an adapter in `dflash_mlx/adapters.py`. The Qwen3.5 adapter is the reference implementation. See [ADDING_MODELS.md](ADDING_MODELS.md) for the full checklist.

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
