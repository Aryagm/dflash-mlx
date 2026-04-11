# dflash-mlx

Lossless speculative decoding on Apple Silicon. **Same output as the target model, just faster.**

https://github.com/user-attachments/assets/13411079-7ffd-4f3f-a3cd-fdf3dd44a537

**Qwen3.5-4B bf16** on M4 Max, 36 GB &mdash; **2.8x** faster than llama.cpp, **2.5x** faster than MLX-LM

| | tok/s | speedup |
|---|---:|---:|
| llama.cpp | 35.6 | 1.0x |
| MLX-LM | 40.6 | 1.1x |
| **DFlash + MLX** | **100.5** | **2.8x** |

https://github.com/user-attachments/assets/b0b8f4ed-d41d-498e-8d39-475437fef9ff

**Qwen3.5-4B 4-bit** on M4 Max, 36 GB &mdash; **2.1x** faster than llama.cpp, **1.4x** faster than MLX-LM

| | tok/s | speedup |
|---|---:|---:|
| llama.cpp (Q4_K_M) | 76.4 | 1.0x |
| MLX-LM | 119.4 | 1.6x |
| **DFlash + MLX** | **161.9** | **2.1x** |

![Benchmarks](assets/benchmark-chart.png)

> Absolute numbers vary by chip. The speedup ratios are what matter.

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
  --max-new-tokens 128
```

## Supported models

| Target | Draft | Status |
|---|---|---|
| `mlx-community/Qwen3.5-4B-MLX-4bit` | `z-lab/Qwen3.5-4B-DFlash` | Stable |
| `mlx-community/Qwen3.5-4B-MLX-bf16` | `z-lab/Qwen3.5-4B-DFlash` | Stable |
| `mlx-community/Qwen3-4B-{bf16,8bit,4bit}` | `z-lab/Qwen3-4B-DFlash-b16` | Experimental |

Upstream DFlash checkpoints exist for Llama 3.1, Qwen3 Coder, Kimi-K2.5, and more ([HF collection](https://huggingface.co/collections/z-lab/dflash)). Adding a new model family is a single adapter file &mdash; see [Adding models](#adding-new-models) below.

## What we built

MLX has no speculative decoding primitives. Everything below was written from scratch:

- **Draft-then-verify loop** running entirely on Metal: proposal generation, batched verification, token acceptance, and KV cache management in one tight loop
- **Hidden-state extraction** from target model intermediate layers &mdash; DFlash's drafter needs internal representations, not just logits
- **KV cache rollback** when the target rejects a proposed token (Qwen3.5's hybrid sliding-window + global attention needs per-layer rollback logic)
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
