# dflash-mlx

Lossless speculative decoding on Apple Silicon. Output is identical to the target model, just faster.

https://github.com/user-attachments/assets/13411079-7ffd-4f3f-a3cd-fdf3dd44a537

**Qwen3.5-4B bf16** on MacBook Pro M4 Max, 36 GB

| | tok/s | vs llama.cpp |
|---|---:|---:|
| llama.cpp | 35.6 | 1.0x |
| MLX-LM | 40.6 | 1.1x |
| **DFlash + MLX** | **100.5** | **2.8x** |

https://github.com/user-attachments/assets/b0b8f4ed-d41d-498e-8d39-475437fef9ff

**Qwen3.5-4B 4-bit** on MacBook Pro M4 Max, 36 GB

| | tok/s | vs llama.cpp |
|---|---:|---:|
| llama.cpp (Q4_K_M) | 76.4 | 1.0x |
| MLX-LM | 119.4 | 1.6x |
| **DFlash + MLX** | **161.9** | **2.1x** |

![Benchmarks](assets/benchmark-chart.png)

> Absolute numbers vary by chip. The speedup ratios are what matter.

## Quick Start

```bash
git clone https://github.com/aryagm/dflash-mlx.git
cd dflash-mlx
python3 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install mlx mlx-lm

python3 scripts/run_dflash_mlx.py \
  --target-model mlx-community/Qwen3.5-4B-MLX-4bit \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --max-new-tokens 128
```

## How It Works

[DFlash](https://arxiv.org/abs/2602.06036) trains a small block-diffusion model to propose multiple tokens at once. The target model verifies them in a single forward pass and accepts the longest correct prefix. You get the exact same output, fewer forward passes, higher throughput.

The original DFlash targets CUDA. This project is a native MLX port for Apple Silicon.

## What We Had to Build

MLX has no speculative decoding primitives. Everything below had to be written from scratch:

- **Draft-then-verify loop** in pure MLX, running entirely on the Metal GPU: proposal generation, batched verification, token acceptance, and KV cache management in one tight loop.
- **Hidden state extraction** from the target model's intermediate layers. DFlash's drafter needs internal representations, not just logits. We hook into the forward pass without breaking the inference path or cache.
- **KV cache rollback** when the target rejects a proposed token. Qwen3.5 uses hybrid sliding-window + global attention, so each layer type needs different rollback logic.
- **Pluggable model adapters** so adding a new architecture (Llama, Qwen, etc.) doesn't require touching the core decode loop.

## Supported Models

| Target Model | Draft Model | Status |
|---|---|---|
| `mlx-community/Qwen3.5-4B-MLX-4bit` | `z-lab/Qwen3.5-4B-DFlash` | Stable |
| `mlx-community/Qwen3.5-4B-MLX-bf16` | `z-lab/Qwen3.5-4B-DFlash` | Stable |
| `mlx-community/Qwen3-4B-{bf16,8bit,4bit}` | `z-lab/Qwen3-4B-DFlash-b16` | Experimental |

Upstream DFlash checkpoints exist for Llama 3.1, Qwen3 Coder, Kimi-K2.5, and more ([HuggingFace collection](https://huggingface.co/collections/z-lab/dflash)). Adding a new model family is a single adapter file.

## Benchmarking

```bash
# DFlash
python3 scripts/run_dflash_mlx.py \
  --target-model mlx-community/Qwen3.5-4B-MLX-4bit \
  --max-new-tokens 128 --warmup-runs 1

# Plain MLX baseline
python3 scripts/benchmark_mlx.py
```

Results are logged to [`benchmarks/metrics_history.csv`](benchmarks/metrics_history.csv) with full reproducibility metadata.

## Roadmap

- [ ] More model families (Llama 3.1, Qwen3-Coder)
- [ ] Streaming API: yield tokens as they're accepted
- [ ] Python library interface: importable `dflash.generate()`
- [ ] Metal kernel optimizations for the verify step
- [ ] M1/M2/M3/M4 Pro/Max/Ultra benchmark matrix

## Contributing

1. **New model adapters**: each family needs an adapter in `scripts/mlx_dflash_adapters.py`. The Qwen3.5 adapter is a good reference.
2. **Benchmark results**: run on your Mac and open a PR.
3. **Bug reports**: issues with specific hardware or model configs.

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
