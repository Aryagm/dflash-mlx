# dflash-mlx

Lossless DFlash speculative decoding for Apple Silicon. Same target-model output, fewer target forward passes, higher local throughput.

https://github.com/user-attachments/assets/13411079-7ffd-4f3f-a3cd-fdf3dd44a537

## Why It Matters

- **Exact output**: the target model verifies every draft block and accepts only the longest matching prefix.
- **Native Mac path**: the runtime is built on MLX and keeps the draft/verify loop on Apple Silicon.
- **Real speedups**: Qwen3.5-4B hits 100.5 tok/s in bf16 and 161.9 tok/s in 4-bit on an M4 Max 36 GB run.

## Quick Start

```bash
git clone https://github.com/aryagm/dflash-mlx.git
cd dflash-mlx
uv sync

uv run dflash-mlx \
  --target-model mlx-community/Qwen3.5-4B-MLX-4bit \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --max-new-tokens 128
```

For the plain MLX-LM baseline:

```bash
uv run dflash-mlx-bench \
  --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --prompt-file prompts/functional_equation.txt \
  --max-new-tokens 128 \
  --warmup-prompts 0
```

## Benchmarks

**Qwen3.5-4B bf16** on MacBook Pro M4 Max, 36 GB

| | tok/s | vs llama.cpp |
|---|---:|---:|
| llama.cpp | 35.6 | 1.0x |
| MLX-LM | 40.6 | 1.1x |
| **DFlash + MLX** | **100.5** | **2.8x** |

**Qwen3.5-4B 4-bit** on MacBook Pro M4 Max, 36 GB

| | tok/s | vs llama.cpp |
|---|---:|---:|
| llama.cpp (Q4_K_M) | 76.4 | 1.0x |
| MLX-LM | 119.4 | 1.6x |
| **DFlash + MLX** | **161.9** | **2.1x** |

![Benchmarks](assets/benchmark-chart.png)

BF16 is the cleanest exact speedup story. 4-bit is the fastest absolute throughput story. Absolute numbers vary by chip; the speedup ratios are the useful comparison.

See [docs/benchmarks.md](docs/benchmarks.md) for metric definitions, benchmark rules, and why `accept-all` is not lossless.

## Supported Models

| Target Model | Draft Model | Status |
|---|---|---|
| `mlx-community/Qwen3.5-4B-MLX-4bit` | `z-lab/Qwen3.5-4B-DFlash` | Stable |
| `mlx-community/Qwen3.5-4B-MLX-bf16` | `z-lab/Qwen3.5-4B-DFlash` | Stable |
| `mlx-community/Qwen3-4B-{bf16,8bit,4bit}` | `z-lab/Qwen3-4B-DFlash-b16` | Experimental |

Upstream DFlash checkpoints exist for Llama 3.1, Qwen3 Coder, Kimi-K2.5, and more in the [Hugging Face collection](https://huggingface.co/collections/z-lab/dflash). New model families need an adapter in `dflash_mlx/adapters.py`.

## Project Layout

- `dflash_mlx/`: public MLX runtime package and CLI implementations.
- `scripts/`: thin public wrappers and chart generation scripts.
- `docs/`: architecture and benchmark details.
- `tools/`: demo/video/capture utilities, not required for runtime work.
- `experiments/`: older prototypes kept for reference.
- `assets/`: generated charts, captures, and demo videos.

See [docs/architecture.md](docs/architecture.md) for the draft/verify/cache pipeline.

## Contributing

Use `uv` for setup and checks:

```bash
uv sync --extra dev
uv run --extra dev pytest
uv run --extra dev python -m py_compile dflash_mlx/*.py scripts/*.py tools/*.py experiments/*.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for smoke tests, exactness rules, adapter requirements, and benchmark standards.

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
