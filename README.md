# dflash-mlx

**Qwen3 on Apple Silicon, 4.6× faster.** Bit-for-bit identical output to the target model.

![Benchmarks](assets/benchmark-chart.png)

https://github.com/user-attachments/assets/e7a78bca-1a62-42eb-ba75-da32b3b3ad40

## Quick start

```bash
git clone https://github.com/aryagm/dflash-mlx.git && cd dflash-mlx
uv sync

uv run dflash-mlx --max-new-tokens 128
```

Defaults to Qwen3-4B BF16. Pass `--target-model` and `--draft-model` to override. `dflash-mlx-chat` for interactive chat, `--json` for machine-readable output.

```python
from dflash_mlx import DFlashGenerator

runner = DFlashGenerator()
result = runner.generate("Write a quicksort in Python.", max_new_tokens=128)
print(result.text)
```

## How it works

[DFlash](https://arxiv.org/abs/2602.06036) trains a small block-diffusion model to propose multiple tokens at once. The target verifies them in a single forward pass and accepts the longest correct prefix &mdash; identical output, fewer forward passes, higher throughput.

The original DFlash targets CUDA. `dflash-mlx` is a native MLX port for Apple Silicon: target hidden-state extraction, draft block proposal, batched verification, and per-layer KV cache rollback, all running on Metal.

## Supported models

| Target | Draft |
|---|---|
| Qwen3-4B (default) | `z-lab/Qwen3-4B-DFlash-b16` |
| Qwen3.5-4B | `z-lab/Qwen3.5-4B-DFlash` |

Upstream DFlash has checkpoints for Llama 3.1, Qwen3 Coder, Kimi-K2.5, GPT-OSS, and more in the [Hugging Face collection](https://huggingface.co/collections/z-lab/dflash). Adding a new family is a single adapter in `dflash_mlx/adapters.py` &mdash; see [ADDING_MODELS.md](ADDING_MODELS.md).

## Benchmarks

Full run details, acceptance stats, and quantized comparisons:
- [benchmarks/qwen3-results.md](benchmarks/qwen3-results.md) &mdash; headline Qwen3 results
- [benchmarks/qwen35-results.md](benchmarks/qwen35-results.md) &mdash; archived Qwen3.5 runs

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
