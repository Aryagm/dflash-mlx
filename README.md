# dflash-mlx

**DFlash implementation for Apple Silicon, using MLX.**

![Benchmarks](assets/benchmark-chart.png)

https://github.com/user-attachments/assets/e7a78bca-1a62-42eb-ba75-da32b3b3ad40

## Quick start

```bash
git clone https://github.com/aryagm/dflash-mlx.git && cd dflash-mlx
uv sync

uv run dflash-mlx --max-new-tokens 128
```

Defaults to Qwen3-4B BF16. First run downloads the target and draft checkpoints into the Hugging Face cache, which is roughly 12 GB for the default pair. Pass `--target-model` and `--draft-model` to override. `dflash-mlx-chat` for interactive chat, `--json` for machine-readable output. Benchmark history is opt-in with `--history` or `--history-file`.

```python
from dflash_mlx import DFlashGenerator

# First run also downloads the default Qwen3-4B target and DFlash draft weights.
runner = DFlashGenerator()
result = runner.generate("Write a quicksort in Python.", max_new_tokens=128)
print(result.text)
```

## How it works

[DFlash](https://arxiv.org/abs/2602.06036) trains a small block-diffusion model to propose multiple tokens at once. The target verifies them in a single forward pass and accepts the longest correct prefix &mdash; identical output, fewer forward passes, higher throughput.

The original DFlash targets CUDA. `dflash-mlx` is a native MLX port for Apple Silicon. MLX has no speculative-decoding primitives, so every piece of the draft/verify loop had to be built from scratch on top of Metal:

- **Hidden-state extraction from the target.** DFlash's drafter conditions on intermediate layer activations, not just logits. We hook into specific Qwen layers and surface those tensors without breaking the standard forward path or the KV cache, so a single target pass gives us both verification logits and the hidden states the next draft block needs.

- **Parallel block proposal.** The draft model runs a block-diffusion denoising loop to propose several tokens at once. This runs entirely on the GPU with its own cache, sharing tokenization and positional context with the target.

- **Single-pass batched verification.** Every proposed block is verified in one target forward pass. The target's logits are compared greedily against the draft's samples; we accept the longest matching prefix plus one bonus correction token, which is what makes the output bit-for-bit identical to plain target decoding.

- **Per-layer KV cache rollback on rejection.** When the target rejects the tail of a proposal, the KV cache has to be rewound to the exact accepted length &mdash; per layer, because Qwen3.5 mixes full attention, sliding-window attention, and recurrent linear-attention state, and each has its own cache shape and rollback rule. Plain MLX caches don't expose this; we extend them.

- **Pluggable adapters.** Target-specific concerns (layer ids to tap, cache types, stop tokens, chat template) are isolated in `dflash_mlx/adapters.py`. The core draft/verify loop is architecture-agnostic, so adding a new family is one adapter file rather than a rewrite.

- **Warm-path throughput engineering.** MLX kernel compilation, lazy evaluation, and graph caching all affect the numbers. The bench CLI separates warmup from measurement and pins evaluation points so the reported tok/s reflects steady-state Metal performance, not first-run overhead.

## Supported models

| Target | Draft |
|---|---|
| `mlx-community/Qwen3-4B-bf16` (default) | `z-lab/Qwen3-4B-DFlash-b16` |
| `mlx-community/Qwen3.5-4B-MLX-bf16` | `z-lab/Qwen3.5-4B-DFlash` |

Qwen3.5 support is functional but incomplete. It is not as fast as the Qwen3 path today because Qwen3.5 uses a more complicated hybrid attention stack with recurrent linear-attention state, so exact partial-block acceptance needs custom cache rollback and currently has weaker long-generation acceptance.

Upstream DFlash has checkpoints for Llama 3.1, Qwen3 Coder, Kimi-K2.5, GPT-OSS, and more in the [Hugging Face collection](https://huggingface.co/collections/z-lab/dflash). Adding a new family starts with an adapter in `dflash_mlx/adapters.py` &mdash; see [ADDING_MODELS.md](ADDING_MODELS.md).

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
