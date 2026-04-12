# Qwen3.5 Results Archive

These numbers are kept so the Qwen3.5 path can be revisited later. The current default benchmark target is Qwen3-4B BF16.

Hardware: MacBook Pro M4 Max, 36 GB. Prompt: built-in functional-equation prompt. Decode: exact `parallel-replay`, temperature 0, 128-token warmup, BF16 target `mlx-community/Qwen3.5-4B-MLX-bf16`, draft `z-lab/Qwen3.5-4B-DFlash`.

| Draft mask | Max new tokens | MLX-LM BF16 tok/s | dflash-mlx tok/s | Speedup | Avg acceptance |
|---|---:|---:|---:|---:|---:|
| causal | 512 | 39.15 | 84.75 | 2.17x | 6.10 |
| none | 512 | 39.15 | 93.73 | 2.39x | 6.74 |
| none | 1024 | 39.80 | 76.40 | 1.92x | 5.49 |
| none | 2048 | 39.74 | 73.98 | 1.86x | 5.37 |

Acceptance drift with `draft_attention_mask=none`:

| Max new tokens | First 20 verifier-step avg | Last 20 verifier-step avg |
|---|---:|---:|
| 1024 | 9.40 | 6.55 |
| 2048 | 9.40 | 5.60 |

The main issue was not rollback cost. In the 2048-token profiled run, verifier forward/logits time was about 21.59s of 27.68s decode time, while rollback was about 0.11s.
