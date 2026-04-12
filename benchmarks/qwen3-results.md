# Qwen3 Benchmark Results

Environment:

- Hardware: MacBook Pro M4 Max, 36 GB
- Prompt: built-in functional-equation prompt
- Sampling: greedy, temp 0
- dflash-mlx mode: `parallel-replay`, exact verification
- Draft model: `z-lab/Qwen3-4B-DFlash-b16`
- MLX warmup: 1 warmup run, 128 max new tokens
- llama.cpp flags: `-c 8192 -ngl all -fa on --temp 0 --top-k 1 --top-p 1 --min-p 0 --seed 0`

## BF16 Long Generation

| Max new tokens | MLX-LM BF16 tok/s | dflash-mlx BF16 tok/s | Speedup | Avg acceptance |
|---:|---:|---:|---:|---:|
| 512 | 42.3 | 133.1 | 3.1x | 8.81 |
| 1024 | 42.0 | 144.6 | 3.4x | 9.66 |
| 2048 | 41.3 | 174.4 | 4.2x | 11.97 |
| 4028 | 40.6 | 186.4 | 4.6x | 13.55 |

## 4028-Token Runtime Comparison

| Target | Runtime | Model | tok/s | vs plain MLX |
|---|---|---|---:|---:|
| BF16 | llama.cpp | `Qwen3-4B-BF16.gguf` | 41.1 | 1.0x |
| BF16 | MLX-LM | `mlx-community/Qwen3-4B-bf16` | 40.6 | 1.0x |
| BF16 | dflash-mlx | `mlx-community/Qwen3-4B-bf16` with dflash-mlx | 186.4 | 4.6x |
| 4-bit / Q4_K_M | llama.cpp | `Qwen3-4B-Q4_K_M.gguf` | 97.8 | 0.9x |
| 4-bit | MLX-LM | `mlx-community/Qwen3-4B-4bit` | 110.5 | 1.0x |
| 4-bit | dflash-mlx | `mlx-community/Qwen3-4B-4bit` with dflash-mlx | 159.2 | 1.4x |

The quantized comparison is runtime-level rather than byte-identical: llama.cpp used Q4_K_M GGUF, while MLX used the MLX 4-bit checkpoint.

## dflash-mlx Profile Notes

| Target | Max new tokens | Avg acceptance | Steps | Draft time | Verify time | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 4028 | 13.55 | 298 | 3.89 s | 17.70 s | 10.00 GB |
| 4-bit | 4028 | 8.92 | 453 | 5.01 s | 20.27 s | 4.42 GB |
