from dflash_mlx import DFlashGenerator


runner = DFlashGenerator(
    target_model="mlx-community/Qwen3.5-4B-MLX-bf16",
    draft_model="z-lab/Qwen3.5-4B-DFlash",
)

result = runner.generate(
    "Write a quicksort in Python.",
    max_new_tokens=128,
    skip_special_tokens=True,
)

print(result.text)
print(f"generation_tps={result.metrics['generation_tps']:.2f}")
