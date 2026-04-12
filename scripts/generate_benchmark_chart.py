#!/usr/bin/env python3
"""Generate the benchmark chart PNG for the README."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# Single-prompt 4,028-token generation throughput, MacBook Pro M4 Max, 36 GB.
# Exact DFlash: Qwen3-4B BF16 + z-lab/Qwen3-4B-DFlash-b16.
entries = [
    ("DFlash + MLX bf16", 186.4, "#5f8ff0"),
    ("MLX-LM 4-bit", 110.5, "#454b68"),
    ("llama.cpp Q4_K_M", 97.8, "#454b68"),
    ("llama.cpp bf16", 41.1, "#454b68"),
    ("MLX-LM bf16", 40.6, "#454b68"),
]
entries = sorted(entries, key=lambda entry: entry[1], reverse=True)

labels = [entry[0] for entry in entries]
values = [entry[1] for entry in entries]
colors = [entry[2] for entry in entries]

BG = "#0d1117"

mpl.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
})

fig, ax = plt.subplots(figsize=(8.2, 4.2))
fig.subplots_adjust(left=0.28, right=0.90, top=0.78, bottom=0.12)

bars = ax.barh(range(len(entries)), values, height=0.54, color=colors, edgecolor="none")

for i, (bar, val) in enumerate(zip(bars, values)):
    is_ours = "DFlash" in labels[i]
    if is_ours:
        col = "#ffffff"
        weight = "bold"
    else:
        col = "#8b95a5"
        weight = "normal"
    ax.text(
        val + 2.5,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}",
        va="center",
        fontsize=10.5,
        fontweight=weight,
        color=col,
        fontfamily="monospace",
    )

ax.set_yticks(range(len(entries)))
tick_labels = ax.set_yticklabels(labels, fontsize=10.0, color="#c0c8d4")
for tick, label in zip(tick_labels, labels):
    if "DFlash" in label:
        tick.set_color("#ffffff")
        tick.set_fontweight("bold")
ax.invert_yaxis()
ax.set_xlim(0, 205)
ax.xaxis.set_visible(False)
ax.spines[:].set_visible(False)
ax.tick_params(left=False, bottom=False)

fig.text(
    0.5,
    0.95,
    "dflash-mlx \u2014 throughput comparison",
    fontsize=14,
    fontweight="bold",
    color="#e6eaf0",
    ha="center",
)
fig.text(
    0.5,
    0.88,
    "Qwen3-4B BF16  \u00b7  4,028-token generation  \u00b7  MacBook Pro M4 Max, 36 GB",
    fontsize=9.5,
    color="#6b7585",
    ha="center",
)

out = Path(__file__).resolve().parent.parent / "assets" / "benchmark-chart.png"
fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
print(f"Saved to {out}")
plt.close()
