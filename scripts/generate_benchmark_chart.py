#!/usr/bin/env python3
"""Generate the benchmark chart PNG for the README."""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# --- Data (sorted smallest to largest) ---
entries = [
    ("MLX bf16",              40.6,  "#3d4460"),
    ("llama.cpp Q4_K_M",      76.4,  "#3d4460"),
    ("DFlash + MLX bf16",    100.5,  "#3668b2"),
    ("MLX 4-bit",            119.4,  "#3d4460"),
    ("DFlash + MLX 4-bit",   161.9,  "#4f8cff"),
]

labels = [e[0] for e in entries]
values = [e[1] for e in entries]
colors = [e[2] for e in entries]

# --- Style ---
BG = "#0d1117"

mpl.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
})

fig, ax = plt.subplots(figsize=(8, 3.4))
fig.subplots_adjust(left=0.28, right=0.92, top=0.78, bottom=0.08)

bars = ax.barh(range(len(entries)), values, height=0.52, color=colors, edgecolor="none")

for i, (bar, val) in enumerate(zip(bars, values)):
    is_hero = i == len(entries) - 1
    is_dflash = "DFlash" in labels[i]
    if is_hero:
        col = "#ffffff"
        weight = "bold"
    elif is_dflash:
        col = "#a0c4ff"
        weight = "medium"
    else:
        col = "#8b95a5"
        weight = "normal"
    ax.text(val + 2.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=11, fontweight=weight,
            color=col, fontfamily="monospace")

ax.set_yticks(range(len(entries)))
ax.set_yticklabels(labels, fontsize=10.5, color="#c0c8d4")
ax.set_xlim(0, 195)
ax.xaxis.set_visible(False)
ax.spines[:].set_visible(False)
ax.tick_params(left=False, bottom=False)

fig.text(0.5, 0.94, "Qwen3.5-4B on Mac",
         fontsize=14, fontweight="bold", color="#e6eaf0", ha="center")
fig.text(0.5, 0.87, "tok/s, higher is better  \u00b7  MacBook Pro M4 Max, 36 GB",
         fontsize=9.5, color="#6b7585", ha="center")

out = Path(__file__).resolve().parent.parent / "assets" / "benchmark-chart.png"
fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
print(f"Saved to {out}")
plt.close()
