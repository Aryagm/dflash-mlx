#!/usr/bin/env python3
"""Generate a side-by-side inference speed comparison video for Twitter."""

import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- Config ---
OUTPUT = Path(__file__).resolve().parent.parent / "assets" / "demo-comparison.mp4"
FPS = 30
DURATION_S = 27
WIDTH, HEIGHT = 1920, 1080

# Colors
BG = (13, 17, 23)
PANEL_BG = (22, 27, 34)
BORDER = (48, 54, 61)
PROMPT_BG = (30, 35, 44)
TEXT_DIM = (120, 130, 145)
TEXT_NORMAL = (200, 210, 220)
TEXT_BRIGHT = (240, 245, 250)
GREEN = (80, 220, 120)
BLUE = (90, 160, 255)
RED = (255, 80, 80)
YELLOW = (255, 220, 60)

# Fonts
FONT_MONO = "/System/Library/Fonts/Menlo.ttc"
FONT_HEADER = ImageFont.truetype(FONT_MONO, 18)
FONT_MODEL = ImageFont.truetype(FONT_MONO, 14)
FONT_TEXT = ImageFont.truetype(FONT_MONO, 15)
FONT_PROMPT = ImageFont.truetype(FONT_MONO, 14)
FONT_TITLE = ImageFont.truetype(FONT_MONO, 14)
FONT_TIMER = ImageFont.truetype(FONT_MONO, 15)
FONT_SPEED = ImageFont.truetype(FONT_MONO, 60)
FONT_TAG = ImageFont.truetype(FONT_MONO, 26)

# Load real captured outputs
import json
CAPTURE_DIR = Path(__file__).resolve().parent.parent / "assets" / "captures"

def load_capture(filename):
    with open(CAPTURE_DIR / filename) as f:
        data = json.load(f)
        return data["text"], data.get("tps", 0)

llama_text, _ = load_capture("llama_cpp.json")
mlx_text, _ = load_capture("mlx_bf16.json")
dflash_text, _ = load_capture("dflash_bf16.json")

# Peak TPS from benchmarks (plugged in, cool machine, M4 Max 36 GB)
FRAMEWORKS = [
    {"name": "llama.cpp", "tps": 35.6, "speed_color": RED,
     "text": llama_text, "model_label": "Qwen3.5-4B · bf16",
     "header_color": (60, 30, 30), "header_border": (120, 50, 50),
     "tag": None, "tag_color": None},
    {"name": "MLX-LM", "tps": 40.6, "speed_color": YELLOW,
     "text": mlx_text, "model_label": "Qwen3.5-4B · bf16",
     "header_color": (50, 45, 20), "header_border": (100, 90, 30),
     "tag": None, "tag_color": None},
    {"name": "DFlash + MLX", "tps": 100.5, "speed_color": GREEN,
     "text": dflash_text, "model_label": "Qwen3.5-4B · bf16",
     "header_color": (25, 50, 35), "header_border": (50, 120, 70),
     "tag": "LOSSLESS", "tag_color": GREEN},
]

PROMPT_TEXT = (
    "The function $f$ satisfies the functional equation f(x) + f(y) = f(x + y) - xy - 1 "
    "for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ "
    "such that $f(n) = n$. Enter all such integers, separated by commas. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

NUM_PANELS = len(FRAMEWORKS)
PANEL_W = (WIDTH - 20 * (NUM_PANELS + 1)) // NUM_PANELS
PROMPT_H = 72
PANEL_TOP = PROMPT_H + 18
PANEL_H = HEIGHT - PANEL_TOP - 12
HEADER_H = 52
TEXT_AREA_TOP = HEADER_H + 6

# Precompute finish time for each framework
for fw in FRAMEWORKS:
    total_tokens = len(fw["text"]) / 4
    fw["finish_time"] = 0.5 + total_tokens / fw["tps"]


def wrap_text(text: str, chars_per_line: int = 38) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            wrapped = textwrap.wrap(paragraph, width=chars_per_line)
            lines.extend(wrapped if wrapped else [""])
    return lines


def draw_frame(t: float) -> np.ndarray:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    # Prompt bar at top (centered label)
    draw.rounded_rectangle(
        [10, 8, WIDTH - 10, PROMPT_H + 8],
        radius=6, fill=PROMPT_BG, outline=BORDER
    )
    label = "Input Prompt"
    draw.text((24, 12), label, fill=TEXT_DIM, font=FONT_TITLE)
    # Wrap the full prompt across two lines
    prompt_lines = textwrap.wrap(PROMPT_TEXT, width=220)
    for j, pline in enumerate(prompt_lines[:3]):
        draw.text((24, 30 + j * 18), pline, fill=TEXT_NORMAL, font=FONT_PROMPT)

    for i, fw in enumerate(FRAMEWORKS):
        x = 20 + i * (PANEL_W + 20)
        y = PANEL_TOP

        # Panel background
        draw.rounded_rectangle(
            [x, y, x + PANEL_W, y + PANEL_H],
            radius=6, fill=PANEL_BG, outline=BORDER
        )

        tps = fw["tps"]
        gen_t = max(0, t - 0.5)
        tokens_generated = int(gen_t * tps)
        total_chars = tokens_generated * 4
        finished = total_chars >= len(fw["text"])
        finish_time = fw["finish_time"]

        # --- Colored header bar ---
        draw.rounded_rectangle(
            [x + 1, y + 1, x + PANEL_W - 1, y + HEADER_H],
            radius=5, fill=fw["header_color"]
        )
        draw.line([(x + 6, y + 1), (x + PANEL_W - 6, y + 1)],
                  fill=fw["header_border"], width=2)

        # Framework name (line 1)
        draw.text((x + 12, y + 8), fw["name"], fill=TEXT_BRIGHT, font=FONT_HEADER)

        # Model label (line 2)
        draw.text((x + 12, y + 30), fw["model_label"], fill=TEXT_DIM, font=FONT_MODEL)

        # Timer on the right (vertically centered)
        elapsed_display = min(gen_t, finish_time - 0.5) if finished else gen_t
        if gen_t > 0:
            timer_str = f"({elapsed_display:.1f}s)"
            timer_color = GREEN if finished else TEXT_DIM
            timer_bbox = draw.textbbox((0, 0), timer_str, font=FONT_TIMER)
            timer_w = timer_bbox[2] - timer_bbox[0]
            timer_x = x + PANEL_W - timer_w - 12
            draw.text((timer_x, y + 18), timer_str, fill=timer_color, font=FONT_TIMER)

        # --- Text area ---
        chars_per_line = int(PANEL_W / 9.0)
        fw_text = fw["text"]
        all_lines = wrap_text(fw_text, chars_per_line=chars_per_line)
        max_lines = (PANEL_H - TEXT_AREA_TOP - 10) // 19

        char_count = 0
        total_lines_generated = 0
        for ln in all_lines:
            line_chars = len(ln) + 1
            if char_count + line_chars > total_chars:
                break
            char_count += line_chars
            total_lines_generated += 1

        page = total_lines_generated // max_lines
        line_on_page = total_lines_generated % max_lines
        page_start = page * max_lines
        visible_lines = all_lines[page_start:page_start + line_on_page]

        text_y = y + TEXT_AREA_TOP + 4
        for line in visible_lines:
            if text_y > y + PANEL_H - 10:
                break
            draw.text((x + 12, text_y), line, fill=TEXT_NORMAL, font=FONT_TEXT)
            text_y += 19

        # Blinking cursor
        if not finished:
            cursor_visible = int(t * 3) % 2 == 0
            if cursor_visible and visible_lines:
                last_line = visible_lines[-1] if visible_lines else ""
                cursor_x = x + 12 + len(last_line) * 9.0
                cursor_y_pos = min(text_y - 19, y + PANEL_H - 20)
                draw.rectangle(
                    [cursor_x, cursor_y_pos, cursor_x + 9, cursor_y_pos + 17],
                    fill=TEXT_BRIGHT
                )

        # --- Speed overlay (centered, on top of text with background) ---
        if True:
            speed_color = fw["speed_color"]
            tps_str = f"{tps:.1f} TOK/S"

            # Measure text sizes for centering
            tps_bbox_size = draw.textbbox((0, 0), tps_str, font=FONT_SPEED)
            tps_w = tps_bbox_size[2] - tps_bbox_size[0]
            tps_h = tps_bbox_size[3] - tps_bbox_size[1]

            tag_text = None
            if fw["tag"]:
                tag_text = "DONE" if finished else fw["tag"]
            elif finished:
                tag_text = "DONE"

            tag_w, tag_h = 0, 0
            if tag_text:
                tag_bbox_size = draw.textbbox((0, 0), tag_text, font=FONT_TAG)
                tag_w = tag_bbox_size[2] - tag_bbox_size[0]
                tag_h = tag_bbox_size[3] - tag_bbox_size[1]

            # Total content height
            gap = 14 if tag_text else 0
            total_h = tps_h + gap + (tag_h if tag_text else 0)
            content_w = max(tps_w, tag_w)

            # Center on the whole page vertically
            panel_center_x = x + PANEL_W // 2
            panel_center_y = HEIGHT // 2

            tps_x = panel_center_x - tps_w // 2
            tps_y = panel_center_y - total_h // 2

            # Background rect with padding
            pad_x, pad_y = 20, 14
            bg_left = panel_center_x - content_w // 2 - pad_x
            bg_top = tps_y - pad_y
            bg_right = panel_center_x + content_w // 2 + pad_x
            bg_bottom = tps_y + total_h + pad_y

            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rounded_rectangle(
                [bg_left, bg_top, bg_right, bg_bottom],
                radius=10, fill=(13, 17, 23, 200)
            )
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(img)

            draw.text((tps_x, tps_y), tps_str, fill=speed_color, font=FONT_SPEED)

            if tag_text:
                tag_color = speed_color if fw["tag"] else GREEN
                tag_x = panel_center_x - tag_w // 2
                tag_y = tps_y + tps_h + gap
                draw.text((tag_x, tag_y), tag_text, fill=tag_color, font=FONT_TAG)

    return np.array(img)[:, :, ::-1]


def main():
    total_frames = int(FPS * DURATION_S)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT), fourcc, FPS, (WIDTH, HEIGHT))

    np.random.seed(42)
    for frame_idx in range(total_frames):
        t = frame_idx / FPS
        frame = draw_frame(t)
        out.write(frame)

        if frame_idx % FPS == 0:
            print(f"  {int(t)}s / {DURATION_S}s")

    for _ in range(FPS * 2):
        out.write(frame)

    out.release()

    final = OUTPUT.with_suffix(".final.mp4")
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", str(OUTPUT),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "fast",
        str(final)
    ], capture_output=True)
    final.rename(OUTPUT)
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
