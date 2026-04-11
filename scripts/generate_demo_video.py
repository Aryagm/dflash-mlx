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
DURATION_S = 27  # total video duration (llama.cpp needs ~24s)
WIDTH, HEIGHT = 1920, 1080

# Colors (BGR for OpenCV, RGB for Pillow)
BG = (13, 17, 23)
PANEL_BG = (22, 27, 34)
BORDER = (48, 54, 61)
PROMPT_BG = (30, 35, 44)
TEXT_DIM = (120, 130, 145)
TEXT_NORMAL = (200, 210, 220)
TEXT_BRIGHT = (240, 245, 250)
GREEN = (80, 220, 120)
BLUE = (90, 160, 255)
YELLOW = (240, 180, 40)

# Fonts
FONT_MONO = "/System/Library/Fonts/Menlo.ttc"
FONT_LABEL = ImageFont.truetype(FONT_MONO, 15)
FONT_TEXT = ImageFont.truetype(FONT_MONO, 14)
FONT_SPEED = ImageFont.truetype(FONT_MONO, 48)
FONT_SPEED_UNIT = ImageFont.truetype(FONT_MONO, 20)
FONT_PROMPT = ImageFont.truetype(FONT_MONO, 13)
FONT_TITLE = ImageFont.truetype(FONT_MONO, 13)
FONT_TAG = ImageFont.truetype(FONT_MONO, 14)
FONT_TIMER = ImageFont.truetype(FONT_MONO, 13)
FONT_DONE = ImageFont.truetype(FONT_MONO, 12)

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
    {"name": "llama.cpp", "tps": 35.6, "color": TEXT_DIM, "tag": None, "tag_color": None,
     "text": llama_text},
    {"name": "MLX-LM", "tps": 40.6, "color": TEXT_DIM, "tag": None, "tag_color": None,
     "text": mlx_text},
    {"name": "DFlash + MLX", "tps": 100.5, "color": BLUE, "tag": "LOSSLESS", "tag_color": GREEN,
     "text": dflash_text},
]

PROMPT_TEXT = (
    "The function f satisfies f(x) + f(y) = f(x + y) - xy - 1 "
    "for all real numbers x and y. If f(1) = 1, find all integers n "
    "such that f(n) = n."
)

NUM_PANELS = len(FRAMEWORKS)
PANEL_W = (WIDTH - 20 * (NUM_PANELS + 1)) // NUM_PANELS
PROMPT_H = 60
PANEL_TOP = PROMPT_H + 30
PANEL_H = HEIGHT - PANEL_TOP - 20
HEADER_H = 42  # within panel, for name + model + timer
TEXT_AREA_TOP = HEADER_H + 10  # within panel, below header
TEXT_AREA_BOTTOM = 90  # within panel, above speed display

# Precompute finish time for each framework
for fw in FRAMEWORKS:
    total_tokens = len(fw["text"]) / 4  # ~4 chars per token
    fw["finish_time"] = 0.5 + total_tokens / fw["tps"]  # 0.5s prefill


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
    """Draw a single frame at time t seconds."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    # Prompt bar at top
    draw.rounded_rectangle(
        [10, 10, WIDTH - 10, PROMPT_H + 10],
        radius=8, fill=PROMPT_BG, outline=BORDER
    )
    draw.text((20, 16), "Input Prompt", fill=TEXT_DIM, font=FONT_TITLE)
    prompt_wrapped = textwrap.shorten(PROMPT_TEXT, width=140, placeholder="...")
    draw.text((20, 36), prompt_wrapped, fill=TEXT_NORMAL, font=FONT_PROMPT)

    for i, fw in enumerate(FRAMEWORKS):
        x = 20 + i * (PANEL_W + 20)
        y = PANEL_TOP

        # Panel background
        draw.rounded_rectangle(
            [x, y, x + PANEL_W, y + PANEL_H],
            radius=8, fill=PANEL_BG, outline=BORDER
        )

        tps = fw["tps"]
        gen_t = max(0, t - 0.5)
        tokens_generated = int(gen_t * tps)
        total_chars = tokens_generated * 4
        finished = total_chars >= len(fw["text"])
        finish_time = fw["finish_time"]

        # Header line 1: framework name + tag (inline) + timer
        label_color = fw["color"]
        draw.text((x + 12, y + 8), fw["name"], fill=label_color, font=FONT_LABEL)

        # Tag (e.g. LOSSLESS) - inline after the name
        if fw["tag"]:
            tag_text = fw["tag"]
            name_bbox = draw.textbbox((x + 12, y + 8), fw["name"], font=FONT_LABEL)
            tag_x = name_bbox[2] + 10
            tag_bbox = draw.textbbox((0, 0), tag_text, font=FONT_DONE)
            tag_w = tag_bbox[2] - tag_bbox[0] + 12
            tag_h = 17
            tag_y = y + 10
            draw.rounded_rectangle(
                [tag_x, tag_y, tag_x + tag_w, tag_y + tag_h],
                radius=4, fill=(30, 60, 40), outline=fw["tag_color"]
            )
            draw.text((tag_x + 6, tag_y + 2), tag_text, fill=fw["tag_color"], font=FONT_DONE)

        # Timer (stops when generation finishes)
        elapsed_display = min(gen_t, finish_time - 0.5) if finished else gen_t
        if gen_t > 0:
            timer_str = f"{elapsed_display:.1f}s"
            timer_color = GREEN if finished else TEXT_DIM
            timer_bbox = draw.textbbox((0, 0), timer_str, font=FONT_TIMER)
            timer_w = timer_bbox[2] - timer_bbox[0]
            timer_x = x + PANEL_W - timer_w - 12
            draw.text((timer_x, y + 10), timer_str, fill=timer_color, font=FONT_TIMER)

        # Header line 2: model name
        draw.text((x + 12, y + 30), "Qwen3.5-4B · bf16", fill=TEXT_DIM, font=FONT_PROMPT)

        # Separator below header
        draw.line([(x + 12, y + HEADER_H + 6), (x + PANEL_W - 12, y + HEADER_H + 6)],
                  fill=BORDER, width=1)

        # Allow wrapping: text repeats from top when viewport fills
        chars_per_line = int(PANEL_W / 8.5)
        fw_text = fw["text"]
        all_lines = wrap_text(fw_text, chars_per_line=chars_per_line)
        max_lines = (PANEL_H - TEXT_AREA_TOP - TEXT_AREA_BOTTOM) // 18

        # Figure out how many wrapped lines correspond to total_chars generated
        char_count = 0
        total_lines_generated = 0
        for ln in all_lines:
            line_chars = len(ln) + 1  # +1 for newline
            if char_count + line_chars > total_chars:
                break
            char_count += line_chars
            total_lines_generated += 1

        # Which "page" are we on?
        page = total_lines_generated // max_lines
        line_on_page = total_lines_generated % max_lines

        # Get the lines for the current page
        page_start = page * max_lines
        visible_lines = all_lines[page_start:page_start + line_on_page]

        # Draw
        text_y = y + TEXT_AREA_TOP
        for line in visible_lines:
            if text_y > y + PANEL_H - TEXT_AREA_BOTTOM:
                break
            draw.text((x + 12, text_y), line, fill=TEXT_NORMAL, font=FONT_TEXT)
            text_y += 18

        # Blinking cursor
        if not finished:
            cursor_visible = int(t * 3) % 2 == 0
            if cursor_visible and visible_lines:
                last_line = visible_lines[-1] if visible_lines else ""
                cursor_x = x + 12 + len(last_line) * 8.4
                cursor_y_pos = min(text_y - 18, y + PANEL_H - TEXT_AREA_BOTTOM - 18)
                draw.rectangle(
                    [cursor_x, cursor_y_pos, cursor_x + 8, cursor_y_pos + 16],
                    fill=TEXT_BRIGHT
                )

        # Speed display at bottom
        speed_y = y + PANEL_H - 70
        # Separator line
        draw.line([(x + 12, speed_y - 8), (x + PANEL_W - 12, speed_y - 8)],
                  fill=BORDER, width=1)

        if gen_t > 0.2:
            speed_color = fw["color"] if fw["color"] != TEXT_DIM else TEXT_NORMAL
            tps_str = f"{tps:.1f}"
            draw.text((x + 12, speed_y), tps_str, fill=speed_color, font=FONT_SPEED)

            # "tok/s" unit
            tps_bbox = draw.textbbox((x + 12, speed_y), tps_str, font=FONT_SPEED)
            draw.text((tps_bbox[2] + 8, speed_y + 24), "tok/s",
                      fill=TEXT_DIM, font=FONT_SPEED_UNIT)

            # DONE badge when finished
            if finished:
                done_text = "DONE"
                done_bbox = draw.textbbox((0, 0), done_text, font=FONT_DONE)
                done_w = done_bbox[2] - done_bbox[0] + 14
                done_x = x + PANEL_W - done_w - 12
                done_y = speed_y + 8
                draw.rounded_rectangle(
                    [done_x, done_y, done_x + done_w, done_y + 22],
                    radius=4, fill=(30, 60, 40), outline=GREEN
                )
                draw.text((done_x + 7, done_y + 4), done_text, fill=GREEN, font=FONT_DONE)

    return np.array(img)[:, :, ::-1]  # RGB to BGR for OpenCV


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

    # Hold final frame for 2 extra seconds
    for _ in range(FPS * 2):
        out.write(frame)

    out.release()

    # Re-encode with ffmpeg for better compatibility
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
