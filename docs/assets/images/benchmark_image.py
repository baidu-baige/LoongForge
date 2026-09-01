# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""
Generate the LoongForge benchmark speedup chart used in README.

Style mirrors the ``#benchmark`` section of the LoongForge GitHub Pages site
(rounded gradient bars on a light track, model + type pill on the left, speedup
inside the bar). Baseline frameworks are intentionally *not* named — the chart
only shows LoongForge's own speedup numbers.

Usage:
    python docs/assets/images/benchmark_image.py

Output:
    docs/assets/images/benchmark_speedup.png

To update the chart:
    1. Edit the ROWS list below — model name, modality tag, speedup. That's it.
    2. Re-run this script.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

# ──────────────────────────── EDIT ME ────────────────────────────
# Each row: (model, type, speedup). Grouped by category in the order
# VLA → WAM → VLM → LLM, and sorted by speedup within each group.
# Bars are normalised against the largest speedup, and the fastest row gets
# the pink accent gradient.
ROWS = [
    # VLA
    ("Pi0.5",              "VLA", 2.80),
    ("GR00T N1.6",         "VLA", 2.31),
    ("GR00T N1.7",         "VLA", 1.79),
    ("X-VLA",              "VLA", 1.79),
    # WAM
    ("DreamZero",          "WAM", 4.38),
    ("FastWAM",            "WAM", 2.25),
    ("LingBot VA",         "WAM", 2.20),
    # VLM
    ("Qwen3-VL-30B-A3B",   "VLM", 1.45),
    # LLM
    ("DeepSeek-V3.2 Lite", "LLM", 5.04),
]

TITLE = "LoongForge — Training Throughput Speedup vs Open-Source Baselines"

BASELINE_CAPTION = "1.0× baseline"

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "benchmark_speedup.png")

# Preferred typefaces, first available wins. The site uses Inter; installing it
# (or Manrope) locally makes this chart match the web look exactly, otherwise it
# falls back to matplotlib's bundled DejaVu Sans.
FONT_CANDIDATES = ["Inter", "Manrope", "Plus Jakarta Sans", "Figtree",
                   "Helvetica Neue", "Arial", "DejaVu Sans"]

# ─────────────────────────── CONSTANTS ───────────────────────────
# Layout follows the site stylesheet (assets/css/style.css); the indigo /
# violet / ink values are taken from the architecture diagram
# (docs/assets/images/architecture/loongforge-architecture.svg) so the two
# README images read as one set.
COLOR_CANVAS     = "#FFFFFF"   # blends into the GitHub light-theme page
COLOR_TRACK      = "#E9ECF5"   # .bench-track, darkened to hold up on white
COLOR_BAR_A      = "#3B4FD8"   # architecture indigo (gradient start)
COLOR_BAR_B      = "#7C3AED"   # architecture violet (gradient end)
COLOR_BAR_TOP_B  = "#EC4899"   # .bench-bar-top gradient end (brand pink)
COLOR_LABEL      = "#2A2E37"   # architecture ink
COLOR_PILL_BG    = "#EAEEFF"   # architecture light indigo fill
COLOR_PILL_FG    = "#3B4FD8"   # architecture indigo
COLOR_BASELINE   = "#9CA3AF"   # .bench-baseline
COLOR_TITLE      = "#2A2E37"

# The axes uses pixel-sized data units at DPI, so 1 unit == 1 px in the PNG.
DPI = 150
PX_PER_PT = DPI / 72.0

# The PNG is a 2x asset: README pins it to W / 2 CSS px so browsers do an exact
# 2:1 downscale (and no downscale at all on HiDPI screens). Type is therefore
# sized for W / 2 — anything below ~20 px here turns to mush on screen.
W          = 1720          # canvas width  (px)
PAD_X      = 60            # left / right margin
LABEL_X    = PAD_X + 20    # model name baseline
TRACK_X1   = W - PAD_X     # track right edge
LABEL_GAP  = 44            # gap between the pill column and the track
PILL_GAP   = 16            # gap between the name column and the pill column
ROW_PITCH  = 84
GROUP_GAP  = 0             # extra spacing when the modality changes (0 = even)
BAR_H      = 48
BAR_PAD_R  = 26            # gap between bar end and the speedup text

FS_TITLE    = 33
FS_NAME     = 25
FS_SPEEDUP  = 22
FS_PILL     = 15
FS_BASELINE = 20
PILL_H      = 32           # modality pill height


def _font():
    """First available preferred typeface."""
    available = {f.name for f in fm.fontManager.ttflist}
    return next((n for n in FONT_CANDIDATES if n in available), "DejaVu Sans")


def _fs(px):
    """Font size in points for a target pixel height."""
    return px / PX_PER_PT


def _rounded(ax, x0, y0, x1, y1, **kw):
    """Fully rounded ("pill") rectangle in data coordinates."""
    h = y1 - y0
    r = h / 2.0
    patch = FancyBboxPatch(
        (x0 + r, y0), max(x1 - x0 - 2 * r, 0.0), h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        mutation_aspect=1, **kw,
    )
    ax.add_patch(patch)
    return patch


def _gradient_bar(ax, x0, y0, x1, y1, c_from, c_to, zorder=3):
    """Horizontal gradient fill clipped to a pill-shaped bar."""
    clip = _rounded(ax, x0, y0, x1, y1, fc="none", ec="none", zorder=zorder)
    cmap = LinearSegmentedColormap.from_list("bar", [c_from, c_to])
    im = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=cmap,
                   extent=(x0, x1, y1, y0), aspect="auto",
                   interpolation="bilinear", zorder=zorder)
    im.set_clip_path(clip)
    return clip


def _text_width(fig, artist):
    """Rendered width of a text artist, in data units (== px)."""
    bb = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    return bb.width


def main():
    plt.rcParams["font.family"] = _font()

    speedups = [r[2] for r in ROWS]
    peak = max(speedups)
    best = speedups.index(peak)                    # gets the pink accent bar

    # Row centres, with a little extra air whenever the modality changes.
    y_title = 64 if TITLE else 0
    y_rows = []
    y = y_title + (68 if TITLE else 40) + BAR_H / 2
    for i, (_, mtype, _) in enumerate(ROWS):
        if i and mtype != ROWS[i - 1][1]:
            y += GROUP_GAP
        y_rows.append(y)
        y += ROW_PITCH
    y_baseline = y_rows[-1] + BAR_H / 2 + 40       # "1.0× baseline" caption
    height = y_baseline + 48

    fig = plt.figure(figsize=(W / DPI, height / DPI), dpi=DPI)
    fig.patch.set_facecolor(COLOR_CANVAS)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(height, 0)                         # y grows downward
    ax.set_facecolor(COLOR_CANVAS)
    ax.set_axis_off()

    # Measure the label column first, so the pills line up in their own column
    # and the track starts right after them instead of at a hardcoded offset.
    def measure(s, px, weight="normal"):
        t = ax.text(0, 0, s, fontsize=_fs(px), fontweight=weight)
        w = _text_width(fig, t)
        t.remove()
        return w

    name_w = [measure(m, FS_NAME, "bold") for m, _, _ in ROWS]
    pill_w = max(measure(t, FS_PILL, "bold") for _, t, _ in ROWS) + 28
    pill_x0 = LABEL_X + max(name_w) + PILL_GAP
    track_x0 = pill_x0 + pill_w + LABEL_GAP

    if TITLE:
        ax.text(W / 2, y_title, TITLE, ha="center", va="center",
                fontsize=_fs(FS_TITLE), fontweight="bold", color=COLOR_TITLE)

    for i, (model, mtype, speedup) in enumerate(ROWS):
        yc = y_rows[i]
        y0, y1 = yc - BAR_H / 2, yc + BAR_H / 2

        # Track
        _rounded(ax, track_x0, y0, TRACK_X1, y1,
                 fc=COLOR_TRACK, ec="none", zorder=2)

        # Bar (the fastest row gets the pink accent gradient)
        frac = speedup / peak
        bar_x1 = track_x0 + frac * (TRACK_X1 - track_x0)
        c_to = COLOR_BAR_TOP_B if i == best else COLOR_BAR_B
        if i == best:   # soft glow under the highlighted bar
            _rounded(ax, track_x0 + 6, y0 + 8, bar_x1 - 6, y1 + 8,
                     fc=COLOR_BAR_TOP_B, ec="none", alpha=0.10, zorder=1)
        _gradient_bar(ax, track_x0, y0, bar_x1, y1, COLOR_BAR_A, c_to)

        # Speedup value, right-aligned inside the bar
        ax.text(bar_x1 - BAR_PAD_R, yc, f"{speedup:.2f}×",
                ha="right", va="center", zorder=5,
                fontsize=_fs(FS_SPEEDUP), fontweight="bold", color="white")

        # Model name (left column) + modality pill (aligned column)
        ax.text(LABEL_X, yc, model, ha="left", va="center", zorder=4,
                fontsize=_fs(FS_NAME), fontweight="bold", color=COLOR_LABEL)
        ax.text(pill_x0 + pill_w / 2, yc, mtype, ha="center", va="center",
                zorder=5, fontsize=_fs(FS_PILL), fontweight="bold",
                color=COLOR_PILL_FG)
        _rounded(ax, pill_x0, yc - PILL_H / 2, pill_x0 + pill_w, yc + PILL_H / 2,
                 fc=COLOR_PILL_BG, ec="none", zorder=4)

    # Baseline caption, centred under the track
    ax.text((track_x0 + TRACK_X1) / 2, y_baseline, BASELINE_CAPTION,
            ha="center", va="center", fontsize=_fs(FS_BASELINE),
            color=COLOR_BASELINE, style="italic")

    fig.savefig(OUTPUT_PATH, dpi=DPI, facecolor=COLOR_CANVAS)
    print(f"Saved: {OUTPUT_PATH}  ({plt.rcParams['font.family'][0]})")


if __name__ == "__main__":
    main()
