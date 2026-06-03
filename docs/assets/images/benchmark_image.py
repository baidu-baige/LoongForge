"""
Generate the LoongForge benchmark speedup chart used in README.

Usage:
    python docs/assets/images/benchmark_speedup.py

Output:
    docs/assets/images/benchmark_speedup.png

To update the chart:
    1. Edit the ROWS list below (already sorted descending by speedup).
    2. Update VERSION_TAG and FOOTNOTES if needed.
    3. Re-run this script.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────── EDIT ME ────────────────────────────
# Each row: (model, type, baseline, config, speedup, marker)
# - Sort manually: largest speedup at TOP for visual impact.
# - marker: "" or one of the footnote symbols ("§", "*", etc.)
ROWS = [
    ("DeepSeek-V3.2 Lite", "MoE + DSA", "Megatron-LM", "Reduced layers · GBS 128 · 8K",      5.04, "§"),
    ("GR00T N1.6",         "VLA",       "LeRobot",     "8 × A800 · GBS 128 · 224×224",       2.31, ""),
    ("Pi 0.5",             "VLA",       "OpenPI",      "8 × A800 · GBS 112 · 224×224",       1.65, ""),
    ("Qwen3-VL-30B-A3B",   "VLM",       "VeOmni",      "32 × A800 · GBS 128 · 32K",          1.45, ""),
    ("Qwen3-30B-A3B",      "MoE",       "Megatron-LM", "32 × A800 · GBS 1024 · 32K",         1.16, ""),
]

VERSION_TAG = "v0.1.1"

FOOTNOTES = [
    "§  DeepSeek-V3.2 was validated on a reduced-layer setup; LoongForge's DSA kernels still deliver ~5× speedup and reach 64K sequence (baseline OOMs beyond 8K).",
    "†  Numbers reflect baseline and LoongForge versions at measurement time, and may evolve as implementations change.",
]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "benchmark_speedup.png")

# ─────────────────────────── CONSTANTS ───────────────────────────
COLOR_LOONGFORGE = "#8B5CF6"   # brand purple (matches logo gradient mid-stop)
COLOR_BASELINE   = "#B0B7C0"
COLOR_NUM        = "#5B21B6"   # deep purple, for speedup numbers
COLOR_GREY_DARK  = "#5A6068"
COLOR_GREY_MID   = "#888888"
COLOR_FOOTNOTE_1 = "#2E3338"   # darker for primary footnote
COLOR_FOOTNOTE_2 = "#5A6068"   # mid for secondary footnote
COLOR_GREY_LITE  = "#CCCCCC"
COLOR_TEXT       = "#1A1A1A"

# ────────────────────────────── PLOT ─────────────────────────────
def main():
    n = len(ROWS)
    y = np.arange(n)
    height = 0.36

    fig, ax = plt.subplots(figsize=(12, 7.4), dpi=150)
    fig.patch.set_facecolor("white")

    speedups = [r[4] for r in ROWS]
    baselines = [1.0] * n

    ax.barh(y - height / 2, speedups, height, label="LoongForge",
            color=COLOR_LOONGFORGE, edgecolor="white", linewidth=0.5, zorder=3)
    ax.barh(y + height / 2, baselines, height, label="Baseline",
            color=COLOR_BASELINE, edgecolor="white", linewidth=0.5, zorder=3)

    # Speedup numbers
    for i, (_, _, _, _, sp, _) in enumerate(ROWS):
        ax.text(sp + 0.10, y[i] - height / 2, f"{sp:.2f}×",
                va="center", ha="left", fontsize=12,
                fontweight="bold", color=COLOR_NUM)

    # Baseline labels (small, grey) — append † superscript referencing the 2nd footnote
    for i, (_, _, baseline, _, _, _) in enumerate(ROWS):
        ax.text(1.0 + 0.10, y[i] + height / 2,
                f"1.00×  ({baseline}$^{{\\dagger}}$)",
                va="center", ha="left", fontsize=9, color=COLOR_GREY_DARK)

    # Y-tick labels: model (with optional superscript marker) + type.
    # Use mathtext ($^{...}$) to render the marker as a true superscript.
    # `§` in mathtext is `\S`; other markers (e.g. `*`, `†`) pass through.
    SUP_MAP = {"§": r"\S", "†": r"\dagger", "*": "*"}
    def _label(r):
        sup = f"$^{{{SUP_MAP.get(r[5], r[5])}}}$" if r[5] else ""
        return f"{r[0]}{sup}  ({r[1]})"
    ax.set_yticks(y)
    ax.set_yticklabels([_label(r) for r in ROWS],
                       fontsize=11, fontweight="bold", color=COLOR_TEXT)

    # Secondary y-line: config, italic grey
    for i, (_, _, _, config, _, _) in enumerate(ROWS):
        ax.annotate(config,
                    xy=(0, y[i]), xycoords=("axes fraction", "data"),
                    xytext=(-6, -14), textcoords="offset points",
                    ha="right", va="center",
                    fontsize=8.5, color=COLOR_GREY_MID, style="italic")

    # X-axis
    ax.set_xlabel("Training Speedup (× over baseline)",
                  fontsize=11.5, fontweight="bold", labelpad=12)
    upper = max(6.4, max(speedups) + 1.4)
    ax.set_xlim(0, upper)
    ax.set_xticks([t for t in range(0, int(upper) + 1)])
    # Shift xlabel slightly left so it visually centers around the bar cluster
    # (default centers it on the plot area, which here ends at ~6.4 → label looks right-shifted)
    ax.xaxis.set_label_coords(0.4, -0.09)
    ax.invert_yaxis()

    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_GREY_LITE)
    ax.spines["bottom"].set_color(COLOR_GREY_LITE)

    ax.set_title("LoongForge — Training Speedup vs Open-Source Baselines",
                 fontsize=14, fontweight="bold", pad=20,
                 loc="left", color=COLOR_TEXT)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)

    # Footnotes — slight-negative x forces tight-crop's left edge to align here,
    # so footnotes sit flush-left in the saved image with only pad_inches of margin.
    fig.subplots_adjust(bottom=0.20)
    base_y = 0.075
    step = 0.035
    for i, fn in enumerate(FOOTNOTES):
        color = COLOR_FOOTNOTE_1 if i == 0 else COLOR_FOOTNOTE_2
        fig.text(-0.02, base_y - i * step, fn,
                 fontsize=8.6, color=color, ha="left")

    # Version tag (top-right)
    fig.text(0.98, 0.965, VERSION_TAG, fontsize=9,
             color=COLOR_GREY_MID, ha="right", va="top", style="italic")

    plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight",
                pad_inches=0.08, facecolor="white")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
