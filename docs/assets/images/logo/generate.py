"""Regenerate the LoongForge logo assets that sit next to this script.

The wordmark and tagline are converted to outlines, so the lockup renders
identically everywhere instead of depending on Inter being installed on the
reader's machine. That also means the SVGs cannot be edited by hand — change
the constants below and re-run this script instead.

Setup, then run:

    pip install uharfbuzz fonttools
    curl -LO https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip
    unzip -q Inter-4.1.zip -d inter
    python generate.py --font-dir inter/extras/otf

Writes banner.svg, banner-dark.svg and logo.svg. Inter is SIL OFL 1.1;
embedding outlines in a logo is a permitted use.
"""
import argparse
import os

from fontTools.misc.transform import Transform
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
import uharfbuzz as hb

WORDMARK = "LoongForge"
TAGLINE = "Train LLMs, VLMs, diffusion & embodied models — faster."

WM_FONT, WM_SIZE, WM_TRACK = "InterDisplay-ExtraBold.otf", 86, -2.5
TL_FONT, TL_SIZE, TL_TRACK = "Inter-SemiBold.otf", 19, 0.3

PAD = 21          # left/right padding, matches the icon's own left inset
GAP = 52          # optical gap between the mark and the wordmark
LINE_GAP = 9      # ink gap between wordmark descenders and the tagline cap line
TOP = 15.0        # the wordmark cap line and the top of the mark both start here
BOTTOM_PAD = 14.4

THEMES = {
    "light": dict(icon=("#4F46E5", "#7C3AED", "#DB2777"),
                  text=("#4F46E5", "#7C3AED", "#DB2777"), tag="#423C55"),
    "dark": dict(icon=("#6366F1", "#8B5CF6", "#F472B6"),
                 text=("#A5B4FC", "#FDE68A", "#F472B6"), tag="#E8E4F0"),
}

# The mark, drawn on a 64-unit grid. The inner group recentres and enlarges the
# glyph inside the squircle; logo.svg reuses this verbatim so the standalone
# icon and the lockup cannot drift apart.
MARK = """    <g transform="translate(32,32) scale(1.3) translate(-32.45,-33.85)">
      <path d="M18 40 C 22 30, 28 28, 32 32 C 36 36, 42 34, 46 24"
            stroke="#fff" stroke-width="3.2" stroke-linecap="round" fill="none"/>
      <circle cx="46" cy="24" r="3.2" fill="#fff"/>
      <circle cx="18" cy="40" r="2.2" fill="#fff" opacity="0.85"/>
      <path d="M24 46 L40 46" stroke="#fff" stroke-width="2" stroke-linecap="round" opacity="0.7"/>
    </g>"""


def outline(font_path, text, size, tracking):
    """Shape with HarfBuzz so kerning applies, then emit one combined path.

    Returns (d, ink) where ink is the inked bbox relative to an origin at x=0
    on the baseline, y growing downward as in SVG.
    """
    tt = TTFont(font_path)
    scale = size / tt["head"].unitsPerEm
    glyphs = tt.getGlyphSet()

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb.Font(hb.Face(hb.Blob.from_file_path(font_path))), buf)
    shaped = list(zip(buf.glyph_infos, buf.glyph_positions))

    def replay(pen):
        x = 0.0
        for info, pos in shaped:
            t = (Transform()
                 .translate(x + pos.x_offset * scale, -pos.y_offset * scale)
                 .scale(scale, -scale))
            glyphs[tt.getGlyphName(info.codepoint)].draw(TransformPen(pen, t))
            x += pos.x_advance * scale + tracking

    svg_pen = SVGPathPen(glyphs, ntos=lambda v: f"{v:.2f}")
    replay(svg_pen)
    bounds_pen = BoundsPen(glyphs)
    replay(bounds_pen)
    return svg_pen.getCommands(), bounds_pen.bounds


def banner(theme, font_dir):
    wd, wink = outline(os.path.join(font_dir, WM_FONT), WORDMARK, WM_SIZE, WM_TRACK)
    td, tink = outline(os.path.join(font_dir, TL_FONT), TAGLINE, TL_SIZE, TL_TRACK)

    # Key the mark's height to the two strongest horizontals in the type block:
    # the wordmark cap line and the tagline baseline. Both edges then read
    # flush, which matching the two centres does not achieve when the masses
    # differ in height.
    tagline_baseline = wink[3] - tink[1] + LINE_GAP   # relative to the wordmark baseline
    scale = (tagline_baseline - wink[1]) / 60.0       # the squircle is 60 units tall

    icon_x = PAD - 2 * scale                         # squircle ink starts at PAD
    icon_y = TOP - 2 * scale
    text_x = PAD + 60 * scale + GAP
    b1 = TOP - wink[1]
    b2 = b1 + tagline_baseline

    width = round(text_x + max(wink[2], tink[2]) + PAD)
    height = round(max(TOP + 60 * scale, b2 + tink[3]) + BOTTOM_PAD)
    t = THEMES[theme]

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}"
     role="img" aria-label="LoongForge — Train LLMs, VLMs, diffusion and embodied models, faster.">
  <!-- Wordmark and tagline are outlined Inter (SIL OFL 1.1), so the lockup
       renders identically everywhere instead of falling back to Arial. -->
  <defs>
    <linearGradient id="lf-icon" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%"   stop-color="{t['icon'][0]}"/>
      <stop offset="50%"  stop-color="{t['icon'][1]}"/>
      <stop offset="100%" stop-color="{t['icon'][2]}"/>
    </linearGradient>
    <linearGradient id="lf-text" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%"   stop-color="{t['text'][0]}"/>
      <stop offset="50%"  stop-color="{t['text'][1]}"/>
      <stop offset="100%" stop-color="{t['text'][2]}"/>
    </linearGradient>
  </defs>

  <!-- Icon: squircle + mark, keyed to cap line -> tagline baseline -->
  <g transform="translate({icon_x:.2f},{icon_y:.2f}) scale({scale:.4f})">
    <rect x="2" y="2" width="60" height="60" rx="14" fill="url(#lf-icon)"/>
{MARK}
  </g>

  <!-- Wordmark: InterDisplay ExtraBold {WM_SIZE}px, tracking {WM_TRACK} -->
  <g transform="translate({text_x:.1f},{b1:.2f})" fill="url(#lf-text)">
    <path d="{wd}"/>
  </g>

  <!-- Tagline: Inter SemiBold {TL_SIZE}px, tracking {TL_TRACK} -->
  <g transform="translate({text_x:.1f},{b2:.2f})" fill="{t['tag']}">
    <path d="{td}"/>
  </g>
</svg>
"""


def standalone():
    """The mark on its own, for favicons and avatars."""
    icon = THEMES["light"]["icon"]
    mark = "\n".join(line[2:] if line.startswith("  ") else line
                     for line in MARK.split("\n"))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64"
     role="img" aria-label="LoongForge logo">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%"   stop-color="{icon[0]}"/>
      <stop offset="50%"  stop-color="{icon[1]}"/>
      <stop offset="100%" stop-color="{icon[2]}"/>
    </linearGradient>
  </defs>
  <rect x="2" y="2" width="60" height="60" rx="14" fill="url(#g)"/>
{mark}
</svg>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--font-dir", required=True,
                    help="directory holding the Inter static OTFs "
                         "(the extras/otf folder of the Inter release)")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--check", action="store_true",
                    help="report whether the files on disk are up to date "
                         "instead of writing them")
    args = ap.parse_args()

    assets = {"banner.svg": banner("light", args.font_dir),
              "banner-dark.svg": banner("dark", args.font_dir),
              "logo.svg": standalone()}

    stale = []
    for name, content in assets.items():
        path = os.path.join(args.out_dir, name)
        if args.check:
            on_disk = open(path, encoding="utf-8").read() if os.path.exists(path) else None
            status = "ok" if on_disk == content else "STALE"
            if status == "STALE":
                stale.append(name)
            print(f"{status:5} {name}")
        else:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            print(f"wrote {name} ({len(content)} bytes)")
    raise SystemExit(1 if stale else 0)


if __name__ == "__main__":
    main()
