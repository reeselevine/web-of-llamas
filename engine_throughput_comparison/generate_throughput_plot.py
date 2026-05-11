#!/usr/bin/env python3
"""
Generate 2-panel throughput comparison SVG (Prefill / Decode) — side by side.
Groups shown: Chrome/fp16, Safari/fp16  (q4 data kept but not plotted)
Engines per group: wllama, WebLLM, Transformers.js

Usage: python3 generate_throughput_plot.py
Output: engine_throughput_comparison.svg
"""

OUT  = 'engine_throughput_comparison.svg'
FONT = "Georgia, 'Times New Roman', Times, serif"

# ── Colors (Wong colorblind-safe palette) ─────────────────────────────────────
WLLAMA = "#009E73"   # bluish green  (Wong)
WEBLLM = "#56B4E9"   # sky blue      (Wong)
TJS    = "#E69F00"   # orange        (Wong)
NATIVE = "#CC79A7"   # reddish purple (Wong)

POS_PCT = "#1a5c35"  # dark green for positive % labels
NEG_PCT = "#7a1515"  # dark red for negative % labels

# ── Data ──────────────────────────────────────────────────────────────────────
# Groups: (label, wllama_prefill, wllama_decode,
#                 webllm_prefill, webllm_decode,
#                 tjs_prefill,   tjs_decode)
groups = [
    ("Chrome / fp16", 1010.5, 74.2,  1828.2, 51.2,  955.3,  38.3),
    ("Safari / fp16", 167.5,  60.2,  588.4,  43.1,  732.9,  22.2),
    ("Chrome / q4",   611.5,  110.0, 1832.3, 68.1,  959.3,  124.4),
    ("Safari / q4",   152.5,  63.0,  501.5,  62.0,  634.8,  50.3),
]

native = {
    "q4_k_m": {"prefill": 2843.7, "decode": 180.3},
    "fp16":   {"prefill": 3140.1, "decode": 85.2},
}

# Only fp16 groups are plotted
plot_groups = [g for g in groups if "fp16" in g[0]]

# ── Layout ────────────────────────────────────────────────────────────────────
W       = 700
H       = 355
HALF    = W // 2        # 350 — boundary between left and right panels
LMARGIN = 58
RMARGIN = 10

NG = len(plot_groups)   # 2
NB = 3                  # bars per group (wllama, WebLLM, TJS)
BAR_W    = 26
BAR_GAP  = 5
TOTAL_BW = NB * BAR_W + (NB - 1) * BAR_GAP   # 88px

# Horizontal extents of each panel's plot area
P1_X0 = LMARGIN           # 58
P1_X1 = HALF - RMARGIN    # 340
P2_X0 = HALF + LMARGIN    # 408
P2_X1 = W - RMARGIN       # 690

GROUP_W = (P1_X1 - P1_X0) / NG   # same width for both panels

# Shared vertical extents
P_Y0 = 74    # plot top
P_Y1 = 310   # plot bottom

P_YMAX_1 = 2100   # prefill scale
P_YMAX_2 = 145    # decode scale

GRIDS_1 = list(range(0, 2001, 200))
GRIDS_2 = list(range(0, 141, 20))

# ── Helpers ───────────────────────────────────────────────────────────────────
def bar_x(gi, bi, x0):
    group_start = x0 + gi * GROUP_W
    offset = (GROUP_W - TOTAL_BW) / 2
    return group_start + offset + bi * (BAR_W + BAR_GAP)

def yp(v, ymax, y0, y1):
    return max(y0, y1 - (v / ymax) * (y1 - y0))

def yg(v, ymax, y0, y1):
    return y1 - (v / ymax) * (y1 - y0)

def group_cx(gi, x0):
    return x0 + gi * GROUP_W + GROUP_W / 2

def make_grids(vals, ymax, y0, y1, x0, x1):
    out = ""
    for v in vals:
        y = yg(v, ymax, y0, y1)
        out += (f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                f'stroke="#ccc" stroke-width="0.8" stroke-dasharray="4,3"/>\n')
        out += (f'<text x="{x0-5}" y="{y+4:.1f}" text-anchor="end" font-size="13" '
                f'fill="#111" font-family="{FONT}">{v}</text>\n')
    return out

def make_bars(panel_idx, ymax, y0, y1, x0):
    """panel_idx: 0=prefill, 1=decode"""
    out = ""
    for gi, row in enumerate(plot_groups):
        label, wp, wd, wlp, wld, tp, td = row
        w_val  = wp  if panel_idx == 0 else wd
        wl_val = wlp if panel_idx == 0 else wld
        t_val  = tp  if panel_idx == 0 else td

        for bi, (val, col) in enumerate([(w_val, WLLAMA), (wl_val, WEBLLM), (t_val, TJS)]):
            bx = bar_x(gi, bi, x0)
            cx = bx + BAR_W / 2
            by = yp(val, ymax, y0, y1)
            bh = y1 - by
            out += f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BAR_W}" height="{bh:.1f}" fill="{col}" rx="2"/>\n'

            if bi > 0:
                pct  = (val - w_val) / w_val * 100
                sign = "+" if pct >= 0 else ""
                pcol = POS_PCT if pct >= 0 else NEG_PCT
                label_y = max(y0 + 10, by - 5)
                out += (f'<text x="{cx:.1f}" y="{label_y:.1f}" text-anchor="middle" '
                        f'font-size="11" fill="{pcol}" font-family="{FONT}" '
                        f'font-weight="bold">{sign}{pct:.0f}%</text>\n')
    return out

def make_xlabels(y_base, x0):
    out = ""
    for gi, row in enumerate(plot_groups):
        cx = group_cx(gi, x0)
        out += (f'<text x="{cx:.0f}" y="{y_base}" text-anchor="middle" font-size="14" '
                f'fill="#111" font-family="{FONT}" font-weight="bold">{row[0]}</text>\n')
    return out

def axes(y0, y1, x0, x1):
    return (f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#111" stroke-width="1.5"/>\n'
            f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#111" stroke-width="1.5"/>\n')

# ── Legend geometry ───────────────────────────────────────────────────────────
# Three swatches spaced evenly at W/4, W/2, 3W/4
LEG_X, LEG_Y, LEG_W, LEG_H = 50, 8, 600, 26
LEG_PAD_Y = 6
TEXT_Y    = LEG_Y + LEG_PAD_Y + 11

SW1_X = 118   # wllama  swatch x
SW2_X = 288   # WebLLM  swatch x
SW3_X = 458   # TJS     swatch x

# ── Build SVG ──────────────────────────────────────────────────────────────────
P1_CX = (P1_X0 + P1_X1) / 2
P2_CX = (P2_X0 + P2_X1) / 2
TITLE_Y = 57
YLABEL_MID = -(P_Y0 + P_Y1) / 2   # rotated-coordinate midpoint (shared)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<rect width="100%" height="100%" fill="white"/>

<!-- Legend box -->
<rect x="{LEG_X}" y="{LEG_Y}" width="{LEG_W}" height="{LEG_H}"
      fill="#f7f7f7" stroke="#111" stroke-width="1.5"/>
<rect x="{SW1_X}"    y="{LEG_Y + LEG_PAD_Y}" width="14" height="14" fill="{WLLAMA}"/>
<text x="{SW1_X+18}" y="{TEXT_Y}" font-size="14" font-family="{FONT}" fill="#111" font-weight="bold">wllama</text>
<rect x="{SW2_X}"    y="{LEG_Y + LEG_PAD_Y}" width="14" height="14" fill="{WEBLLM}"/>
<text x="{SW2_X+18}" y="{TEXT_Y}" font-size="14" font-family="{FONT}" fill="#111" font-weight="bold">WebLLM</text>
<rect x="{SW3_X}"    y="{LEG_Y + LEG_PAD_Y}" width="14" height="14" fill="{TJS}"/>
<text x="{SW3_X+18}" y="{TEXT_Y}" font-size="14" font-family="{FONT}" fill="#111" font-weight="bold">Transformers.js</text>

<!-- ── PANEL 1: Prefill ── -->
<text x="{P1_CX:.1f}" y="{TITLE_Y}" text-anchor="middle" font-size="16" font-weight="bold" font-family="{FONT}" fill="#111">Prefill &#x2014; Apple M4 Pro</text>
<text transform="rotate(-90)" x="{YLABEL_MID:.1f}" y="16" text-anchor="middle" font-size="13" font-family="{FONT}" fill="#111">tok/s</text>

{make_grids(GRIDS_1, P_YMAX_1, P_Y0, P_Y1, P1_X0, P1_X1)}
{axes(P_Y0, P_Y1, P1_X0, P1_X1)}
{make_bars(0, P_YMAX_1, P_Y0, P_Y1, P1_X0)}
{make_xlabels(P_Y1 + 18, P1_X0)}

<!-- ── PANEL 2: Decode ── -->
<text x="{P2_CX:.1f}" y="{TITLE_Y}" text-anchor="middle" font-size="16" font-weight="bold" font-family="{FONT}" fill="#111">Decode &#x2014; Apple M4 Pro</text>
<text transform="rotate(-90)" x="{YLABEL_MID:.1f}" y="{P2_X0 - 22}" text-anchor="middle" font-size="13" font-family="{FONT}" fill="#111">tok/s</text>

{make_grids(GRIDS_2, P_YMAX_2, P_Y0, P_Y1, P2_X0, P2_X1)}
{axes(P_Y0, P_Y1, P2_X0, P2_X1)}
{make_bars(1, P_YMAX_2, P_Y0, P_Y1, P2_X0)}
{make_xlabels(P_Y1 + 18, P2_X0)}
</svg>'''

with open(OUT, 'w') as f:
    f.write(svg)
print(f"Written {OUT}")
