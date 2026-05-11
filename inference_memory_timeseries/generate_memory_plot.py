#!/usr/bin/env python3
"""
Generate 2x2 memory comparison SVG for the wllama paper.
Usage: python3 generate_memory_plot.py
Outputs: memory_comparison.svg

Data files expected (edit paths below):
  M4 Chrome  : transformers_js q4f16, webllm q4f16_1, wllama q4_k_m  (combined_mb)
  M4 Safari  : transformers_js q4,    webllm q4f16_1, wllama q4_k_m  (total_mb)
  NV Linux   : transformers_js q4f16, webllm q4f16_1, wllama q4_k_m  (combined_mb)
  NV Windows : forthcoming
"""
import csv
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
M4C_TJS    = 'transformers_js-llama-3_2-1b-instruct-q4_f16-onnx.csv'
M4C_WEBLLM = 'webllm-chrome-llama-3_2-1b-instruct-q4_1f16.csv'
M4C_WLLAMA = 'wllama_chrome_llama-3_2-1b-qkm.csv'
M4S_TJS    = 'transformers_js-llama-3_2-1b-instruct-onnx.csv'       # total_mb
M4S_WEBLLM = 'webllm-safari-llama-3_2-1b-instruct-q4_1f16.csv'      # total_mb
M4S_WLLAMA = 'wllama_safari_llama-3_2-1b-qkm.csv'                    # total_mb
NV_TJS     = '5080_linux_transformers_q4f16_csv.csv'
NV_WEBLLM  = '5080_linux_webllm_q4f16_1.csv'
NV_WLLAMA  = '5080_linux_wllama_q4_k_m.csv'
OUT        = 'memory_comparison.svg'

TMAX = 15   # seconds to plot

# ── Helpers ────────────────────────────────────────────────────────────────────
def read_csv(path, mem_col='combined_mb', tmax=TMAX):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            t = datetime.fromisoformat(r['timestamp'])
            rows.append((t, float(r[mem_col])))
    t0 = rows[0][0]
    return [((t-t0).total_seconds(), m) for t,m in rows
            if (t-t0).total_seconds() <= tmax]

def split_oom(data, threshold=100):
    """Split safari data into valid points and OOM start time."""
    oom_t, valid = None, []
    for t,m in data:
        if m < threshold:
            if oom_t is None: oom_t = t
        else:
            valid.append((t,m))
    return valid, oom_t

# ── Load data ──────────────────────────────────────────────────────────────────
m4c_tjs    = read_csv(M4C_TJS)
m4c_webllm = read_csv(M4C_WEBLLM)
m4c_wllama = read_csv(M4C_WLLAMA)
m4s_tjs_raw= read_csv(M4S_TJS, mem_col='total_mb')
m4s_webllm = read_csv(M4S_WEBLLM, mem_col='total_mb')
m4s_wllama = read_csv(M4S_WLLAMA, mem_col='total_mb')
nv_tjs     = read_csv(NV_TJS)
nv_webllm  = read_csv(NV_WEBLLM)
nv_wllama  = read_csv(NV_WLLAMA)

m4s_tjs, oom_t = split_oom(m4s_tjs_raw)

# ── Style ──────────────────────────────────────────────────────────────────────
FONT   = "Georgia, 'Times New Roman', Times, serif"
TJS    = "#E69F00"   # orange      (colorblind-safe, Wong 2011)
WEBLLM = "#56B4E9"   # sky blue
WLLAMA = "#009E73"   # bluish green
OOM_C  = "#D55E00"   # vermillion

# ── Layout ─────────────────────────────────────────────────────────────────────
W        = 680
LMARGIN  = 68    # room for y-tick labels
RMARGIN  = 10
HGAP     = 24    # gap between left and right panels
PW       = (W - LMARGIN - RMARGIN - HGAP) / 2   # panel width

LEG_Y0   = 8
LEG_H    = 54
LEG_Y1   = LEG_Y0 + LEG_H        # 62

TOP_TTL  = LEG_Y1 + 18           # 80  panel title baseline
TOP_Y0   = TOP_TTL + 8           # 88  plot top
TOP_Y1   = TOP_Y0 + 210          # 298 plot bottom
TOP_XLY  = TOP_Y1 + 16           # 314 x-tick labels

ROW_GAP  = 54
BOT_TTL  = TOP_Y1 + ROW_GAP - 14 # 338 bottom panel title baseline
BOT_Y0   = TOP_Y1 + ROW_GAP      # 352 bottom plot top
BOT_Y1   = BOT_Y0 + 210          # 562 bottom plot bottom
BOT_XLY  = BOT_Y1 + 16           # 578 x-tick labels

H        = BOT_XLY + 22          # total height

LX0 = LMARGIN
LX1 = LMARGIN + PW
RX0 = LMARGIN + PW + HGAP
RX1 = LMARGIN + 2*PW + HGAP

YMAX   = 10000
YGRID  = [(0,"0"),(2000,"2,000"),(4000,"4,000"),
           (6000,"6,000"),(8000,"8,000"),(10000,"10,000")]
XTICS  = [0, 5, 10, 15]

# ── SVG helpers ────────────────────────────────────────────────────────────────
def ymap(v, y0, y1): return y1 - (v/YMAX)*(y1-y0)
def xmap(t, x0, x1): return x0 + (t/TMAX)*(x1-x0)

def polyline(data, x0, x1, y0, y1, color, sw=2.2):
    pts = []
    for t,m in data:
        x = xmap(t, x0, x1)
        y = max(y0, min(y1, ymap(m, y0, y1)))
        pts.append(f"{x:.1f},{y:.1f}")
    return (f'<polyline fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linejoin="round" points="{" ".join(pts)}"/>\n')

def make_ygrid(x0, x1, y0, y1, labels):
    out = ""
    for v, lbl in YGRID:
        y = ymap(v, y0, y1)
        out += (f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                f'stroke="#ccc" stroke-width="0.8" stroke-dasharray="4,3"/>\n')
        if labels:
            out += (f'<text x="{x0-5}" y="{y+4:.1f}" text-anchor="end" '
                    f'font-size="12" fill="#111" font-family="{FONT}">{lbl}</text>\n')
    return out

def make_xgrid(x0, x1, y0, y1, label_y):
    out = ""
    n = len(XTICS)
    for i,t in enumerate(XTICS):
        x = xmap(t, x0, x1)
        out += (f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" '
                f'stroke="#ccc" stroke-width="0.8" stroke-dasharray="4,3"/>\n')
        anc = "start" if i==0 else ("end" if i==n-1 else "middle")
        out += (f'<text x="{x:.1f}" y="{label_y}" text-anchor="{anc}" '
                f'font-size="12" fill="#111" font-family="{FONT}">{t}s</text>\n')
    return out

def axes(x0, x1, y0, y1):
    return (f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#111" stroke-width="1.5"/>\n'
            f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#111" stroke-width="1.5"/>\n')

def title(cx, y, text):
    return (f'<text x="{cx:.0f}" y="{y}" text-anchor="middle" font-size="14" '
            f'font-weight="bold" font-family="{FONT}" fill="#111">{text}</text>\n')

def ylabel(cy):
    return (f'<text transform="rotate(-90)" x="{-cy:.0f}" y="15" text-anchor="middle" '
            f'font-size="13" font-family="{FONT}" fill="#111" font-weight="bold">Memory (MB)</text>\n')

def oom_marker(x, y0):
    return (f'<line x1="{x-7:.1f}" y1="{y0+4}" x2="{x+7:.1f}" y2="{y0+18}" '
            f'stroke="{OOM_C}" stroke-width="2.8"/>\n'
            f'<line x1="{x+7:.1f}" y1="{y0+4}" x2="{x-7:.1f}" y2="{y0+18}" '
            f'stroke="{OOM_C}" stroke-width="2.8"/>\n'
            f'<text x="{x+11:.1f}" y="{y0+18}" font-size="13" fill="{OOM_C}" '
            f'font-weight="bold" font-family="{FONT}">OOM</text>\n')

# ── Legend ─────────────────────────────────────────────────────────────────────
LBX, LBW = LMARGIN, 335
L1Y = LEG_Y0 + 18
L2Y = LEG_Y0 + 40

def leg_item(lx, ly, color, label, dash=False):
    d = f' stroke-dasharray="6,3"' if dash else ''
    return (f'<line x1="{lx}" y1="{ly}" x2="{lx+16}" y2="{ly}" '
            f'stroke="{color}" stroke-width="3"{d}/>\n'
            f'<text x="{lx+20}" y="{ly+4}" font-size="13" font-family="{FONT}" '
            f'fill="#111" font-weight="bold">{label}</text>\n')

oom_x_safari = xmap(oom_t, RX0, RX1)

# ── Build SVG ──────────────────────────────────────────────────────────────────
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<rect width="100%" height="100%" fill="white"/>

<!-- Legend box -->
<rect x="{LBX}" y="{LEG_Y0}" width="{LBW}" height="{LEG_H}" fill="white" stroke="#888" stroke-width="1" rx="3"/>
{leg_item(LBX+6,  L1Y, TJS,    "Transformers.js")}
{leg_item(LBX+140,L1Y, WEBLLM, "WebLLM")}
{leg_item(LBX+220,L1Y, WLLAMA, "wllama")}
<line x1="{LBX+6}"  y1="{L2Y}"   x2="{LBX+22}"  y2="{L2Y}"   stroke="{OOM_C}" stroke-width="2.5"/>
<line x1="{LBX+11}" y1="{L2Y-6}" x2="{LBX+17}"  y2="{L2Y+6}" stroke="{OOM_C}" stroke-width="2.5"/>
<line x1="{LBX+17}" y1="{L2Y-6}" x2="{LBX+11}"  y2="{L2Y+6}" stroke="{OOM_C}" stroke-width="2.5"/>
<text x="{LBX+26}" y="{L2Y+4}" font-size="13" font-family="{FONT}" fill="#111" font-weight="bold">Out of Memory (OOM)</text>

<!-- Y labels -->
{ylabel((TOP_Y0+TOP_Y1)/2)}
{ylabel((BOT_Y0+BOT_Y1)/2)}

<!-- TOP LEFT: M4 Pro Chrome -->
{title((LX0+LX1)/2, TOP_TTL, "Apple M4 Pro / Chrome")}
{make_ygrid(LX0,LX1,TOP_Y0,TOP_Y1,True)}
{make_xgrid(LX0,LX1,TOP_Y0,TOP_Y1,TOP_XLY)}
{axes(LX0,LX1,TOP_Y0,TOP_Y1)}
{polyline(m4c_tjs,    LX0,LX1,TOP_Y0,TOP_Y1,TJS)}
{polyline(m4c_webllm, LX0,LX1,TOP_Y0,TOP_Y1,WEBLLM)}
{polyline(m4c_wllama, LX0,LX1,TOP_Y0,TOP_Y1,WLLAMA)}

<!-- TOP RIGHT: M4 Pro Safari -->
{title((RX0+RX1)/2, TOP_TTL, "Apple M4 Pro / Safari")}
{make_ygrid(RX0,RX1,TOP_Y0,TOP_Y1,False)}
{make_xgrid(RX0,RX1,TOP_Y0,TOP_Y1,TOP_XLY)}
{axes(RX0,RX1,TOP_Y0,TOP_Y1)}
{polyline(m4s_tjs,    RX0,RX1,TOP_Y0,TOP_Y1,TJS)}
{polyline(m4s_webllm, RX0,RX1,TOP_Y0,TOP_Y1,WEBLLM)}
{polyline(m4s_wllama, RX0,RX1,TOP_Y0,TOP_Y1,WLLAMA)}
{oom_marker(oom_x_safari, TOP_Y0)}

<!-- BOTTOM LEFT: RTX 5080 Linux Chrome -->
{title((LX0+LX1)/2, BOT_TTL, "NVIDIA RTX 5080 (Linux) / Chrome")}
{make_ygrid(LX0,LX1,BOT_Y0,BOT_Y1,True)}
{make_xgrid(LX0,LX1,BOT_Y0,BOT_Y1,BOT_XLY)}
{axes(LX0,LX1,BOT_Y0,BOT_Y1)}
{polyline(nv_tjs,    LX0,LX1,BOT_Y0,BOT_Y1,TJS)}
{polyline(nv_webllm, LX0,LX1,BOT_Y0,BOT_Y1,WEBLLM)}
{polyline(nv_wllama, LX0,LX1,BOT_Y0,BOT_Y1,WLLAMA)}

<!-- BOTTOM RIGHT: RTX 5080 Windows (placeholder) -->
{title((RX0+RX1)/2, BOT_TTL, "NVIDIA RTX 5080 (Windows) / Chrome")}
{make_ygrid(RX0,RX1,BOT_Y0,BOT_Y1,False)}
{make_xgrid(RX0,RX1,BOT_Y0,BOT_Y1,BOT_XLY)}
{axes(RX0,RX1,BOT_Y0,BOT_Y1)}
<text x="{(RX0+RX1)/2:.0f}" y="{(BOT_Y0+BOT_Y1)/2:.0f}" text-anchor="middle"
      font-size="13" fill="#aaa" font-family="{FONT}" font-style="italic">Data forthcoming</text>

<!-- X axis label -->
<text x="{(LX0+RX1)/2:.0f}" y="{BOT_XLY+14}" text-anchor="middle" font-size="13"
      font-family="{FONT}" fill="#111" font-weight="bold">Elapsed time (s)</text>
</svg>'''

with open(OUT, 'w') as f:
    f.write(svg)
print(f"Written {OUT}  ({W}x{H}px)")
