#!/usr/bin/env python3
"""
Generate 2x2 memory comparison SVG for the wllama paper.
Usage: python3 generate_memory_plot.py
Outputs: memory_comparison.svg
"""
import csv
import itertools
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
M4C_TJS    = 'apple_m4_pro_chrome_memory_timeseries/m4_chrome_transformers_fp16.csv'
M4C_WEBLLM = 'apple_m4_pro_chrome_memory_timeseries/m4_chrome_webllm_fp16.csv'
M4C_WLLAMA = 'apple_m4_pro_chrome_memory_timeseries/m4_chrome_wllama_fp16.csv'
M4S_TJS    = 'apple_m4_pro_safari_memory_timeseries/m4_safari_transformers_fp16.csv'   # total_mb
M4S_WEBLLM = 'apple_m4_pro_safari_memory_timeseries/m4_safari_webllm_fp16.csv'         # total_mb
M4S_WLLAMA = 'apple_m4_pro_safari_memory_timeseries/m4_safari_wllama_fp16.csv'         # total_mb
NV_TJS     = 'linux_nvidia_rtx5080_chrome_memory_timeseries/5080_linux_transformers_q4f16_csv.csv'
NV_WEBLLM  = 'linux_nvidia_rtx5080_chrome_memory_timeseries/5080_linux_webllm_q4f16_1.csv'
NV_WLLAMA  = 'linux_nvidia_rtx5080_chrome_memory_timeseries/5080_linux_wllama_q4_k_m.csv'
WIN_TJS    = 'windows_nvidia_rtx5080_chrome_timeseries/transformers_fp16_win.csv'
WIN_WEBLLM = 'windows_nvidia_rtx5080_chrome_timeseries/webllm_q0f16_win.csv'
WIN_WLLAMA = 'windows_nvidia_rtx5080_chrome_timeseries/wllama_fp16_win.csv'
OUT        = 'memory_comparison.svg'

TMAX     = 15    # seconds to plot (all panels except Windows)
WIN_TMAX = 288   # full recording length for Windows panel

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

def read_csv_windows(path, tmax=TMAX, trim_start=None, trim_to_steady=False):
    """Windows CSV has per-process columns; combined_mb = sum(*_priv_mb) + gpu_vram_mb.
    trim_start: skip leading rows where combined_mb < trim_start before anchoring t=0.
    trim_to_steady: skip the load spike by starting just after the peak value."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            t = datetime.fromisoformat(r['timestamp'])
            priv = sum(float(v) for k, v in r.items() if k.endswith('_priv_mb'))
            combined = priv + float(r['gpu_vram_mb'])
            rows.append((t, combined))
    if trim_start is not None:
        rows = list(itertools.dropwhile(lambda x: x[1] < trim_start, rows))
    if trim_to_steady:
        peak_idx = max(range(len(rows)), key=lambda i: rows[i][1])
        rows = rows[peak_idx + 1:]
    t0 = rows[0][0]
    return [((t-t0).total_seconds(), m) for t, m in rows
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
win_tjs    = read_csv_windows(WIN_TJS)
win_webllm = read_csv_windows(WIN_WEBLLM, trim_start=1000, trim_to_steady=True)
win_wllama = read_csv_windows(WIN_WLLAMA)

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
LEG_H    = 26
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

YMAX   = 12000
YGRID  = [(0,"0"),(2000,"2k"),(4000,"4k"),
           (6000,"6k"),(8000,"8k"),(10000,"10k"),(12000,"12k")]
XTICS     = [0, 5, 10, 15]
WIN_XTICS = [0, 60, 120, 180, 240, 288]

# ── SVG helpers ────────────────────────────────────────────────────────────────
def ymap(v, y0, y1): return y1 - (v/YMAX)*(y1-y0)
def xmap(t, x0, x1, tmax=TMAX): return x0 + (t/tmax)*(x1-x0)

def polyline(data, x0, x1, y0, y1, color, sw=2.2, tmax=TMAX):
    pts = []
    for t,m in data:
        x = xmap(t, x0, x1, tmax)
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

def make_xgrid(x0, x1, y0, y1, label_y, xtics=XTICS, tmax=TMAX):
    out = ""
    n = len(xtics)
    for i,t in enumerate(xtics):
        x = xmap(t, x0, x1, tmax)
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
LBX, LBW = LMARGIN, 360
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
{leg_item(LBX+6,  L1Y, WLLAMA, "wllama")}
{leg_item(LBX+100,L1Y, WEBLLM, "WebLLM")}
{leg_item(LBX+210,L1Y, TJS,    "Transformers.js")}

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

<!-- BOTTOM RIGHT: RTX 5080 Windows Chrome -->
{title((RX0+RX1)/2, BOT_TTL, "NVIDIA RTX 5080 (Windows) / Chrome")}
{make_ygrid(RX0,RX1,BOT_Y0,BOT_Y1,False)}
{make_xgrid(RX0,RX1,BOT_Y0,BOT_Y1,BOT_XLY)}
{axes(RX0,RX1,BOT_Y0,BOT_Y1)}
{polyline(win_tjs,    RX0,RX1,BOT_Y0,BOT_Y1,TJS)}
{polyline(win_webllm, RX0,RX1,BOT_Y0,BOT_Y1,WEBLLM)}
{polyline(win_wllama, RX0,RX1,BOT_Y0,BOT_Y1,WLLAMA)}

<!-- X axis label -->
<text x="{(LX0+RX1)/2:.0f}" y="{BOT_XLY+14}" text-anchor="middle" font-size="13"
      font-family="{FONT}" fill="#111" font-weight="bold">Elapsed time (s)</text>
</svg>'''

with open(OUT, 'w') as f:
    f.write(svg)
print(f"Written {OUT}  ({W}x{H}px)")
