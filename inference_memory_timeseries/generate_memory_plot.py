#!/usr/bin/env python3
"""
Generate 2x2 memory comparison figure using matplotlib.
Usage: python3 generate_memory_plot.py
Output: memory_comparison.pdf
"""
import csv
import itertools
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

OUT = "memory_comparison.pdf"

# ── Paths ─────────────────────────────────────────────────────────────────────
M4C_TJS    = 'apple_m4_pro_chrome_memory_timeseries/transformers_fp16.csv'
M4C_WEBLLM = 'apple_m4_pro_chrome_memory_timeseries/webllm_fp16.csv'
M4C_WLLAMA = 'apple_m4_pro_chrome_memory_timeseries/wllama_fp16.csv'
M4S_TJS    = 'apple_m4_pro_safari_memory_timeseries/transformers_fp16.csv'  # total_mb (OOM)
M4S_WEBLLM = 'apple_m4_pro_safari_memory_timeseries/webllm_fp16.csv'          # total_mb
M4S_WLLAMA = 'apple_m4_pro_safari_memory_timeseries/wllama_fp16.csv'          # total_mb
NV_TJS     = 'linux_nvidia_rtx5080_chrome_memory_timeseries/5080_linux_transformers_fp16.csv'
NV_WEBLLM  = 'linux_nvidia_rtx5080_chrome_memory_timeseries/5080_linux_webllm_q0f16.csv'
NV_WLLAMA  = 'linux_nvidia_rtx5080_chrome_memory_timeseries/5080_linux_wllama_fp16.csv'
WIN_TJS    = 'windows_nvidia_rtx5080_chrome_timeseries/transformers_fp16_win.csv'
WIN_WEBLLM = 'windows_nvidia_rtx5080_chrome_timeseries/webllm_q0f16_win.csv'
WIN_WLLAMA = 'windows_nvidia_rtx5080_chrome_timeseries/wllama_fp16_win.csv'

TMAX = 30

# ── Colors ────────────────────────────────────────────────────────────────────
WLLAMA = "#009E73"
WEBLLM = "#56B4E9"
TJS    = "#E69F00"
OOM_C  = "#D55E00"

# ── Data loading ──────────────────────────────────────────────────────────────
def read_csv(path, mem_col='combined_mb', tmax=TMAX, trim_start=None, trim_to_steady=False):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            t = datetime.fromisoformat(r['timestamp'])
            rows.append((t, float(r[mem_col])))
    if trim_start is not None:
        rows = list(itertools.dropwhile(lambda x: x[1] < trim_start, rows))
    if trim_to_steady:
        peak_idx = max(range(len(rows)), key=lambda i: rows[i][1])
        rows = rows[peak_idx + 1:]
    t0 = rows[0][0]
    return [((t - t0).total_seconds(), m) for t, m in rows
            if (t - t0).total_seconds() <= tmax]

def read_csv_windows(path, tmax=TMAX, trim_start=None, trim_to_steady=False, start_time=None):
    """Windows CSV: combined_mb = sum(*_priv_mb) + gpu_vram_mb."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            t = datetime.fromisoformat(r['timestamp'])
            priv = sum(float(v) for k, v in r.items() if k.endswith('_priv_mb'))
            combined = priv + float(r['gpu_vram_mb'])
            rows.append((t, combined))
    if start_time is not None:
        cutoff = datetime.fromisoformat(start_time)
        rows = [(t, m) for t, m in rows if t >= cutoff]
    if trim_start is not None:
        rows = list(itertools.dropwhile(lambda x: x[1] < trim_start, rows))
    if trim_to_steady:
        peak_idx = max(range(len(rows)), key=lambda i: rows[i][1])
        rows = rows[peak_idx + 1:]
    t0 = rows[0][0]
    return [((t - t0).total_seconds(), m) for t, m in rows
            if (t - t0).total_seconds() <= tmax]

def split_oom(data, threshold=100):
    oom_t, valid = None, []
    for t, m in data:
        if m < threshold:
            if oom_t is None: oom_t = t
        else:
            valid.append((t, m))
    return valid, oom_t

# ── Load data ─────────────────────────────────────────────────────────────────
m4c_tjs    = read_csv(M4C_TJS)
m4c_webllm = read_csv(M4C_WEBLLM)
m4c_wllama = read_csv(M4C_WLLAMA)
m4s_tjs_raw = read_csv(M4S_TJS, mem_col='total_mb')
m4s_webllm  = read_csv(M4S_WEBLLM, mem_col='total_mb')
m4s_wllama  = read_csv(M4S_WLLAMA, mem_col='total_mb')
m4s_tjs, _ = split_oom(m4s_tjs_raw)
oom_pt = m4s_tjs[-1] if m4s_tjs else None
nv_tjs     = read_csv(NV_TJS, trim_start=6300)
nv_webllm  = read_csv(NV_WEBLLM, trim_start=6050)
nv_wllama  = read_csv(NV_WLLAMA, trim_start=400, trim_to_steady=True)
win_tjs    = read_csv_windows(WIN_TJS)
win_webllm = read_csv_windows(WIN_WEBLLM, start_time='2026-05-12 14:19:21')
win_wllama = read_csv_windows(WIN_WLLAMA)


# ── Helpers ───────────────────────────────────────────────────────────────────
def unzip(data):
    return [t for t, m in data], [m for t, m in data]

def fmt_mb(v, _):
    if v >= 1000:
        return f"{int(v / 1000)}k" if v % 1000 == 0 else f"{v / 1000:.0f}k"
    return str(int(v))

def style_ax(ax, title, ylim, ylabel=False, tmax=TMAX):
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, ylim)
    ticks = [0, 10, 20, 30]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}s" for t in ticks], fontsize=22)
    ax.tick_params(axis="y", labelsize=22)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_mb))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=24, fontweight="bold")
    if ylabel:
        ax.set_ylabel("Memory (MB)", fontsize=24)

def plot_lines(ax, datasets, lw=2.0):
    for data, color in datasets:
        ts, ms = unzip(data)
        ax.plot(ts, ms, color=color, linewidth=lw)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# Top row — Apple M4 Pro (ylim 10k)
plot_lines(axes[0, 0], [(m4c_wllama, WLLAMA), (m4c_webllm, WEBLLM), (m4c_tjs, TJS)])
style_ax(axes[0, 0], "Apple M4 Pro\nChrome", ylim=10000, ylabel=True)

plot_lines(axes[0, 1], [(m4s_wllama, WLLAMA), (m4s_webllm, WEBLLM), (m4s_tjs, TJS)])
style_ax(axes[0, 1], "Apple M4 Pro\nSafari", ylim=10000)
if oom_pt:
    tx, ty = oom_pt
    axes[0, 1].plot(tx, ty, 'x', color=OOM_C, markersize=10, markeredgewidth=2.5, zorder=5)
    axes[0, 1].annotate("OOM", (tx, ty), xytext=(6, 0),
                         textcoords="offset points",
                         va="center", fontsize=20,
                         color=OOM_C, fontweight="bold")

# Bottom row — NVIDIA RTX 5080 (ylim 12k)
plot_lines(axes[1, 0], [(nv_wllama, WLLAMA), (nv_webllm, WEBLLM), (nv_tjs, TJS)])
style_ax(axes[1, 0], "NVIDIA RTX 5080 (Linux)\nChrome", ylim=12000, ylabel=True)

plot_lines(axes[1, 1], [(win_wllama, WLLAMA), (win_webllm, WEBLLM), (win_tjs, TJS)])
style_ax(axes[1, 1], "NVIDIA RTX 5080 (Windows)\nChrome", ylim=12000)

# Shared x-axis label
fig.text(0.5, 0.0, "Elapsed time (s)", ha="center", fontsize=24)

# Legend above all panels
legend_handles = [
    Line2D([0], [0], color=WLLAMA, linewidth=2.5, label="wllama"),
    Line2D([0], [0], color=WEBLLM, linewidth=2.5, label="WebLLM"),
    Line2D([0], [0], color=TJS,    linewidth=2.5, label="Transformers.js"),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.0),
    frameon=True, facecolor="white",
    edgecolor="#cfcfcf", framealpha=0.95,
    fontsize=20,
)

fig.tight_layout(rect=[0, 0.04, 1, 0.93])
fig.savefig(OUT, bbox_inches="tight")
print(f"Written {OUT}")
