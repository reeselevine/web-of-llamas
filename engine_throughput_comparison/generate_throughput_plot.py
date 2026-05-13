#!/usr/bin/env python3
"""
Generate 2-panel throughput comparison figure (Prefill / Decode) — Apple M4 Pro.
Groups shown: Chrome/fp16, Safari/fp16  (q4 data kept but not plotted)
Engines: wllama, WebLLM, Transformers.js

Usage: python3 generate_throughput_plot.py
Output: engine_throughput_comparison.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT = "engine_throughput_comparison.pdf"

# ── Colors (Wong colorblind-safe) ──────────────────────────────────────────────
WLLAMA  = "#009E73"
WEBLLM  = "#56B4E9"
TJS     = "#E69F00"
POS_PCT = "#1a5c35"
NEG_PCT = "#7a1515"

# ── Data ──────────────────────────────────────────────────────────────────────
# (label, wllama_pre, wllama_dec, webllm_pre, webllm_dec, tjs_pre, tjs_dec)
groups = [
    ("Chrome / fp16", 1010.5, 74.2,  1828.2, 51.2,  955.3,  38.3),
    ("Safari / fp16", 167.5,  60.2,  588.4,  43.1,  732.9,  22.2),
    ("Chrome / q4",   611.5,  110.0, 1832.3, 68.1,  959.3,  124.4),
    ("Safari / q4",   152.5,  63.0,  501.5,  62.0,  634.8,  50.3),
]
plot_groups = [g for g in groups if "fp16" in g[0]]

engines = ["wllama", "WebLLM", "Transformers.js"]
clrs    = [WLLAMA, WEBLLM, TJS]

# ── Y-axis formatter ──────────────────────────────────────────────────────────
def fmt_tps(v, _):
    if v >= 1000:
        return f"{int(v / 1000)}k" if v % 1000 == 0 else f"{v / 1000:.1f}k"
    return f"{int(v)}" if v == int(v) else f"{v:.0f}"

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

x     = np.arange(len(plot_groups))
NB    = 3
bar_w = 0.22
offs  = (np.arange(NB) - (NB - 1) / 2) * (bar_w + 0.03)

for ax, pidx, title in [
    (axes[0], 0, "Prefill  —  Apple M4 Pro"),
    (axes[1], 1, "Decode  —  Apple M4 Pro"),
]:
    # engine_vals[bi][gi] = value for engine bi, group gi
    engine_vals = [
        [row[1 + pidx + 2 * bi] for row in plot_groups]
        for bi in range(NB)
    ]

    for bi, (engine, color, ev) in enumerate(zip(engines, clrs, engine_vals)):
        ax.bar(
            x + offs[bi], ev,
            width=bar_w, color=color,
            edgecolor="#2f2f2f", linewidth=0.6,
            label=engine, zorder=3,
        )

    # set ylim before placing % labels so headroom is correct
    data_max = max(v for ev in engine_vals for v in ev)
    ax.set_ylim(0, data_max * 1.25)

    # % labels vs wllama (bi=0)
    for gi in range(len(plot_groups)):
        base = engine_vals[0][gi]
        label_ys = {}
        for bi in (1, 2):
            val = engine_vals[bi][gi]
            label_ys[bi] = val + data_max * 0.008

        # nudge apart if too close in y
        MIN_GAP = data_max * 0.06
        if abs(label_ys[1] - label_ys[2]) < MIN_GAP:
            mid = (label_ys[1] + label_ys[2]) / 2
            label_ys[1] = mid - MIN_GAP / 2
            label_ys[2] = mid + MIN_GAP / 2

        for bi in (1, 2):
            val  = engine_vals[bi][gi]
            pct  = (val - base) / base * 100
            sign = "+" if pct >= 0 else ""
            col  = POS_PCT if pct >= 0 else NEG_PCT
            ax.text(
                x[gi] + offs[bi], label_ys[bi],
                f"{sign}{pct:.0f}%",
                ha="center", va="bottom",
                fontsize=8.5, color=col, fontweight="bold", zorder=4,
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in plot_groups], fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("tok / s", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_tps))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend(
    loc="upper right",
    frameon=True, facecolor="white",
    edgecolor="#cfcfcf", framealpha=0.95,
    fontsize=11,
)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"Written {OUT}")
