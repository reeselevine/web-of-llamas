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
    ("Apple\nM4 Pro",         1010.5, 74.2,  1828.2, 51.2,  955.3,  38.3),
    # ("RTX 5080\nLinux",    1170.9, 103.1,  1922.2, 69.2, 1534.0,  66.2),
    ("NVIDIA\nRTX 5080",  1300.8,  96.6,  2759.9, 50.7, 1763.2,  41.7),
    ("Intel\nArc B580",  644.9,  42.1,  1467, 40.1, 1205.5,  38.5),
    ("AMD\nRX 9700 XT",  1228.8,  71.3,  2552.8, 35.9, 1343.2,  43.2)
]
plot_groups = groups

engines = ["wllama", "WebLLM", "Transformers.js"]
clrs    = [WLLAMA, WEBLLM, TJS]

# ── Y-axis formatter ──────────────────────────────────────────────────────────
def fmt_tps(v, _):
    if v >= 1000:
        return f"{int(v / 1000)}k" if v % 1000 == 0 else f"{v / 1000:.1f}k"
    return f"{int(v)}" if v == int(v) else f"{v:.0f}"

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 11))

x     = np.arange(len(plot_groups))
NB    = 3
bar_w = 0.22
offs  = (np.arange(NB) - (NB - 1) / 2) * (bar_w + 0.03)

for ax, pidx, title in [
    (axes[0], 0, "Prefill"),
    (axes[1], 1, "Decode"),
]:
    ax.set_ylabel("tokens / second", fontsize=30)
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
    ax.set_ylim(0, 3000 if pidx == 0 else 110)

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

        # assign higher label y to whichever engine has the higher bar
        ys = sorted([label_ys[1], label_ys[2]])
        if engine_vals[1][gi] >= engine_vals[2][gi]:
            label_ys[1], label_ys[2] = ys[1], ys[0]
        else:
            label_ys[1], label_ys[2] = ys[0], ys[1]

        for bi in (1, 2):
            val  = engine_vals[bi][gi]
            pct  = (val - base) / base * 100
            sign = "+" if pct >= 0 else ""
            col  = POS_PCT if pct >= 0 else NEG_PCT
            ax.text(
                x[gi] + offs[bi], label_ys[bi],
                f"{sign}{pct:.0f}%",
                ha="center", va="bottom",
                fontsize=17, color=col, fontweight="bold", zorder=4,
            )

    ax._pending_title = title
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in plot_groups], fontsize=25, rotation=0, ha='center')
    ax.tick_params(axis="y", labelsize=28)
    ax.set_xlim(-0.6, len(plot_groups) - 0.4)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_tps))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles, labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.0),
    frameon=True, facecolor="white",
    edgecolor="#cfcfcf", framealpha=0.95,
    fontsize=26,
)

fig.tight_layout(rect=[0, 0, 1, 0.88])

pos = axes[1].get_position()
axes[1].set_position([pos.x0, pos.y0 - 0.08, pos.width, pos.height])

for ax, y_off in zip(axes, [0.01, 0.01]):
    bbox = ax.get_position()
    fig.text(0.5, bbox.y1 + y_off, ax._pending_title,
             ha="center", va="bottom", fontsize=36, fontweight="bold",
             transform=fig.transFigure)

fig.savefig(OUT, bbox_inches="tight")
print(f"Written {OUT}")
