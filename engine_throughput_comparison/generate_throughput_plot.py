#!/usr/bin/env python3
"""
Generate separate throughput comparison figures for prefill and decode.

Usage: python3 generate_throughput_plot.py
Outputs: engine_throughput_prefill.pdf, engine_throughput_decode.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_FILES = {
    "prefill": "engine_throughput_prefill.pdf",
    "decode": "engine_throughput_decode.pdf",
}

# ── Colors (Wong colorblind-safe) ──────────────────────────────────────────────
WLLAMA  = "#009E73"
WEBLLM  = "#56B4E9"
TJS     = "#E69F00"
POS_PCT = "#1a5c35"
NEG_PCT = "#7a1515"

# ── Data ──────────────────────────────────────────────────────────────────────
# (label, wllama_pre, wllama_dec, webllm_pre, webllm_dec, tjs_pre, tjs_dec)
groups = [
    ("NVIDIA\nRTX 5080",  1300.8,  96.6,  2759.9, 50.7, 1763.2,  41.7),
    ("AMD\nRX 7900 XT",  1228.8,  71.3,  2552.8, 35.9, 1343.2,  43.2),
    ("Intel\nArc B580",  644.9,  42.1,  1467, 40.1, 1205.5,  38.5),
    ("Apple\nM4 Pro",         1010.5, 74.2,  1828.2, 51.2,  955.3,  38.3)
    # ("RTX 5080\nLinux",    1170.9, 103.1,  1922.2, 69.2, 1534.0,  66.2),
]
plot_groups = groups

engines = ["wllama", "WebLLM", "Transformers.js"]
clrs    = [WLLAMA, WEBLLM, TJS]

# ── Y-axis formatter ──────────────────────────────────────────────────────────
def fmt_tps(v, _):
    if v >= 1000:
        return f"{int(v / 1000)}k" if v % 1000 == 0 else f"{v / 1000:.1f}k"
    return f"{int(v)}" if v == int(v) else f"{v:.0f}"

def plot_metric(kind, pidx):
    fig, ax = plt.subplots(figsize=(10, 5.1))

    x = np.arange(len(plot_groups))
    NB = 3
    bar_w = 0.22
    offs = (np.arange(NB) - (NB - 1) / 2) * bar_w

    ax.set_ylabel("Tokens / second", fontsize=20)
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

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in plot_groups], fontsize=18, rotation=0, ha='center')
    ax.tick_params(axis="y", labelsize=18)
    ax.set_xlim(-0.6, len(plot_groups) - 0.4)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_tps))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    top_margin = 0.9
    if kind == "prefill":
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.0),
            frameon=True, facecolor="white",
            edgecolor="#cfcfcf", framealpha=0.95,
            fontsize=18,
        )
        top_margin = 0.88

    fig.tight_layout(rect=[0, 0, 1, top_margin])
    out = OUT_FILES[kind]
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Written {out}")


plot_metric("prefill", 0)
plot_metric("decode", 1)
