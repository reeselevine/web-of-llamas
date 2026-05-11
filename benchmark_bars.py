import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


devices = [
    "NVIDIA RTX 5080",
    "AMD 7900 XT",
    "Intel B580",
    "Apple M4 Pro",
]

device_order = [0, 2, 1, 3]

series = [
    ("q4_k_m", "Native"),
    ("q4_k_m", "Vulkan"),
    ("q4_k_m", "WebGPU No Checks"),
    ("q4_k_m", "WebGPU Checks"),
    ("f16", "Native"),
    ("f16", "Vulkan"),
    ("f16", "WebGPU No Checks"),
    ("f16", "WebGPU Checks"),
]

metrics = {
    "prefill": {
        "title": "Llama 3.2 1B Prefill Throughput",
        "ylabel": "Tokens / second",
        "values": [
            [32792.0, 4747.75, 13085.22, 3006.8],
            [26448.0, 4421.8, 1378.11, np.nan],
            [3466.0, 2970.39, 4123.03, 1279.83],
            [3117.0, 2356.82, 4212.83, 1045.24],
            [30120.0, 4408.21, 13185.3, 3248.96],
            [25011.0, 4509.44, 1297.01, np.nan],
            [7848.0, 5426.72, 6026.94, 2043.31],
            [5516.0, 4349.57, 6052.34, 1634.91],
        ],
        "output_stem": "llama32_1b_prefill_bars",
    },
    "decode": {
        "title": "Llama 3.2 1B Decode Throughput",
        "ylabel": "Tokens / second",
        "values": [
            [683.1, 122.31, 305.85, 210.88],
            [662.1, 120.0, 423.19, np.nan],
            [266.0, 125.49, 272.41, 120.67],
            [264.6, 114.26, 265.04, 112.92],
            [309.7, 91.29, 213.2, 88.7],
            [293.9, 143.01, 182.27, np.nan],
            [196.2, 100.82, 181.52, 76.97],
            [195.0, 97.35, 180.08, 77.62],
        ],
        "output_stem": "llama32_1b_decode_bars",
    },
}

colors = {
    "Native": "#7da9d8",
    "Vulkan": "#f2b37e",
    "WebGPU No Checks": "#8cc89a",
    "WebGPU Checks": "#d8a6cf",
}

hatches = {
    "q4_k_m": "",
    "f16": "//",
}

legend_labels = {
    ("q4_k_m", "Native"): "Q4_K_M Native",
    ("q4_k_m", "Vulkan"): "Q4_K_M Vulkan",
    ("q4_k_m", "WebGPU No Checks"): "Q4_K_M WebGPU No Checks",
    ("q4_k_m", "WebGPU Checks"): "Q4_K_M WebGPU Checks",
    ("f16", "Native"): "f16 Native",
    ("f16", "Vulkan"): "f16 Vulkan",
    ("f16", "WebGPU No Checks"): "f16 WebGPU No Checks",
    ("f16", "WebGPU Checks"): "f16 WebGPU Checks",
}


def format_tokens_per_second(value, _):
    if value >= 1000:
        if value % 1000 == 0:
            return f"{int(value / 1000)}k"
        return f"{value / 1000:.1f}k"
    if value == int(value):
        return str(int(value))
    return f"{value:.0f}"


def plot_metric(metric):
    x = np.arange(len(devices))
    bar_width = 0.095
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(10.5, 4.6))

    max_value = np.nanmax(metric["values"])
    na_y = max_value * 0.015

    for idx, ((quant, backend), values) in enumerate(zip(series, metric["values"])):
        values = np.array(values, dtype=float)[device_order]
        valid = ~np.isnan(values)

        ax.bar(
            x[valid] + offsets[idx],
            values[valid],
            width=bar_width,
            color=colors[backend],
            hatch=hatches[quant],
            edgecolor="#2f2f2f",
            linewidth=0.7,
            label=legend_labels[(quant, backend)],
        )

        for missing_x in x[~valid]:
            ax.text(
                missing_x + offsets[idx],
                na_y,
                "N/A",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#666666",
            )

    ax.set_title(metric["title"], pad=20)
    ax.set_ylabel(metric["ylabel"], fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(devices, fontsize=12)
    ax.set_ylim(0, max_value * 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(format_tokens_per_second))
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    ax.legend(
        loc="center right",
        bbox_to_anchor=(0.98, 0.75),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#cfcfcf",
        framealpha=0.95,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{metric['output_stem']}.pdf", bbox_inches="tight")


def main():
    for metric in metrics.values():
        plot_metric(metric)

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
