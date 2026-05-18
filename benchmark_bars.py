import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


devices = [
    "NVIDIA\nRTX 5080",
    "AMD\nRX 7900 XT",
    "Intel\nArc B580",
    "Apple\nM4 Pro",
]

device_order = [0, 2, 1, 3]

series = [
    ("q4_k_m", "Native"),
    ("q4_k_m", "Vulkan"),
    ("q4_k_m", "WebGPU w/o Checks"),
    ("q4_k_m", "WebGPU w/ Checks"),
    ("f16", "Native"),
    ("f16", "Vulkan"),
    ("f16", "WebGPU w/o Checks"),
    ("f16", "WebGPU w/ Checks"),
]

metrics = {
    "prefill": {
        "ylabel": "Tokens / second",
        "show_legend": True,
        "values": [
            [32792.0, 4747.75, 13085.22, 3006.8],
            [26448.0, 4421.8, 1378.11, 997.33],
            [3466.0, 2970.39, 4123.03, 1279.83],
            [3117.0, 2356.82, 4212.83, 1045.24],
            [30120.0, 4408.21, 13185.3, 3248.96],
            [25011.0, 4509.44, 1297.01, 1286.76],
            [7848.0, 5426.72, 6026.94, 2043.31],
            [5516.0, 4349.57, 6052.34, 1634.91],
        ],
        "output_stem": "llama32_1b_prefill_bars",
    },
    "decode": {
        "ylabel": "Tokens / second",
        "show_legend": False,
        "values": [
            [683.1, 122.31, 305.85, 210.88],
            [662.1, 120.0, 423.19, 95.15],
            [266.0, 125.49, 272.41, 120.67],
            [264.6, 114.26, 265.04, 112.92],
            [309.7, 91.29, 213.2, 88.7],
            [293.9, 143.01, 182.27, 53.09],
            [196.2, 100.82, 181.52, 76.97],
            [195.0, 97.35, 180.08, 77.62],
        ],
        "output_stem": "llama32_1b_decode_bars",
    },
}

colors = {
    "Native": "#7da9d8",
    "Vulkan": "#f2b37e",
    "WebGPU w/o Checks": "#8cc89a",
    "WebGPU w/ Checks": "#d8a6cf",
}

hatches = {
    "q4_k_m": "",
    "f16": "//",
}


def format_tokens_per_second(value, _):
    if value >= 1000:
        if value % 1000 == 0:
            return f"{int(value / 1000)}k"
        return f"{value / 1000:.1f}k"
    if value == int(value):
        return str(int(value))
    return f"{value:.0f}"


def build_metric_lookup(metric):
    return {
        series_key: np.array(values, dtype=float)[device_order]
        for series_key, values in zip(series, metric["values"])
    }


def print_percentage_summary():
    comparisons = [
        ("Vulkan", "Native"),
        ("Native", "WebGPU w/o Checks"),
        ("Vulkan", "WebGPU w/o Checks"),
        ("WebGPU w/o Checks", "WebGPU w/ Checks"),
    ]

    for metric_name, metric in metrics.items():
        lookup = build_metric_lookup(metric)
        print(f"\n{metric_name.upper()} PERCENTAGE SUMMARY")

        for quant in ("q4_k_m", "f16"):
            print(f"\n  {quant}")

            for lhs, rhs in comparisons:
                lhs_values = lookup[(quant, lhs)]
                rhs_values = lookup[(quant, rhs)]
                ratio = 100.0 * lhs_values / rhs_values
                aggregate_ratio = 100.0 * np.sum(lhs_values) / np.sum(rhs_values)
                mean_ratio = np.mean(ratio)

                print(
                    f"    {lhs} vs {rhs}: "
                    f"aggregate {aggregate_ratio:.1f}%, "
                    f"mean across devices {mean_ratio:.1f}%"
                )
                for device, device_ratio in zip(devices, ratio):
                    print(f"      {device}: {device_ratio:.1f}%")


def plot_metric(metric):
    x = np.arange(len(devices))
    bar_width = 0.095
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * bar_width
    has_legend = metric["show_legend"]

    fig_height = 6.2 if has_legend else 4.6
    fig, ax = plt.subplots(figsize=(10.5, fig_height))

    values_array = np.array(metric["values"], dtype=float)
    max_value = np.nanmax(values_array)
    min_positive_value = np.nanmin(values_array[values_array > 0])
    y_min = 10 ** np.floor(np.log10(min_positive_value))
    na_y = y_min * 1.15

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
        )

        for missing_x in x[~valid]:
            ax.text(
                missing_x + offsets[idx],
                na_y,
                "N/A",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=14,
                color="#666666",
            )

    ax.set_ylabel(metric["ylabel"], fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(devices, fontsize=20)
    ax.set_yscale("log")
    ax.set_ylim(y_min, max_value * 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(format_tokens_per_second))
    ax.tick_params(axis="y", labelsize=20)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    extra_artists = []

    if has_legend:
        backend_handles = [
            Patch(facecolor=colors[name], edgecolor="#2f2f2f", label=name)
            for name in colors
        ]
        quant_handles = [
            Patch(facecolor="white", edgecolor="#2f2f2f", hatch=hatches["q4_k_m"], label="q4_k_m"),
            Patch(facecolor="white", edgecolor="#2f2f2f", hatch=hatches["f16"], label="f16"),
        ]

        backend_legend = ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.42, 1.02),
            ncol=2,
            frameon=True,
            facecolor="white",
            edgecolor="#cfcfcf",
            framealpha=0.95,
            fontsize=18,
            title="Backend",
            title_fontsize=18,
            handles=backend_handles,
        )
        backend_legend.set_in_layout(False)
        ax.add_artist(backend_legend)
        extra_artists.append(backend_legend)

        quant_legend = ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.88, 1.02),
            ncol=1,
            frameon=True,
            facecolor="white",
            edgecolor="#cfcfcf",
            framealpha=0.95,
            fontsize=18,
            title="Weight Format",
            title_fontsize=18,
            handles=quant_handles,
        )
        quant_legend.set_in_layout(False)
        quant_legend._legend_box.align = "left"
        extra_artists.append(quant_legend)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if has_legend:
        fig.subplots_adjust(top=0.78)
    fig.savefig(
        f"{metric['output_stem']}.pdf",
        bbox_inches="tight",
        bbox_extra_artists=extra_artists,
    )


def main():
    print_percentage_summary()

    for metric in metrics.values():
        plot_metric(metric)

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
