"""
MAIN-SECTION figure for the quantization study (Sec 4.3).

Llama 3.2 1B Instruct across 4 weight formats (Q2_K, Q4_K_M, Q8_0, F16),
laid out as a 2x2 with quadrants:
  TL: prefill (pp512) @ depth 0      TR: prefill (pp512) @ depth 2048
  BL: decode  (tg128) @ depth 0      BR: decode  (tg128) @ depth 2048

Each panel: x = 4 quant variants (sorted by bit-width), grouped bars per
RAM bucket. Chrome where available; Safari only for iOS-only devices.

Output: quantization_main_2x2.pdf
"""

import os
from collections import defaultdict
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from portability_bench import (
    EDGE_COLOR, EDGE_WIDTH, GRID_COLOR,
    resolve_device, fmt_tps,
)


RUNS_DIR = "/tmp/webgpu-all/runs"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = "Llama-3.2-1B-Instruct"

# Variants sorted by bit-width (smallest first). Sizes are the Llama 3.2 1B
# Q*_K and F16 file sizes from the Hugging Face GGUFs.
VARIANT_ORDER = [
    ("Q2_K",   "Q2_K\n(0.58 GB)"),
    ("Q4_K_M", "Q4_K_M\n(0.81 GB)"),
    ("Q8_0",   "Q8_0\n(1.32 GB)"),
    ("F16",    "F16\n(2.48 GB)"),
]

# RAM buckets — same definition as portability_main_2x2.py
DEVICE_BUCKET = {
    ">=16 GB": [
        "RTX 5080", "RTX 5070", "RTX 40-series", "RX 7900 XT",
        "M4 (32 GB)", "M3 (16 GB)", "Iris Xe (iGPU)", "Snapdragon X Elite",
    ],
    "8-16 GB": [
        "Arc B580", "M2", "Galaxy S24", "Adreno 7xx",
    ],
    "<8 GB": [
        "iPhone 17 Pro Max", "iPhone 15", "PowerVR D-series",
    ],
}
LABEL_TO_BUCKET = {lbl: b for b, lbls in DEVICE_BUCKET.items() for lbl in lbls}

BUCKET_ORDER = [">=16 GB", "8-16 GB", "<8 GB"]
BUCKET_COLORS = {
    ">=16 GB": "#7da9d8",
    "8-16 GB": "#f2b37e",
    "<8 GB":   "#d8a6cf",
}

PANELS = [
    ("pp512_d0",    "Prefill — depth 0"),
    ("pp512_d2048", "Prefill — depth 2048"),
    ("tg128_d0",    "Decode — depth 0"),
    ("tg128_d2048", "Decode — depth 2048"),
]


# ----------------------------------------------------------------------

def chrome_unless_only_safari(recs):
    has_chrome = any(
        (r["browser"] or "").startswith(("chrom", "edge"))
        for r in recs
    )
    if has_chrome:
        return [r for r in recs
                if (r["browser"] or "").startswith(("chrom", "edge"))]
    return recs


def load_filtered(runs_dir):
    import glob, json
    raw = []
    for fp in sorted(glob.glob(os.path.join(runs_dir, "**/*.json"),
                               recursive=True)):
        with open(fp) as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for rec in data:
            if rec.get("status") != "done":
                continue
            if rec.get("model") != MODEL:
                continue
            resolved = resolve_device(rec)
            if resolved is None:
                continue
            _, label = resolved
            tests = (rec.get("metrics") or {}).get("tests") or []
            metric = {"pp512_d0": None, "tg128_d0": None,
                      "pp512_d2048": None, "tg128_d2048": None}
            for t in tests:
                nm, ts = t.get("name"), t.get("avg_ts")
                if nm == "pp512":            metric["pp512_d0"] = ts
                elif nm == "tg128":          metric["tg128_d0"] = ts
                elif nm == "pp512 @ d2048":  metric["pp512_d2048"] = ts
                elif nm == "tg128 @ d2048":  metric["tg128_d2048"] = ts
            raw.append({
                "label": label,
                "variant": rec.get("variant"),
                "browser": (rec.get("browser") or "").lower(),
                "metric": metric,
            })

    by_device = defaultdict(list)
    for r in raw:
        by_device[r["label"]].append(r)
    out = []
    for recs in by_device.values():
        out.extend(chrome_unless_only_safari(recs))
    return out


def aggregate_by_bucket(records):
    """{variant: {bucket: {metric_key: median_value}}}"""
    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        bucket = LABEL_TO_BUCKET.get(r["label"])
        if bucket is None:
            continue   # e.g. "M-series (Safari)" not in any bucket
        for k, v in r["metric"].items():
            if v is None:
                continue
            accum[r["variant"]][bucket][k].append(v)
    out = {}
    for variant, by_bucket in accum.items():
        out[variant] = {b: {k: median(vs) for k, vs in m.items()}
                        for b, m in by_bucket.items()}
    return out


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

def plot_2x2(agg, output_path):
    n_variants = len(VARIANT_ORDER)
    n_buckets = len(BUCKET_ORDER)
    x = np.arange(n_variants)
    bar_width = 0.27
    offsets = (np.arange(n_buckets) - (n_buckets - 1) / 2) * bar_width

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    axes = axes.flatten()

    for ax_idx, (ax, (metric_key, title)) in enumerate(zip(axes, PANELS)):
        all_vals = []
        for bucket in BUCKET_ORDER:
            for vid, _ in VARIANT_ORDER:
                v = (agg.get(vid, {}).get(bucket, {}) or {}).get(metric_key)
                if v is not None:
                    all_vals.append(v)
        panel_max = max(all_vals) if all_vals else 1.0

        for bi, bucket in enumerate(BUCKET_ORDER):
            values = []
            for vid, _ in VARIANT_ORDER:
                v = (agg.get(vid, {}).get(bucket, {}) or {}).get(metric_key)
                values.append(v if v is not None else np.nan)
            values = np.array(values, dtype=float)
            valid = ~np.isnan(values)

            ax.bar(
                x[valid] + offsets[bi],
                values[valid],
                width=bar_width,
                color=BUCKET_COLORS[bucket],
                edgecolor=EDGE_COLOR,
                linewidth=EDGE_WIDTH,
                label=bucket if ax_idx == 0 else None,
            )

        ax.set_title(title, fontsize=11, pad=10)
        ax.set_ylim(0, panel_max * 1.12)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_tps))
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([disp for _, disp in VARIANT_ORDER], fontsize=10)
        ax.set_xlabel("Weight format (sorted by bit-width)", fontsize=10)

    axes[0].set_ylabel("Tokens / second", fontsize=12)
    axes[2].set_ylabel("Tokens / second", fontsize=12)

    fig.suptitle("Cross-quantization — Llama 3.2 1B Instruct",
                 fontsize=13, y=1.005)
    fig.legend(
        loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncol=3, frameon=True, fontsize=10,
        facecolor="white", edgecolor="#cfcfcf", framealpha=0.95,
        title="Device RAM bucket",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def print_summary(agg):
    print(f"{'Variant':<10} {'Bucket':<10} {'pp_d0':>10} {'pp_d2048':>10} "
          f"{'tg_d0':>10} {'tg_d2048':>10}")
    for vid, _ in VARIANT_ORDER:
        for bucket in BUCKET_ORDER:
            vals = agg.get(vid, {}).get(bucket, {})
            def f(k):
                v = vals.get(k)
                return f"{v:>10.1f}" if v is not None else f"{'—':>10}"
            print(f"{vid:<10} {bucket:<10} {f('pp512_d0')} "
                  f"{f('pp512_d2048')} {f('tg128_d0')} {f('tg128_d2048')}")


def main():
    records = load_filtered(RUNS_DIR)
    print(f"Llama 3.2 1B records after Chrome/Safari filter: {len(records)}")
    agg = aggregate_by_bucket(records)
    print_summary(agg)
    out_path = os.path.join(OUT_DIR, "quantization_main_2x2.pdf")
    plot_2x2(agg, out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
