"""
MAIN-SECTION figures for the quantization study (Sec 4.2.4).

Llama 3.2 1B Instruct across 4 weight formats (Q2_K, Q4_K_M, Q8_0, F16),
grouped by the same k-means device clusters used in the portability
study so the paper has one coherent device taxonomy.

Two separate single-panel PDFs:
  quantization_main_prefill.pdf - prefill (pp512)
  quantization_main_decode.pdf  - decode  (tg128)

Each panel: x = 4 quantization variants (sorted by bit-width). Per
variant, three cluster bars (A/B/C) with paired KV-depth 0/2048 bars
(solid + hatched).
"""

import os
from collections import defaultdict
from statistics import median

import matplotlib.pyplot as plt

from portability_bench import resolve_device
from portability_main_2x2 import (
    aggregate_by_bucket, assign_cluster_buckets,
    chrome_unless_only_safari, draw_panel,
    load_filtered as load_portability_records,
)


RUNS_DIR = "/tmp/webgpu-all/runs"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "quantization_study_figures")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL = "Llama-3.2-1B-Instruct"

# Variants sorted by bit-width (smallest first). Sizes are Llama 3.2 1B
# Q*_K and F16 file sizes from the Hugging Face GGUFs.
VARIANT_ORDER = [
    ("Q2_K",   "Q2_K\n(0.58 GB)"),
    ("Q4_K_M", "Q4_K_M\n(0.81 GB)"),
    ("Q8_0",   "Q8_0\n(1.32 GB)"),
    ("F16",    "F16\n(2.48 GB)"),
]
VARIANT_KEYS = [k for k, _ in VARIANT_ORDER]

PANELS = [
    ("prefill", "pp512_d0", "pp512_d2048"),
    ("decode",  "tg128_d0", "tg128_d2048"),
]


def load_quant_records(runs_dir):
    """Llama records across the four quantization variants, with the
    same Chrome-where-available browser preference used elsewhere."""
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
            if rec.get("variant") not in VARIANT_KEYS:
                continue
            resolved = resolve_device(rec)
            if resolved is None:
                continue
            family, label = resolved
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
                "family": family,
                "label": label,
                "variant": rec.get("variant"),
                # aggregate_by_bucket uses key_field; we want to group
                # by quantization variant, so expose it under "model"
                # as a convenience shim, plus an explicit "variant".
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


def print_summary(agg, bucket_order):
    print(f"\n=== quantization main figure ===")
    print(f"{'Variant':<10} {'Bucket':<14} {'pp_d0':>10} {'pp_d2048':>10} "
          f"{'tg_d0':>10} {'tg_d2048':>10}")
    for vid, _ in VARIANT_ORDER:
        for bucket in bucket_order:
            vals = agg.get(vid, {}).get(bucket, {})
            def f(k):
                v = vals.get(k)
                return f"{v:>10.1f}" if v is not None else f"{'—':>10}"
            print(f"{vid:<10} {bucket:<14} {f('pp512_d0')} "
                  f"{f('pp512_d2048')} {f('tg128_d0')} {f('tg128_d2048')}")


def plot_panel(agg, key_d0, key_d2k, output_path,
               bucket_order, bucket_colors, legend_title="Cluster"):
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    draw_panel(ax, agg, key_d0, key_d2k,
               bucket_order, bucket_colors, legend_title,
               x_order=VARIANT_ORDER)
    ax.set_ylabel("Tokens / second", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    # Cluster taxonomy is computed from the full portability data set
    # so the device groupings match the portability figure exactly.
    portability_records = load_portability_records(RUNS_DIR)
    cluster_bucket, cluster_order, cluster_colors = \
        assign_cluster_buckets(portability_records, k=3)

    quant_records = load_quant_records(RUNS_DIR)
    print(f"\nQuantization records (llama, Q2/Q4/Q8/F16): {len(quant_records)}")

    # aggregate_by_bucket groups by `key_field`; for the quantization
    # study we want one row per quantization variant.
    agg = aggregate_by_bucket(
        quant_records,
        device_bucket=cluster_bucket,
        key_field="variant",
    )

    # Drop any cluster that has no data in this study (low-end-mobile
    # devices typically didn't run the quantization sweep) so the
    # legend doesn't list empty entries.
    present_clusters = {b for v in agg.values() for b in v}
    cluster_order = [b for b in cluster_order if b in present_clusters]
    print(f"Clusters with data: {cluster_order}")

    print_summary(agg, cluster_order)

    for slug, key_d0, key_d2k in PANELS:
        out_path = os.path.join(OUT_DIR, f"quantization_main_{slug}.pdf")
        plot_panel(agg, key_d0, key_d2k, out_path,
                   bucket_order=cluster_order,
                   bucket_colors=cluster_colors)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
