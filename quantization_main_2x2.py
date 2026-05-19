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

from portability_main_2x2 import (
    aggregate_by_bucket, assign_cluster_buckets, draw_panel,
    load_portability_records,
)


DATA_TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "quantization_data.tsv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "quantization_study_figures")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL = "Llama-3.2-1B-Instruct"

# Variants sorted by bit-width (smallest first). File sizes for each
# Llama 3.2 1B variant live in Table 3, so they're not duplicated on
# the tick labels here.
VARIANT_ORDER = [
    ("Q2_K",   "q2_k"),
    ("Q4_K_M", "q4_k_m"),
    ("Q8_0",   "q8_0"),
    ("F16",    "f16"),
]
VARIANT_KEYS = [k for k, _ in VARIANT_ORDER]

PANELS = [
    ("prefill", "pp512_d0", "pp512_d2048"),
    ("decode",  "tg128_d0", "tg128_d2048"),
]

# quantization_data.tsv stores one row per (device, phase, depth), with
# a column per variant. The plot code wants the inverted shape — one
# record per (device, variant) carrying all four phase/depth metrics —
# so the loader pivots back into that form.
PHASE_DEPTH_TO_METRIC = {
    ("Prefill", "0"):    "pp512_d0",
    ("Prefill", "2048"): "pp512_d2048",
    ("Decode",  "0"):    "tg128_d0",
    ("Decode",  "2048"): "tg128_d2048",
}


def load_quant_records(path=DATA_TSV):
    """Read quantization_data.tsv and return one record per (device,
    variant) with the four phase/depth metric keys populated."""
    import csv
    accum = {}  # (label, variant) -> {family, metric}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            metric_key = PHASE_DEPTH_TO_METRIC.get(
                (row["Phase"], row["KV depth"])
            )
            if metric_key is None:
                continue
            label = row["Device"]
            family = row["Family"]
            for variant in VARIANT_KEYS:
                cell = row.get(variant, "")
                if not cell:
                    continue
                key = (label, variant)
                if key not in accum:
                    accum[key] = {
                        "family": family,
                        "label": label,
                        "variant": variant,
                        "metric": {k: None for k in PHASE_DEPTH_TO_METRIC.values()},
                    }
                accum[key]["metric"][metric_key] = float(cell)
    return list(accum.values())


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
               bucket_order, bucket_colors, legend_title="Cluster",
               with_legend=True):
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    draw_panel(ax, agg, key_d0, key_d2k,
               bucket_order, bucket_colors, legend_title,
               x_order=VARIANT_ORDER,
               with_legend=with_legend,
               rotate_xticks=False)
    ax.set_ylabel("Tokens / second", fontsize=22)
    fig.tight_layout()
    # bbox_extra_artists tells the tight-bbox calc to include the
    # legends that live above the axes, otherwise their titles get
    # visually clipped at the top edge of the saved area.
    from matplotlib.legend import Legend
    extras = [c for c in ax.get_children() if isinstance(c, Legend)]
    fig.savefig(output_path, bbox_inches="tight",
                bbox_extra_artists=extras)
    plt.close(fig)


def main():
    # Cluster taxonomy is computed from the full portability data set
    # so the device groupings match the portability figure exactly.
    portability_records = load_portability_records()
    cluster_bucket, cluster_order, cluster_colors = \
        assign_cluster_buckets(portability_records, k=3)

    quant_records = load_quant_records()
    print(f"\nQuantization records (llama, Q2/Q4/Q8/F16): {len(quant_records)}")

    # aggregate_by_bucket groups by `key_field`; for the quantization
    # study we want one row per quantization variant.
    agg = aggregate_by_bucket(
        quant_records,
        device_bucket=cluster_bucket,
        key_field="variant",
    )

    # Drop any cluster that has no data in this study. The low cluster
    # is always excluded — the paper prose in §7 frames the study as
    # high+mid only because the larger Llama variants exceed mobile
    # tab-memory budgets, and any stray low-cluster record (e.g. a
    # single Q8_0 submission) would otherwise contradict that framing.
    present_clusters = {b for v in agg.values() for b in v}
    cluster_order = [b for b in cluster_order
                     if b in present_clusters and b != "low"]
    print(f"Clusters with data: {cluster_order}")

    print_summary(agg, cluster_order)

    for slug, key_d0, key_d2k in PANELS:
        out_path = os.path.join(OUT_DIR, f"quantization_main_{slug}.pdf")
        # Decode sits directly under prefill in the paper; legend only
        # on prefill to avoid duplicating it.
        plot_panel(agg, key_d0, key_d2k, out_path,
                   bucket_order=cluster_order,
                   bucket_colors=cluster_colors,
                   with_legend=(slug == "prefill"))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
