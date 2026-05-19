"""
Build cluster-level quantization summaries from quantization_data.tsv.

Inputs:
  quantization_data.tsv

Outputs:
  quantization_cluster_averages.tsv
  quantization_speedup_matrix.tsv

The cluster buckets are fixed to match the portability-study grouping
used by the quantization figures. This script intentionally works from
the checked-in TSV only; it does not read raw benchmark JSON.
"""

import csv
import os
from collections import defaultdict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "quantization_data.tsv")
OUT_AVG_PATH = os.path.join(BASE_DIR, "quantization_cluster_averages.tsv")
OUT_MATRIX_PATH = os.path.join(BASE_DIR, "quantization_speedup_matrix.tsv")

VARIANTS = ["Q2_K", "Q4_K_M", "Q8_0", "F16"]
PHASES = [
    ("Prefill", "0"),
    ("Prefill", "2048"),
    ("Decode", "0"),
    ("Decode", "2048"),
]

# Fixed cluster membership matching the portability-study tiers.
CLUSTER_BY_DEVICE = {
    "RTX 40-series": "Cluster A",
    "RTX 5070": "Cluster A",
    "RTX 5080": "Cluster A",
    "RX 7900 XT": "Cluster A",
    "Arc B580": "Cluster A",
    "M4 (32 GB)": "Cluster A",
    "M3 (16 GB)": "Cluster B",
    "M2": "Cluster B",
    "Iris Xe (iGPU)": "Cluster B",
    "Snapdragon X Elite": "Cluster C",
    "Galaxy S24": "Cluster C",
}
CLUSTER_ORDER = ["Cluster A", "Cluster B", "Cluster C"]
EXCLUDE_LABELS = {"M-series (Safari)"}


def fmt(value):
    return f"{value:.2f}" if value is not None else ""


def fmt_ratio(numer, denom):
    if numer is None or denom is None or denom == 0:
        return ""
    return f"{numer / denom:.2f}x"


def load_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def aggregate_cluster_means(rows):
    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        device = row["Device"]
        if device in EXCLUDE_LABELS:
            continue
        cluster = CLUSTER_BY_DEVICE.get(device)
        if cluster is None:
            continue
        phase = row["Phase"]
        depth = row["KV depth"]
        for variant in VARIANTS:
            raw_value = row.get(variant, "")
            if raw_value:
                accum[cluster][(phase, depth)][variant].append(float(raw_value))

    out = defaultdict(dict)
    for cluster, by_phase in accum.items():
        for phase_key, by_variant in by_phase.items():
            out[cluster][phase_key] = {
                variant: (sum(values) / len(values) if values else None)
                for variant, values in by_variant.items()
            }
    return out


def build_cluster_average_lines(agg):
    lines = ["\t".join(["Cluster", "Phase", "KV depth"] + VARIANTS)]
    for cluster in CLUSTER_ORDER:
        for phase, depth in PHASES:
            metrics = agg.get(cluster, {}).get((phase, depth), {})
            cells = [fmt(metrics.get(variant)) for variant in VARIANTS]
            lines.append("\t".join([cluster, phase, depth] + cells))
    return lines


def build_speedup_matrix_lines(agg):
    lines = []
    for cluster in CLUSTER_ORDER:
        if cluster not in agg:
            continue
        for phase, depth in PHASES:
            metrics = agg.get(cluster, {}).get((phase, depth), {})
            lines.append("\t".join([
                "Cluster", cluster,
                "Phase", phase,
                "KV depth", depth,
            ]))
            lines.append("\t".join(["Lower-bit vs higher-bit"] + VARIANTS))
            for row_idx, row_variant in enumerate(VARIANTS):
                cells = [row_variant]
                row_value = metrics.get(row_variant)
                for col_idx, col_variant in enumerate(VARIANTS):
                    if row_idx == col_idx:
                        cells.append("1.00x")
                    elif row_idx < col_idx:
                        cells.append(fmt_ratio(row_value, metrics.get(col_variant)))
                    else:
                        cells.append("")
                lines.append("\t".join(cells))
            lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def main():
    rows = load_rows(INPUT_PATH)
    agg = aggregate_cluster_means(rows)
    if not agg:
        raise SystemExit(f"No clusterable rows found in {INPUT_PATH}")

    avg_lines = build_cluster_average_lines(agg)
    with open(OUT_AVG_PATH, "w") as f:
        f.write("\n".join(avg_lines) + "\n")

    matrix_lines = build_speedup_matrix_lines(agg)
    with open(OUT_MATRIX_PATH, "w") as f:
        f.write("\n".join(matrix_lines) + "\n")

    print(f"Wrote {OUT_AVG_PATH}")
    print(f"Wrote {OUT_MATRIX_PATH}")


if __name__ == "__main__":
    main()
