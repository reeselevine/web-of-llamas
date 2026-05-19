"""
MAIN-SECTION figures for the portability study (Sec 4.2.1).

Two separate single-panel PDFs (one per phase) so the paper can compose
them as subfigures with subcaptions:
  portability_main_prefill.pdf   - prefill (pp512)
  portability_main_decode.pdf    - decode  (tg128)

Bars are grouped by device cluster, where clusters are k-means assignments
on each device's log-throughput profile across the (model x phase x KV
depth) feature space. Within each cluster, a paired bar shows KV depth 0
(solid) and depth 2048 (hatched). Bars are the median across devices in
the cluster. Chrome where available; Safari only when the device has no
Chrome (iOS).

The per-device breakdown (every device on its own bar) lives in
portability_appendix.py.
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


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DATA_TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "portability_data.tsv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "portability_study_figures")
os.makedirs(OUT_DIR, exist_ok=True)
VARIANT = "Q4_K_M"

# Order matches Table 3 in the paper: Q4_K_M file size ascending, with
# Llama placed last because it's reused across the cross-quantization,
# cross-framework, and browser-vs.-native studies. File sizes and the
# Bonsai (Q1_0) caveat live in Table 3, so they're omitted from the
# tick labels here.
MODEL_ORDER = [
    ("LFM2.5-350M",                    "lfm"),
    ("Bonsai-1.7B",                    "bonsai"),
    ("gemma-3-270m-it",                "gemma3"),
    ("Qwen3-0.6B",                     "qwen3"),
    ("granite-4.0-h-1b",               "granite"),
    ("Qwen3.5-2B",                     "qwen3.5"),
    ("SmolLM3-3B",                     "smollm"),
    ("Ministral-3-3B-Instruct-2512",   "ministral"),
    ("gemma-4-E2B-it",                 "gemma4"),
    ("Llama-3.2-1B-Instruct",          "llama"),
]

# Models that are admitted under a non-Q4_K_M quantization. Bonsai is
# 1-bit only, so its q1_0 records are included as a deliberate wildcard.
# Caption should explain the asterisk on the model label.
EXTRA_VARIANT_MODELS = {
    "Bonsai-1.7B": "Q1_0",
}

# RAM bucket assignment by canonical device label (matches resolve_device).
# Reflects total memory addressable to a WebGPU process — not always equal
# to system RAM. iOS devices have a tight per-process budget (<1.5 GB
# typical) regardless of physical RAM.
DEVICE_BUCKET = {
    # >=16 GB
    "RTX 5080":              ">=16 GB",
    "RTX 5070":              ">=16 GB",
    "RTX 40-series":         ">=16 GB",
    "RX 7900 XT":            ">=16 GB",
    "M4 (32 GB)":            ">=16 GB",
    "M3 (16 GB)":            ">=16 GB",
    "Iris Xe (iGPU)":        ">=16 GB",   # 16 GB shared system memory
    "Snapdragon X Elite":    ">=16 GB",
    # 8-16 GB
    "Arc B580":              "8-16 GB",   # 8 GB dGPU VRAM
    "M2":                    "8-16 GB",   # 8 GB unified
    "Galaxy S24":            "8-16 GB",   # 8 GB total
    "Adreno 7xx":            "8-16 GB",
    # <8 GB
    "iPhone 17 Pro Max":     "<8 GB",
    "iPhone 15":             "<8 GB",
    "PowerVR D-series":      "<8 GB",
    "Mali (Valhall)":        "<8 GB",   # 8 GB Android device, low-power Mali
    "M-series (Safari)":     None,        # exclude Mac Safari from main fig
}

BUCKET_ORDER = [">=16 GB", "8-16 GB", "<8 GB"]
BUCKET_COLORS = {
    ">=16 GB": "#7da9d8",   # blue
    "8-16 GB": "#f2b37e",   # orange
    "<8 GB":   "#d8a6cf",   # pink
}

PANELS = [
    ("prefill", "pp512_d0", "pp512_d2048"),
    ("decode",  "tg128_d0", "tg128_d2048"),
]

D2K_HATCH = "///"  # hatch pattern overlaid on KV depth 2048 bars


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
# portability_data.tsv is the canonical input. portability_data.py
# dumps it from the raw JSON archive; the helpers below
# (chrome_unless_only_safari, _dedup_one_per_cell) are kept exported
# because the dumpers still rely on them.

def chrome_unless_only_safari(records_for_device):
    has_chrome = any(
        (r["browser"] or "").startswith(("chrom", "edge"))
        for r in records_for_device
    )
    if has_chrome:
        return [r for r in records_for_device
                if (r["browser"] or "").startswith(("chrom", "edge"))]
    return records_for_device


METRIC_COLS = ["pp512_d0", "pp512_d2048", "tg128_d0", "tg128_d2048"]


def load_portability_records(path=DATA_TSV):
    """Read portability_data.tsv and return one record per (device,
    model). The TSV is already deduped (browser preference, repetition
    selection, d0/d2048 merge) by portability_data.py, so this is a
    plain row-to-record mapping."""
    import csv
    records = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            metric = {}
            for k in METRIC_COLS:
                cell = row.get(k, "")
                metric[k] = float(cell) if cell else None
            records.append({
                "family": row["Family"],
                "label": row["Device"],
                "model": row["Model"],
                "variant": row.get("Variant") or VARIANT,
                "metric": metric,
            })
    return records


def _dedup_one_per_cell(records, cell_key=None):
    """Keep one record per cell + depth-class. Depth class is derived
    from which metric keys the record populates — d0 and d2048 results
    live in separate records in the JSON archive. After picking one
    record per class (highest nReps; tiebreak latest timestamp), merge
    the two into a single record with both depths' metrics populated.

    cell_key defaults to (label, family, model) for the portability
    study; pass a different callable to dedup by (device, variant) or
    any other grouping."""
    if cell_key is None:
        cell_key = lambda r: (r["label"], r["family"], r["model"])

    d0_keys = ("pp512_d0", "tg128_d0")
    d2k_keys = ("pp512_d2048", "tg128_d2048")

    def depth_class(rec):
        m = rec["metric"]
        has_d0 = any(m.get(k) is not None for k in d0_keys)
        has_d2k = any(m.get(k) is not None for k in d2k_keys)
        if has_d0 and not has_d2k:
            return "d0"
        if has_d2k and not has_d0:
            return "d2k"
        return None

    by_cell = defaultdict(lambda: defaultdict(list))
    for r in records:
        cls = depth_class(r)
        if cls is None:
            continue
        by_cell[cell_key(r)][cls].append(r)

    out = []
    for key, by_class in by_cell.items():
        merged_metric = {"pp512_d0": None, "tg128_d0": None,
                         "pp512_d2048": None, "tg128_d2048": None}
        template = None
        for cls, recs in by_class.items():
            recs.sort(key=lambda r: (r["nReps"], r["timestamp"]),
                      reverse=True)
            chosen = recs[0]
            keys = d0_keys if cls == "d0" else d2k_keys
            for k in keys:
                merged_metric[k] = chosen["metric"].get(k)
            template = template or chosen
        new_rec = {k: v for k, v in template.items()
                   if k not in ("metric", "nReps", "timestamp")}
        new_rec["metric"] = merged_metric
        out.append(new_rec)
    return out


def aggregate_by_bucket(records, device_bucket=None, exclude_labels=(),
                        key_field="model"):
    """{x_key: {bucket: {metric_key: median_value}}}

    key_field selects what becomes the outer dictionary key — "model"
    for the portability figure (one row per model) or "variant" for the
    quantization figure (one row per quantization).
    """
    if device_bucket is None:
        device_bucket = DEVICE_BUCKET
    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        if r["label"] in exclude_labels:
            continue
        bucket = device_bucket.get(r["label"])
        if bucket is None:
            continue
        xkey = r.get(key_field)
        if xkey is None:
            continue
        for k, v in r["metric"].items():
            if v is None:
                continue
            accum[xkey][bucket][k].append(v)
    out = {}
    for xkey, by_bucket in accum.items():
        out[xkey] = {b: {k: median(vs) for k, vs in metrics.items()}
                     for b, metrics in by_bucket.items()}
    return out


# ----------------------------------------------------------------------
# Plot — visual style matched to benchmark_bars.py
# ----------------------------------------------------------------------

def _center_legend_pair(ax, leg_left, leg_right, center=0.5, gap=0.01):
    """Reposition a pair of legends so they sit as a centered group.

    The two legends are initially placed with bbox_to_anchor=(0.5, ...)
    using loc="lower right" (left legend) and loc="lower left" (right
    legend). After the first draw we can read each legend's rendered
    width in axes coordinates and shift both anchors so the pair is
    centered around `center` with `gap` between them. Naive symmetric
    offsets bias the pair toward whichever legend is narrower.
    """
    ax.figure.canvas.draw()
    inv = ax.transAxes.inverted()
    lb = leg_left.get_window_extent().transformed(inv)
    rb = leg_right.get_window_extent().transformed(inv)
    w_left = lb.x1 - lb.x0
    w_right = rb.x1 - rb.x0
    left_anchor_x = center + (w_left - gap - w_right) / 2.0
    right_anchor_x = left_anchor_x + gap
    leg_left.set_bbox_to_anchor((left_anchor_x, 1.02))
    leg_right.set_bbox_to_anchor((right_anchor_x, 1.02))


def draw_panel(ax, agg, key_d0, key_d2k,
               bucket_order, bucket_colors, legend_title,
               *, x_order=None, with_legend=True, with_xticklabels=True,
               rotate_xticks=True):
    """Draw the bar panel + legends onto a given Axes.

    x_order is a list of (key, display_label) tuples — defaults to
    MODEL_ORDER for the portability figures; quantization passes
    VARIANT_ORDER.

    rotate_xticks controls whether x-tick labels are angled at 30°
    (default True for the 10-model portability figure) or rendered
    horizontally (used by the quantization figure with 4 short labels
    that fit without rotation).
    """
    if x_order is None:
        x_order = MODEL_ORDER
    n_x = len(x_order)
    n_buckets = len(bucket_order)
    x = np.arange(n_x)

    # Bar width assumes the max possible bars per x-slot so widths stay
    # constant across the figure. For each slot we drop any (bucket,
    # depth) entry with no data and center the visible bars within the
    # group, so missing data compacts instead of leaving a confusing
    # hole — the eye can still pair the bars.
    n_bars_max = n_buckets * 2
    bar_width = 0.92 / n_bars_max

    all_vals = []
    for bucket in bucket_order:
        for xk, _ in x_order:
            cell = agg.get(xk, {}).get(bucket, {}) or {}
            for k in (key_d0, key_d2k):
                v = cell.get(k)
                if v is not None:
                    all_vals.append(v)
    panel_max = max(all_vals) if all_vals else 1.0

    for xi, (xk, _) in enumerate(x_order):
        entries = []
        for bucket in bucket_order:
            color = bucket_colors[bucket]
            cell = agg.get(xk, {}).get(bucket, {}) or {}
            for di, key in enumerate((key_d0, key_d2k)):
                v = cell.get(key)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                entries.append((v, color, D2K_HATCH if di == 1 else None))
        if not entries:
            continue
        n = len(entries)
        local_offsets = (np.arange(n) - (n - 1) / 2) * bar_width
        for slot, (val, color, hatch) in enumerate(entries):
            ax.bar(
                xi + local_offsets[slot],
                val,
                width=bar_width,
                color=color,
                hatch=hatch,
                edgecolor=EDGE_COLOR,
                linewidth=EDGE_WIDTH,
            )

    ax.set_yscale("log")
    # Legends now live above the axes, so the in-axes headroom can shrink.
    ax.set_ylim(1.0, panel_max * 2.0)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_tps))
    ax.tick_params(axis="y", labelsize=20)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    if with_xticklabels:
        # Portability uses rotation=30 because 10 longer model labels
        # (e.g., "ministral", "qwen3.5") would otherwise collide at
        # 22pt. Quantization passes rotate_xticks=False because its 4
        # short variant labels (q2_k, q4_k_m, q8_0, f16) fit cleanly
        # without rotation.
        rot = 30 if rotate_xticks else 0
        ha = "right" if rotate_xticks else "center"
        ax.set_xticklabels([disp for _, disp in x_order],
                           rotation=rot, ha=ha, fontsize=22)
    else:
        ax.set_xticklabels([])

    if not with_legend:
        return

    bucket_handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=bucket_colors[b],
                      edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
        for b in bucket_order
    ]
    depth_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white",
                      edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", hatch=D2K_HATCH,
                      edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH),
    ]
    # Single horizontal legend above the axes, with the section labels
    # ("Cluster:", "KV depth:") inlined as text-only entries before
    # each group of swatches. Keeps both groups visible without the
    # title-on-top-of-entries vertical stacking matplotlib uses by
    # default for the legend title.
    import matplotlib.patches as mpatches
    spacer = mpatches.Patch(visible=False)
    combined_handles = (
        [spacer] + bucket_handles + [spacer] + depth_handles
    )
    combined_labels = (
        [f"{legend_title}:"] + list(bucket_order)
        + ["KV depth:", "0", "2048"]
    )
    ax.legend(
        combined_handles, combined_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(combined_handles),
        fontsize=22,
        frameon=True, facecolor="white",
        edgecolor="#cfcfcf", framealpha=0.95,
        handlelength=1.4, handletextpad=0.55, borderpad=0.45,
        columnspacing=1.0,
    )


def plot_panel(agg, key_d0, key_d2k, output_path,
               bucket_order=None, bucket_colors=None,
               legend_title="RAM", with_legend=True):
    if bucket_order is None:
        bucket_order = BUCKET_ORDER
    if bucket_colors is None:
        bucket_colors = BUCKET_COLORS

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    draw_panel(ax, agg, key_d0, key_d2k,
               bucket_order, bucket_colors, legend_title,
               with_legend=with_legend)
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


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Device clustering (k-means on log-throughput feature vectors)
# ----------------------------------------------------------------------
# Inspired by GPU Harbor Sec 5.2 (per-device feature vector + k-means).
# We diverge in one place: GPU Harbor L2-normalises features so distance
# is cosine similarity (pattern-only). For a portability story absolute
# throughput matters, so we cluster on raw log-throughput features.
# Missing (model, phase, depth) cells are imputed with the per-column
# median across devices that did run.

PALETTE = {
    "blue":   "#7da9d8",
    "orange": "#f2b37e",
    "pink":   "#d8a6cf",
    "green":  "#8cc89a",
    "purple": "#b39ddb",
}

METRIC_KEYS = ["pp512_d0", "pp512_d2048", "tg128_d0", "tg128_d2048"]

# Mac Safari records duplicate Mac Chrome runs on the same physical
# machine, so they're kept out of the clustering input.
CLUSTER_EXCLUDE = {"M-series (Safari)"}


def build_per_device_matrix(records, normalize=False):
    """Return (device_labels, feature_matrix) for k-means."""
    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        if r["label"] in CLUSTER_EXCLUDE:
            continue
        for k, v in r["metric"].items():
            if v is not None:
                accum[r["label"]][r["model"]][k].append(v)

    device_labels = sorted(accum.keys())
    model_ids = [mid for mid, _ in MODEL_ORDER]
    n_dev = len(device_labels)
    n_cols = len(model_ids) * len(METRIC_KEYS)

    raw = np.full((n_dev, n_cols), np.nan)
    for di, label in enumerate(device_labels):
        for mi, mid in enumerate(model_ids):
            cell = accum[label].get(mid, {})
            for ki, mk in enumerate(METRIC_KEYS):
                vs = cell.get(mk)
                if vs:
                    raw[di, mi * len(METRIC_KEYS) + ki] = median(vs)

    feats = np.log1p(raw)
    col_medians = np.nanmedian(feats, axis=0)
    inds = np.where(np.isnan(feats))
    feats[inds] = np.take(col_medians, inds[1])
    keep = ~np.isnan(feats).any(axis=0)
    feats = feats[:, keep]

    if normalize:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feats / norms
    return device_labels, feats


def kmeans(X, k, *, n_init=25, max_iter=200, seed=0):
    """Tiny numpy k-means, returns (labels, inertia, centroids)."""
    rng = np.random.default_rng(seed)
    n, _ = X.shape
    best = None
    for _ in range(n_init):
        idx = rng.choice(n, size=k, replace=False)
        centroids = X[idx].copy()
        for _it in range(max_iter):
            dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
            labels = dists.argmin(axis=1)
            new_centroids = np.stack([
                X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                for j in range(k)
            ])
            if np.allclose(new_centroids, centroids):
                centroids = new_centroids
                break
            centroids = new_centroids
        inertia = ((X - centroids[labels]) ** 2).sum()
        if best is None or inertia < best[1]:
            best = (labels.copy(), inertia, centroids.copy())
    return best


def assign_cluster_buckets(records, k=3):
    """Return (device_bucket, bucket_order, bucket_colors)."""
    labels, X = build_per_device_matrix(records, normalize=False)
    print(f"k-means on {len(labels)} devices, {X.shape[1]} features")
    print("Inertia by k:")
    for kk in (2, 3, 4, 5):
        _, inertia, _ = kmeans(X, kk)
        print(f"  k={kk:2d}  inertia={inertia:.4f}")

    cluster_idx, inertia, _ = kmeans(X, k)
    print(f"\nChose k={k}; inertia={inertia:.4f}")

    # Order clusters by *measured* throughput (ignoring imputed cells)
    # so the "high" cluster is the fastest devices, "mid" the mid-tier,
    # and "low" the slowest. Ranking off the post-imputation feature
    # matrix inflates devices with lots of missing cells (iPhones get
    # the dataset median per column), which previously flipped mid/low.
    device_mean_log = {}
    for r in records:
        if r["label"] in CLUSTER_EXCLUDE:
            continue
        for v in r["metric"].values():
            if v is None:
                continue
            device_mean_log.setdefault(r["label"], []).append(np.log1p(v))
    device_mean_log = {lbl: float(np.mean(vs))
                       for lbl, vs in device_mean_log.items()}
    label_arr = np.array(labels)
    mean_per_cluster = np.array([
        float(np.mean([device_mean_log[lbl]
                       for lbl in label_arr[cluster_idx == c]]))
        for c in range(k)
    ])
    order = np.argsort(-mean_per_cluster)
    remap = {old: new for new, old in enumerate(order)}
    cluster_idx = np.array([remap[c] for c in cluster_idx])

    # Cluster names match the prose in the paper.
    CLUSTER_NAMES = ["high", "mid", "low", "tier4", "tier5"]
    names = CLUSTER_NAMES[:k]
    device_bucket = {lbl: names[c]
                     for lbl, c in zip(labels, cluster_idx)}

    bucket_order = list(names)
    palette_seq = [PALETTE["blue"], PALETTE["orange"],
                   PALETTE["pink"], PALETTE["green"], PALETTE["purple"]]
    bucket_colors = {b: palette_seq[i] for i, b in enumerate(bucket_order)}

    print("\nCluster membership:")
    for b in bucket_order:
        members = [lbl for lbl, bb in device_bucket.items() if bb == b]
        print(f"  {b}: {', '.join(members)}")

    return device_bucket, bucket_order, bucket_colors


def print_summary(label, agg, bucket_order):
    print(f"\n=== {label} ===")
    print(f"{'Model':<22} {'Bucket':<14} {'pp_d0':>10} {'pp_d2048':>10} "
          f"{'tg_d0':>10} {'tg_d2048':>10}")
    for mid, disp in MODEL_ORDER:
        for bucket in bucket_order:
            vals = agg.get(mid, {}).get(bucket, {})
            def f(k):
                v = vals.get(k)
                return f"{v:>10.1f}" if v is not None else f"{'—':>10}"
            disp_one_line = disp.replace("\n", " ")
            print(f"{disp_one_line:<22} {bucket:<14} {f('pp512_d0')} "
                  f"{f('pp512_d2048')} {f('tg128_d0')} {f('tg128_d2048')}")


def main():
    records = load_portability_records()
    print(f"Loaded {len(records)} records from {DATA_TSV}")

    cluster_bucket, cluster_order, cluster_colors = \
        assign_cluster_buckets(records, k=3)

    agg = aggregate_by_bucket(records, device_bucket=cluster_bucket)

    # Qwen3.5 ran on too few low-cluster devices for a single-bar
    # cluster median to be meaningful — drop it from the main figure.
    # Per-device throughput for these devices still appears in the
    # appendix figure for Qwen3.5.
    if "Qwen3.5-2B" in agg:
        agg["Qwen3.5-2B"].pop("low", None)

    print_summary("cluster main figure", agg, cluster_order)

    for slug, key_d0, key_d2k in PANELS:
        out_path = os.path.join(OUT_DIR, f"portability_main_{slug}.pdf")
        # Decode sits directly under prefill in the paper; legend only
        # on prefill to avoid duplicating it.
        plot_panel(agg, key_d0, key_d2k, out_path,
                   bucket_order=cluster_order,
                   bucket_colors=cluster_colors,
                   legend_title="Cluster",
                   with_legend=(slug == "prefill"))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
