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

RUNS_DIR = "/tmp/webgpu-all/runs"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "portability_study_figures")
os.makedirs(OUT_DIR, exist_ok=True)
VARIANT = "Q4_K_M"

# Q4_K_M models sorted by approximate file size (small -> large).
# Label format: "Name  (Q4_K_M size on disk)".
MODEL_ORDER = [
    ("gemma-3-270m-it",                "gemma3\n(0.25 GB)"),
    ("LFM2.5-350M",                    "lfm\n(0.23 GB)"),
    ("Qwen3-0.6B",                     "qwen3\n(0.40 GB)"),
    ("granite-4.0-h-1b",               "granite\n(0.90 GB)"),
    ("Llama-3.2-1B-Instruct",          "llama\n(0.81 GB)"),
    ("Bonsai-1.7B",                    "bonsai*\n(0.25 GB)"),
    ("Qwen3.5-2B",                     "qwen3.5\n(1.28 GB)"),
    ("gemma-4-E2B-it",                 "gemma4\n(3.11 GB)"),
    ("SmolLM3-3B",                     "smollm\n(1.92 GB)"),
    ("Ministral-3-3B-Instruct-2512",   "ministral\n(2.15 GB)"),
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
# Data loading: Chrome where possible, Safari only when no Chrome
# ----------------------------------------------------------------------

def chrome_unless_only_safari(records_for_device):
    has_chrome = any(
        (r["browser"] or "").startswith(("chrom", "edge"))
        for r in records_for_device
    )
    if has_chrome:
        return [r for r in records_for_device
                if (r["browser"] or "").startswith(("chrom", "edge"))]
    return records_for_device


def load_filtered(runs_dir):
    """Return list of records (one per benchmark, post-browser-preference)."""
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
            model_id = rec.get("model")
            variant = rec.get("variant")
            allowed = (variant == VARIANT
                       or EXTRA_VARIANT_MODELS.get(model_id) == variant)
            if not allowed:
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
                "model": rec.get("model"),
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


def aggregate_by_bucket(records, device_bucket=None, exclude_labels=()):
    """{model_id: {bucket: {metric_key: median_value}}}

    device_bucket: dict mapping device label -> bucket name (or None to
    exclude). Defaults to the module-level RAM-based DEVICE_BUCKET.
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
        for k, v in r["metric"].items():
            if v is None:
                continue
            accum[r["model"]][bucket][k].append(v)
    out = {}
    for model, by_bucket in accum.items():
        out[model] = {b: {k: median(vs) for k, vs in metrics.items()}
                      for b, metrics in by_bucket.items()}
    return out


# ----------------------------------------------------------------------
# Plot — visual style matched to benchmark_bars.py
# ----------------------------------------------------------------------

def draw_panel(ax, agg, key_d0, key_d2k,
               bucket_order, bucket_colors, legend_title,
               *, with_legend=True, with_xticklabels=True):
    """Draw the bar panel + legends onto a given Axes."""
    n_models = len(MODEL_ORDER)
    n_buckets = len(bucket_order)
    x = np.arange(n_models)

    # Bar width assumes the max possible bars per model so widths stay
    # constant across the figure. For each model we drop any
    # (cluster, depth) slot with no data and center the visible bars
    # within the group, so missing data compacts instead of leaving a
    # confusing hole — the eye can still pair the bars.
    n_bars_max = n_buckets * 2
    bar_width = 0.92 / n_bars_max

    all_vals = []
    for bucket in bucket_order:
        for mid, _ in MODEL_ORDER:
            cell = agg.get(mid, {}).get(bucket, {}) or {}
            for k in (key_d0, key_d2k):
                v = cell.get(k)
                if v is not None:
                    all_vals.append(v)
    panel_max = max(all_vals) if all_vals else 1.0

    for mi, (mid, _) in enumerate(MODEL_ORDER):
        entries = []
        for bucket in bucket_order:
            color = bucket_colors[bucket]
            cell = agg.get(mid, {}).get(bucket, {}) or {}
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
                mi + local_offsets[slot],
                val,
                width=bar_width,
                color=color,
                hatch=hatch,
                edgecolor=EDGE_COLOR,
                linewidth=EDGE_WIDTH,
            )

    ax.set_yscale("log")
    ax.set_ylim(1.0, panel_max * 6.0)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_tps))
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    if with_xticklabels:
        ax.set_xticklabels([disp for _, disp in MODEL_ORDER],
                           rotation=0, ha="center", fontsize=12)
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
    leg_ram = ax.legend(
        bucket_handles, list(bucket_order),
        title=legend_title, loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=11, title_fontsize=11,
        frameon=True, facecolor="white",
        edgecolor="#cfcfcf", framealpha=0.95,
        handlelength=1.4, handletextpad=0.55, borderpad=0.45,
    )
    ax.add_artist(leg_ram)

    ax.figure.canvas.draw()
    ram_bbox = leg_ram.get_window_extent().transformed(ax.transAxes.inverted())
    kv_right = ram_bbox.x0 - 0.015

    ax.legend(
        depth_handles, ["0", "2048"],
        title="KV depth", loc="upper right",
        bbox_to_anchor=(kv_right, 1.0),
        fontsize=11, title_fontsize=11,
        frameon=True, facecolor="white",
        edgecolor="#cfcfcf", framealpha=0.95,
        handlelength=1.4, handletextpad=0.55, borderpad=0.45,
    )


def plot_panel(agg, key_d0, key_d2k, output_path,
               bucket_order=None, bucket_colors=None,
               legend_title="RAM"):
    if bucket_order is None:
        bucket_order = BUCKET_ORDER
    if bucket_colors is None:
        bucket_colors = BUCKET_COLORS

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    draw_panel(ax, agg, key_d0, key_d2k,
               bucket_order, bucket_colors, legend_title)
    ax.set_ylabel("Tokens / second", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
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

    # Order clusters by mean log throughput so "Cluster A" is always the
    # fastest set of devices and colouring stays stable across reruns.
    mean_per_cluster = np.array([
        X[cluster_idx == c].sum(axis=1).mean()
        for c in range(k)
    ])
    order = np.argsort(-mean_per_cluster)
    remap = {old: new for new, old in enumerate(order)}
    cluster_idx = np.array([remap[c] for c in cluster_idx])

    names = [chr(ord("A") + i) for i in range(k)]
    device_bucket = {lbl: f"Cluster {names[c]}"
                     for lbl, c in zip(labels, cluster_idx)}

    bucket_order = [f"Cluster {n}" for n in names]
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
    records = load_filtered(RUNS_DIR)
    print(f"Records after Chrome/Safari filter: {len(records)}")

    cluster_bucket, cluster_order, cluster_colors = \
        assign_cluster_buckets(records, k=3)

    agg = aggregate_by_bucket(records, device_bucket=cluster_bucket)
    print_summary("cluster main figure", agg, cluster_order)

    for slug, key_d0, key_d2k in PANELS:
        out_path = os.path.join(OUT_DIR, f"portability_main_{slug}.pdf")
        plot_panel(agg, key_d0, key_d2k, out_path,
                   bucket_order=cluster_order,
                   bucket_colors=cluster_colors,
                   legend_title="Cluster")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
