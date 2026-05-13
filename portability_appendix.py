"""
Appendix figures for the portability study.

App-A: Coverage matrix (one figure per phase)
  Devices (rows) x models (cols). Per figure: two side-by-side
  heatmaps for depth 0 and depth 2048, sharing a log color scale and
  one colorbar. Cells with no run remain blank.
    portability_appendix_coverage_prefill.pdf
    portability_appendix_coverage_decode.pdf

App-B: Per-model device deep-dive (one PDF per (model, phase))
  Each panel: x = device (grouped by vendor family), paired bars for
  depth 0 (solid) vs depth 2048 (alpha-faded, same color). KV depth
  uses alpha rather than hatch in this figure only, because hatch is
  already in use to disambiguate families that share a color
  (Apple-Mac/iOS, Qualcomm/Samsung/Img Tec). Style otherwise matches
  CLAUDE.md.

Outputs:
  portability_appendix_coverage_prefill.pdf
  portability_appendix_coverage_decode.pdf
  portability_appendix_<model_slug>_prefill.pdf
  portability_appendix_<model_slug>_decode.pdf
"""

import os
from collections import defaultdict
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

from portability_bench import (
    EDGE_COLOR, EDGE_WIDTH, GRID_COLOR,
    FAMILY_COLORS, FAMILY_HATCH, FAMILY_ORDER,
    fmt_tps,
)
from portability_main_2x2 import MODEL_ORDER, PANELS, load_filtered


RUNS_DIR = "/tmp/webgpu-all/runs"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "portability_study_figures")
os.makedirs(OUT_DIR, exist_ok=True)


# Two-line display names so per-device x-tick labels fit horizontally
# (matching the main figure's model-label style). Keys are the device
# labels produced by resolve_device.
DEVICE_DISPLAY = {
    "RTX 5080":              "RTX 5080",
    "RTX 5070":              "RTX 5070",
    "RTX 40-series":         "RTX\n40-series",
    "RX 7900 XT":            "RX 7900\nXT",
    "Arc B580":              "Arc B580",
    "Iris Xe (iGPU)":        "Iris Xe\n(iGPU)",
    "M4 (32 GB)":            "M4\n(32 GB)",
    "M3 (16 GB)":            "M3\n(16 GB)",
    "M2":                    "M2",
    "M-series (Safari)":     "M-series\n(Safari)",
    "Snapdragon X Elite":    "Snapdragon\nX Elite",
    "Galaxy S24":            "Galaxy S24",
    "Adreno 7xx":            "Adreno\n7xx",
    "iPhone 17 Pro Max":     "iPhone 17\nPro Max",
    "iPhone 15":             "iPhone 15",
    "PowerVR D-series":      "PowerVR\nD-series",
}


def slugify(s):
    return (s.lower()
            .replace(".", "_").replace("-", "_")
            .replace(" ", "_"))


# ----------------------------------------------------------------------
# Aggregation: per (model, device_label) -> {metrics, family}
# ----------------------------------------------------------------------

def aggregate_per_device(records, key_field="model"):
    """{x_key: {device_label: {family, metrics}}}.

    key_field defaults to "model" for the portability study; quantization
    passes "variant" so the outer key is the quantization format.
    """
    accum = defaultdict(lambda: defaultdict(
        lambda: {"family": None, "metrics": defaultdict(list)}
    ))
    for r in records:
        xkey = r.get(key_field)
        if xkey is None:
            continue
        cell = accum[xkey][r["label"]]
        cell["family"] = r["family"]
        for k, v in r["metric"].items():
            if v is not None:
                cell["metrics"][k].append(v)

    out = {}
    for xkey, by_dev in accum.items():
        out[xkey] = {}
        for label, cell in by_dev.items():
            out[xkey][label] = {
                "family": cell["family"],
                "metrics": {k: median(vs) for k, vs in cell["metrics"].items()},
            }
    return out


def ordered_devices(per_device):
    """Stable global device order: family first, then alphabetical within."""
    seen = {}
    for by_dev in per_device.values():
        for label, info in by_dev.items():
            seen[label] = info["family"]
    family_idx = {f: i for i, f in enumerate(FAMILY_ORDER)}
    return sorted(seen.items(),
                  key=lambda lf: (family_idx.get(lf[1], 99), lf[0]))


# ----------------------------------------------------------------------
# App-A: coverage heatmap
# ----------------------------------------------------------------------

def plot_coverage(per_device, key_d0, key_d2k, phase_label, output_path,
                  x_order=None):
    """Coverage heatmap for one phase (prefill or decode).

    phase_label is e.g. "Prefill" / "Decode" and only appears on the
    colorbar legend; the caption explains the figure.
    x_order defaults to MODEL_ORDER (portability study). Quantization
    passes VARIANT_ORDER.
    """
    if x_order is None:
        x_order = MODEL_ORDER
    devices = ordered_devices(per_device)         # [(label, family), ...]
    models = x_order                              # [(key, display), ...]

    def grid_for(metric_key):
        g = np.full((len(devices), len(models)), np.nan)
        for di, (label, _) in enumerate(devices):
            for mi, (mid, _) in enumerate(models):
                v = (per_device.get(mid, {}).get(label, {}).get("metrics")
                     or {}).get(metric_key)
                if v is not None:
                    g[di, mi] = v
        return g

    grids = {key_d0: grid_for(key_d0), key_d2k: grid_for(key_d2k)}

    combined = np.concatenate([g.ravel() for g in grids.values()])
    vmin = float(np.nanmin(combined)) if np.isfinite(combined).any() else 1.0
    vmax = float(np.nanmax(combined)) if np.isfinite(combined).any() else 1.0
    norm = LogNorm(vmin=max(vmin, 1.0), vmax=vmax)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#f5f5f5")

    # Stack vertically so the model labels along the x-axis only render
    # once (under the bottom heatmap) and can stay horizontal.
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 10.5),
                             sharex=True,
                             gridspec_kw={"hspace": 0.12})

    panels = [
        (axes[0], key_d0,  "depth 0"),
        (axes[1], key_d2k, "depth 2048"),
    ]
    im = None
    log_floor = np.log10(max(vmin, 1.0))
    log_ceil  = np.log10(vmax)
    log_span  = max(log_ceil - log_floor, 1e-9)

    for ax, key, depth_label in panels:
        g = grids[key]
        masked = np.ma.masked_invalid(g)
        im = ax.imshow(masked, cmap=cmap, aspect="auto", norm=norm)

        for di in range(len(devices)):
            for mi in range(len(models)):
                v = g[di, mi]
                if np.isnan(v):
                    continue
                shade = (np.log10(v) - log_floor) / log_span
                color = "white" if shade > 0.55 else "#1f1f1f"
                ax.text(mi, di, fmt_tps(v, None),
                        ha="center", va="center", fontsize=10, color=color)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([d for _, d in models],
                           rotation=0, ha="center", fontsize=11)
        ax.set_yticks(range(len(devices)))
        ax.set_yticklabels([lbl for lbl, _ in devices], fontsize=11)
        ax.set_title(depth_label, fontsize=13, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        shrink=0.85, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(fmt_tps))
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(f"{phase_label} tokens / second (log scale)",
                   fontsize=12)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# App-B: per-model 2x2 device bars
# ----------------------------------------------------------------------

PANEL_KEYS = [
    ("prefill", "pp512_d0", "pp512_d2048"),
    ("decode",  "tg128_d0", "tg128_d2048"),
]


def plot_panel_per_device(per_device, model_id, key_d0, key_d2k, output_path):
    """Single-panel per-device figure for one (model, phase) pair.

    Style follows CLAUDE.md (log y, no title, in-axes top-right legends,
    larger fonts). KV depth is encoded with alpha rather than hatch
    because hatch is already used to disambiguate device families that
    share a color (Apple-iOS/Mac, Qualcomm/Samsung/Img Tec).
    """
    by_dev = per_device.get(model_id, {})
    if not by_dev:
        return False

    family_idx = {f: i for i, f in enumerate(FAMILY_ORDER)}
    items = sorted(by_dev.items(),
                   key=lambda lc: (family_idx.get(lc[1]["family"], 99), lc[0]))
    labels   = [lbl for lbl, _ in items]
    families = [info["family"] for _, info in items]
    metrics_per_dev = [info["metrics"] for _, info in items]

    fig, ax = plt.subplots(figsize=(15.5, 5.0))
    bar_width = 0.38
    x = np.arange(len(labels))

    d0 = np.array([m.get(key_d0, np.nan) if m else np.nan
                   for m in metrics_per_dev], dtype=float)
    d2k = np.array([m.get(key_d2k, np.nan) if m else np.nan
                    for m in metrics_per_dev], dtype=float)

    # Compact within each device slot: a missing d2048 bar shouldn't
    # leave a hole. Center whatever's present.
    for xi, val0, val2k, family in zip(x, d0, d2k, families):
        color = FAMILY_COLORS[family]
        hatch = FAMILY_HATCH[family]
        present = []
        if np.isfinite(val0):
            present.append((val0, 1.0))
        if np.isfinite(val2k):
            present.append((val2k, 0.55))
        if not present:
            continue
        n = len(present)
        local_offsets = (np.arange(n) - (n - 1) / 2) * bar_width
        for slot, (val, alpha) in enumerate(present):
            ax.bar(xi + local_offsets[slot], val, width=bar_width,
                   color=color, alpha=alpha, hatch=hatch,
                   edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)

    all_vals = np.concatenate([d0, d2k])
    panel_max = np.nanmax(all_vals) if np.isfinite(all_vals).any() else 1.0

    ax.set_yscale("log")
    ax.set_ylim(1.0, panel_max * 6.0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DEVICE_DISPLAY.get(lbl, lbl) for lbl in labels],
        rotation=0, ha="center", fontsize=10,
    )
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_tps))
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Tokens / second", fontsize=15)

    present = []
    for f in FAMILY_ORDER:
        if f in families and f not in present:
            present.append(f)
    family_handles = [plt.Rectangle((0, 0), 1, 1,
                                    facecolor=FAMILY_COLORS[f],
                                    hatch=FAMILY_HATCH[f],
                                    edgecolor=EDGE_COLOR,
                                    linewidth=EDGE_WIDTH)
                      for f in present]
    depth_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#7a7a7a",
                      edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH),
        plt.Rectangle((0, 0), 1, 1, facecolor="#7a7a7a", alpha=0.55,
                      edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH),
    ]

    leg_fam = ax.legend(
        family_handles, present,
        title="Family", loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=11, title_fontsize=11,
        frameon=True, facecolor="white",
        edgecolor="#cfcfcf", framealpha=0.95,
        handlelength=1.4, handletextpad=0.55, borderpad=0.45,
    )
    ax.add_artist(leg_fam)

    fig.canvas.draw()
    fam_bbox = leg_fam.get_window_extent().transformed(ax.transAxes.inverted())
    kv_right = fam_bbox.x0 - 0.015

    ax.legend(
        depth_handles, ["0", "2048"],
        title="KV depth", loc="upper right",
        bbox_to_anchor=(kv_right, 1.0),
        fontsize=11, title_fontsize=11,
        frameon=True, facecolor="white",
        edgecolor="#cfcfcf", framealpha=0.95,
        handlelength=1.4, handletextpad=0.55, borderpad=0.45,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


# ----------------------------------------------------------------------

def main():
    records = load_filtered(RUNS_DIR)
    print(f"Loaded {len(records)} Q4_K_M records (Chrome where available)")
    per_device = aggregate_per_device(records)

    # App-A (one coverage figure per phase)
    coverage_specs = [
        ("prefill", "Prefill", "pp512_d0", "pp512_d2048"),
        ("decode",  "Decode",  "tg128_d0", "tg128_d2048"),
    ]
    for slug, label, key_d0, key_d2k in coverage_specs:
        out = os.path.join(OUT_DIR,
                           f"portability_appendix_coverage_{slug}.pdf")
        plot_coverage(per_device, key_d0, key_d2k, label, out)
        print(f"Wrote {out}")

    # App-B (one PDF per (model, phase) for subfigure composition)
    for mid, _disp in MODEL_ORDER:
        for slug, key_d0, key_d2k in PANEL_KEYS:
            out = os.path.join(
                OUT_DIR,
                f"portability_appendix_{slugify(mid)}_{slug}.pdf",
            )
            ok = plot_panel_per_device(per_device, mid, key_d0, key_d2k, out)
            if ok:
                print(f"Wrote {out}")
            else:
                print(f"Skipped {mid} {slug} (no devices)")


if __name__ == "__main__":
    main()
