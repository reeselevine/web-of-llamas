"""
Dump portability-study benchmark data to a TSV used as the canonical
data source for the portability and quantization figure scripts.

For every device that ran any of the ten study models under Q4_K_M
(or Q1_0 for Bonsai-1.7B, which is 1-bit only), we emit one row with
all four phase/depth throughputs (pp512 d0/d2048, tg128 d0/d2048).
Chrome is preferred; Safari is used only when the device has no Chrome
record. The downstream plot scripts read this TSV directly instead of
walking the raw JSON archive.

Output: portability_data.tsv
"""

import os
from collections import defaultdict

from portability_bench import resolve_device
from portability_main_2x2 import (
    MODEL_ORDER, EXTRA_VARIANT_MODELS,
    chrome_unless_only_safari, _dedup_one_per_cell,
)


RUNS_DIR = "/tmp/webgpu-bench-hf/runs"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "portability_data.tsv")

DEFAULT_VARIANT = "Q4_K_M"

# Family ordering matches the quantization TSV's vendor grouping.
FAMILY_ORDER = [
    "NVIDIA", "AMD", "Intel", "Apple-Mac", "Apple-iOS",
    "Qualcomm", "Samsung", "ARM", "Img Tec",
]

METRIC_COLS = ["pp512_d0", "pp512_d2048", "tg128_d0", "tg128_d2048"]


def load_portability_raw(runs_dir):
    """Walk runs_dir and emit raw records for the ten study models."""
    import glob, json
    model_ids = {mid for mid, _ in MODEL_ORDER}
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
            if model_id not in model_ids:
                continue
            variant = rec.get("variant")
            allowed = (variant == DEFAULT_VARIANT
                       or EXTRA_VARIANT_MODELS.get(model_id) == variant)
            if not allowed:
                continue
            resolved = resolve_device(rec)
            if resolved is None:
                continue
            family, label = resolved
            tests = (rec.get("metrics") or {}).get("tests") or []
            metric = {k: None for k in METRIC_COLS}
            for t in tests:
                nm, ts = t.get("name"), t.get("avg_ts")
                if nm == "pp512":            metric["pp512_d0"] = ts
                elif nm == "tg128":          metric["tg128_d0"] = ts
                elif nm == "pp512 @ d2048":  metric["pp512_d2048"] = ts
                elif nm == "tg128 @ d2048":  metric["tg128_d2048"] = ts
            raw.append({
                "family": family,
                "label": label,
                "model": model_id,
                "variant": variant,
                "browser": (rec.get("browser") or "").lower(),
                "nReps": rec.get("nReps") or 0,
                "timestamp": rec.get("timestamp") or "",
                "metric": metric,
            })
    return raw


def chrome_filter(raw):
    """Apply chrome-where-available browser preference, then dedup to
    one record per (device, model)."""
    by_device = defaultdict(list)
    for r in raw:
        by_device[r["label"]].append(r)
    browser_filtered = []
    for recs in by_device.values():
        browser_filtered.extend(chrome_unless_only_safari(recs))
    return _dedup_one_per_cell(
        browser_filtered,
        cell_key=lambda r: (r["label"], r["family"], r["model"]),
    )


def device_sort_key(label, family):
    fam_idx = {f: i for i, f in enumerate(FAMILY_ORDER)}
    return (fam_idx.get(family, 99), label)


def model_sort_key(model_id):
    order = {mid: i for i, (mid, _) in enumerate(MODEL_ORDER)}
    return order.get(model_id, 99)


def fmt(value):
    return f"{value:.2f}" if value is not None else ""


def main():
    raw = load_portability_raw(RUNS_DIR)
    filtered = chrome_filter(raw)
    devices = {r["label"] for r in filtered}
    print(f"Loaded {len(raw)} portability records; "
          f"{len(filtered)} after chrome/safari preference; "
          f"{len(devices)} unique devices")

    families_by_device = {r["label"]: r["family"] for r in filtered}
    ordered_devices = sorted(
        devices,
        key=lambda lbl: device_sort_key(lbl, families_by_device[lbl]),
    )

    by_cell = {(r["label"], r["model"]): r for r in filtered}

    header = ["Device", "Family", "Model", "Variant"] + METRIC_COLS
    lines = ["\t".join(header)]
    for label in ordered_devices:
        family = families_by_device[label]
        models_for_device = sorted(
            (mid for (lbl, mid) in by_cell.keys() if lbl == label),
            key=model_sort_key,
        )
        for mid in models_for_device:
            rec = by_cell[(label, mid)]
            cells = [fmt(rec["metric"].get(k)) for k in METRIC_COLS]
            lines.append("\t".join(
                [label, family, mid, rec["variant"]] + cells
            ))

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
