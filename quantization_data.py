"""
Dump Llama-3.2-1B-Instruct quantization benchmark data to a TSV that
slots cleanly into the quantization-study tracking sheet.

For every device that ran the model under any of Q2_K / Q4_K_M / Q8_0 /
F16, we emit four rows (prefill@d0, prefill@d2048, decode@d0,
decode@d2048) with one column per quantization variant. Chrome is
preferred; Safari is used only when the device has no Chrome record.

Output: quantization_data.tsv
"""

import os
from collections import defaultdict
from statistics import median

from portability_bench import resolve_device
from portability_main_2x2 import chrome_unless_only_safari


RUNS_DIR = "/tmp/webgpu-all/runs"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "quantization_data.tsv")

MODEL = "Llama-3.2-1B-Instruct"
VARIANTS = ["Q2_K", "Q4_K_M", "Q8_0", "F16"]

# Order the device rows by family then label, matching the sheet's
# vendor grouping.
FAMILY_ORDER = [
    "NVIDIA", "AMD", "Intel", "Apple-Mac", "Apple-iOS",
    "Qualcomm", "Samsung", "Img Tec",
]

PHASES = [
    ("Prefill", 0,    "pp512_d0"),
    ("Prefill", 2048, "pp512_d2048"),
    ("Decode",  0,    "tg128_d0"),
    ("Decode",  2048, "tg128_d2048"),
]


def load_llama_records(runs_dir):
    """Return list of raw per-run records for the llama model."""
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
            if rec.get("variant") not in VARIANTS:
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
                "browser": (rec.get("browser") or "").lower(),
                "metric": metric,
            })
    return raw


def chrome_filter(raw):
    """Same browser-preference rule as the portability scripts."""
    by_device = defaultdict(list)
    for r in raw:
        by_device[r["label"]].append(r)
    out = []
    for recs in by_device.values():
        out.extend(chrome_unless_only_safari(recs))
    return out


def aggregate(records):
    """{label: {variant: {metric_key: median_value, '_family': family}}}."""
    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    family_of = {}
    for r in records:
        family_of[r["label"]] = r["family"]
        for k, v in r["metric"].items():
            if v is None:
                continue
            accum[r["label"]][r["variant"]][k].append(v)
    out = {}
    for label, by_var in accum.items():
        out[label] = {"_family": family_of[label]}
        for var, metrics in by_var.items():
            out[label][var] = {k: median(vs) for k, vs in metrics.items()}
    return out


def device_sort_key(label, family):
    fam_idx = {f: i for i, f in enumerate(FAMILY_ORDER)}
    return (fam_idx.get(family, 99), label)


def fmt(value):
    return f"{value:.2f}" if value is not None else ""


def main():
    raw = load_llama_records(RUNS_DIR)
    filtered = chrome_filter(raw)
    agg = aggregate(filtered)
    print(f"Loaded {len(raw)} llama records; "
          f"{len(filtered)} after chrome/safari preference; "
          f"{len(agg)} unique devices")

    rows = sorted(
        agg.keys(),
        key=lambda lbl: device_sort_key(lbl, agg[lbl]["_family"]),
    )

    lines = ["\t".join(
        ["Device", "Family", "Phase", "KV depth"] + VARIANTS
    )]
    for label in rows:
        family = agg[label]["_family"]
        for phase_name, depth, metric_key in PHASES:
            cells = []
            for var in VARIANTS:
                v = agg[label].get(var, {}).get(metric_key)
                cells.append(fmt(v))
            lines.append("\t".join(
                [label, family, phase_name, str(depth)] + cells
            ))

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWrote {OUT_PATH}\n")
    # Echo to stdout so it can be eyeballed directly.
    print("\n".join(lines))


if __name__ == "__main__":
    main()
