"""
Appendix figures for the quantization study (Sec 4.2.4).

App-A: Coverage matrix (one figure per phase)
  Devices (rows) x quantization variants (cols). Per figure: two
  stacked heatmaps for depth 0 and depth 2048, sharing a log color
  scale and one colorbar. Cells with no run remain blank.
    quantization_appendix_coverage_prefill.pdf
    quantization_appendix_coverage_decode.pdf

App-B: Per-quantization device deep-dive (one PDF per (variant, phase))
  Each panel: x = device (grouped by vendor family), paired bars for
  depth 0 (solid) vs depth 2048 (alpha-faded, same color). Style
  follows CLAUDE.md; KV depth uses alpha rather than hatch because
  hatch is already in use to disambiguate families that share a color.

Outputs:
  quantization_appendix_coverage_prefill.pdf
  quantization_appendix_coverage_decode.pdf
  quantization_appendix_<variant_slug>_prefill.pdf
  quantization_appendix_<variant_slug>_decode.pdf
"""

import os

from portability_appendix import (
    aggregate_per_device, plot_coverage, plot_panel_per_device,
    slugify,
)
from quantization_main_2x2 import (
    MODEL, VARIANT_ORDER, PANELS,
    load_quant_records,
)


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "quantization_study_figures")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    records = load_quant_records()
    print(f"Loaded {len(records)} {MODEL} quantization records")

    per_device = aggregate_per_device(records, key_field="variant")

    # App-A
    coverage_specs = [
        ("prefill", "Prefill", "pp512_d0", "pp512_d2048"),
        ("decode",  "Decode",  "tg128_d0", "tg128_d2048"),
    ]
    for slug, label, key_d0, key_d2k in coverage_specs:
        out = os.path.join(
            OUT_DIR, f"quantization_appendix_coverage_{slug}.pdf"
        )
        plot_coverage(per_device, key_d0, key_d2k, label, out,
                      x_order=VARIANT_ORDER, rotate_xticks=False)
        print(f"Wrote {out}")

    # App-B (one PDF per (variant, phase)). All four variants live on
    # one figure-page in the paper; the legend would be identical
    # across them, so it's drawn on the first variant's prefill only,
    # using a master legend that covers every family that appears in
    # any variant. Every other panel gets no legend.
    first_variant = VARIANT_ORDER[0][0]
    master_families = set()
    for variant, _ in VARIANT_ORDER:
        for _label, info in per_device.get(variant, {}).items():
            if info.get("family"):
                master_families.add(info["family"])

    for variant, _disp in VARIANT_ORDER:
        for slug, key_d0, key_d2k in PANELS:
            out = os.path.join(
                OUT_DIR,
                f"quantization_appendix_{slugify(variant)}_{slug}.pdf",
            )
            show_legend = (slug == "prefill") and (variant == first_variant)
            master = master_families if show_legend else None
            ok = plot_panel_per_device(
                per_device, variant, key_d0, key_d2k, out,
                with_legend=show_legend, legend_families=master,
            )
            if ok:
                print(f"Wrote {out}")
            else:
                print(f"Skipped {variant} {slug} (no devices)")


if __name__ == "__main__":
    main()
