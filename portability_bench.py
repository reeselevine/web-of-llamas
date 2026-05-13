"""
Shared helpers for portability-study figures.

This module is imported by portability_main_2x2.py (and future
appendix scripts). It provides:
  - style constants (matched to benchmark_bars.py / breakdown.py)
  - resolve_device(record): map a raw HF benchmark record to a
    canonical (family, label) pair
  - load_runs(runs_dir, model=None, variant=None): generic loader
    that yields one entry per benchmark record after device resolution
  - fmt_tps(value, _): tick formatter for tokens/sec axes
"""

import glob
import json
import os

import matplotlib.pyplot as plt  # re-exported for downstream scripts
import numpy as np               # re-exported for downstream scripts
from matplotlib.ticker import FuncFormatter  # re-exported


# ----------------------------------------------------------------------
# Style — keep in sync with benchmark_bars.py / breakdown.py
# ----------------------------------------------------------------------

FAMILY_COLORS = {
    "NVIDIA":    "#7da9d8",  # blue
    "AMD":       "#f2b37e",  # orange
    "Intel":     "#8cc89a",  # green
    "Apple-Mac": "#d8a6cf",  # pink
    "Apple-iOS": "#d8a6cf",  # pink (hatched to distinguish)
    "Qualcomm":  "#b39ddb",  # purple
    "Samsung":   "#b39ddb",  # purple
    "Img Tec":   "#b39ddb",  # purple
}

FAMILY_HATCH = {
    "NVIDIA":    "",
    "AMD":       "",
    "Intel":     "",
    "Apple-Mac": "",
    "Apple-iOS": "//",   # iOS distinguished from Mac
    "Qualcomm":  "",
    "Samsung":   "xx",   # Android distinguished from ARM-Windows X Elite
    "Img Tec":   "..",
}

FAMILY_ORDER = [
    "NVIDIA", "AMD", "Intel", "Apple-Mac", "Apple-iOS",
    "Qualcomm", "Samsung", "Img Tec",
]

EDGE_COLOR = "#2f2f2f"
EDGE_WIDTH = 0.7
GRID_COLOR = "#d9d9d9"


# ----------------------------------------------------------------------
# Device resolution
# ----------------------------------------------------------------------

def resolve_device(record):
    """Map a single benchmark record to (family, label) or None to skip."""
    a = record.get("gpuAdapterInfo") or {}
    ur = record.get("userReported") or {}
    machine = record.get("machine") or {}
    mname = (ur.get("machineName") or "").strip().lower()
    ram = machine.get("totalMemoryGB", 0)
    vendor = (a.get("vendor") or "").lower()
    arch = (a.get("architecture") or "").lower()
    dev_id = (a.get("device") or "").lower()
    browser = (record.get("browser") or "").lower()

    # NVIDIA
    if vendor == "nvidia":
        if "blackwell" in arch and "0x2c02" in dev_id:
            return ("NVIDIA", "RTX 5080")
        if "blackwell" in arch:
            # anonymous Blackwell — user-reported as RTX 5070
            return ("NVIDIA", "RTX 5070")
        if "lovelace" in arch:
            return ("NVIDIA", "RTX 40-series")
        return None

    # AMD
    if vendor == "amd":
        if "7900" in dev_id or "7900" in mname:
            return ("AMD", "RX 7900 XT")
        return ("AMD", "AMD (other)")

    # Intel
    if vendor == "intel":
        if "xe-2hpg" in arch or "b580" in mname:
            return ("Intel", "Arc B580")
        if "gen-12lp" in arch:
            return ("Intel", "Iris Xe (iGPU)")
        return None

    # Apple — split Mac vs iOS using machineName / webkit signal
    if vendor == "apple":
        if browser.startswith("webkit"):
            if "iphone 17" in mname or "iphone17" in mname:
                return ("Apple-iOS", "iPhone 17 Pro Max")
            if "iphone 15" in mname or "iphone15" in mname:
                return ("Apple-iOS", "iPhone 15")
            if "macbook" in mname or " m4" in mname or " m3" in mname:
                return ("Apple-Mac", "M-series (Safari)")
            return None  # anonymous webkit run — skip
        # Chromium on macOS
        if "0x0000" in dev_id or " m2" in mname or "m2 " in mname:
            return ("Apple-Mac", "M2")
        if "m3" in mname:
            return ("Apple-Mac", "M3 (16 GB)")
        if "m4" in mname or ram >= 32:
            return ("Apple-Mac", "M4 (32 GB)")
        return None  # anonymous metal-3 — skip

    # Qualcomm
    if vendor == "qualcomm":
        if "x1-85" in mname or "0x36334330" in dev_id:
            return ("Qualcomm", "Snapdragon X Elite")
        if "adreno-7xx" in arch or "adreno-7" in mname:
            return ("Qualcomm", "Adreno 7xx")
        return None

    # Samsung (Xclipse on Galaxy)
    if vendor == "samsung":
        return ("Samsung", "Galaxy S24")

    # Imagination
    if "img" in vendor or "img-tec" in vendor:
        return ("Img Tec", "PowerVR D-series")

    return None


# ----------------------------------------------------------------------
# Generic data loader
# ----------------------------------------------------------------------

def load_runs(runs_dir, model=None, variant=None):
    """
    Walk runs_dir/**/*.json and return a list of dicts:
        {family, label, model, variant, browser, metric: {pp512_d0, ...}}
    Optionally filter to a specific model and/or weight variant.
    """
    out = []
    for fp in sorted(glob.glob(os.path.join(runs_dir, "**/*.json"),
                               recursive=True)):
        with open(fp) as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for rec in data:
            if rec.get("status") != "done":
                continue
            if model is not None and rec.get("model") != model:
                continue
            if variant is not None and rec.get("variant") != variant:
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
            out.append({
                "family": family,
                "label": label,
                "model": rec.get("model"),
                "variant": rec.get("variant"),
                "browser": (rec.get("browser") or "").lower(),
                "metric": metric,
            })
    return out


# ----------------------------------------------------------------------
# Axis formatter
# ----------------------------------------------------------------------

def fmt_tps(value, _):
    if value >= 1000:
        return f"{value/1000:.1f}k" if value % 1000 else f"{int(value/1000)}k"
    if value == int(value):
        return str(int(value))
    return f"{value:.0f}"
