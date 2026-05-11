#!/usr/bin/env bash
# Sample tab memory via `footprint` at ~4 Hz and write a CSV.
#
# Usage:
#   Safari:
#     ./get_tab_memory.sh --browser safari --pid <RENDERER_PID> --out <file.csv>
#     Columns: timestamp, total_mb, webkit_malloc_mb, graphics_mb
#
#   Chrome (renderer only):
#     ./get_tab_memory.sh --browser chrome --pid <RENDERER_PID> --out <file.csv>
#     Columns: timestamp, total_mb, tag16_mb, vm_allocate_mb
#
#   Chrome (renderer + GPU process, recommended for WebGPU workloads):
#     ./get_tab_memory.sh --browser chrome --pid <RENDERER_PID> --gpu-pid <GPU_PID> --out <file.csv>
#     Columns: timestamp, combined_mb, renderer_mb, gpu_mb, tag16_mb, graphics_mb
#
# Tag reference (Chrome):
#   tag16_mb    — app-specific tag 16 (V8 JS heap + WASM linear memory, renderer)
#   vm_allocate — untagged VM_ALLOCATE in renderer (anonymous mmap, additional WASM buffers)
#   graphics_mb — sum of "(graphics)" categories in the GPU process (WebGPU buffers/textures)

set -euo pipefail

BROWSER=""
PID=""
GPU_PID=""
OUT=""
DURATION_SECONDS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --browser) BROWSER="$2"; shift 2 ;;
    --pid)     PID="$2";     shift 2 ;;
    --gpu-pid) GPU_PID="$2"; shift 2 ;;
    --out)     OUT="$2";     shift 2 ;;
    --time)     DURATION_SECONDS="$2";     shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$PID" ]]  && { echo "Error: --pid required"  >&2; exit 1; }
[[ -z "$OUT" ]]  && { echo "Error: --out required"  >&2; exit 1; }

START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION_SECONDS))

# Auto-detect browser from process name if not specified
if [[ -z "$BROWSER" ]]; then
  PROC_NAME=$(ps -p "$PID" -o comm= 2>/dev/null || true)
  if echo "$PROC_NAME" | grep -qi "safari"; then
    BROWSER="safari"
  elif echo "$PROC_NAME" | grep -qi "chrome"; then
    BROWSER="chrome"
  else
    echo "Error: could not auto-detect browser; pass --browser safari|chrome" >&2
    exit 1
  fi
  echo "Auto-detected browser: $BROWSER"
fi

# Extract dirty MB for a grep pattern.
# footprint row format: "  371 MB   0 B   121 MB   12563   category name"
parse_mb() {
  local raw="$1" pattern="$2"
  echo "$raw" | awk -v pat="$pattern" '
    $0 ~ pat {
      val = $1; unit = $2
      if (unit == "MB") { printf "%d", int(val); exit }
      if (unit == "KB") { printf "%.1f", val/1024; exit }
      printf "0"; exit
    }
  '
}

# Sum all "(graphics)" category rows from a footprint output.
sum_graphics_mb() {
  local raw="$1"
  echo "$raw" | awk '
    /\(graphics\)/ {
      val = $1; unit = $2
      if (unit == "MB") sum += int(val)
      else if (unit == "KB") sum += val/1024
    }
    END { printf "%d", int(sum) }
  '
}

# Parse phys_footprint line: "    phys_footprint: 630 MB"
parse_footprint_total() {
  local raw="$1"
  echo "$raw" | grep "phys_footprint:" | grep -v peak | grep -oE '[0-9]+' | head -1
}

mkdir -p "$(dirname "$OUT")"

if [[ "$BROWSER" == "safari" ]]; then
  echo "timestamp,total_mb,webkit_malloc_mb,graphics_mb" > "$OUT"
  echo "Sampling Safari PID $PID for ${DURATION_SECONDS}s → $OUT"
  while (( $(date +%s) < END_TIME )); do
    TS=$(date +"%Y-%m-%d %H:%M:%S")
    RAW=$(footprint -p "$PID" 2>/dev/null)
    TOTAL=$(parse_footprint_total "$RAW")
    WEBKIT=$(parse_mb "$RAW" "WebKit malloc")
    GPU=$(sum_graphics_mb "$RAW")
    echo "$TS,$TOTAL,$WEBKIT,$GPU" >> "$OUT"
    printf "%s  Total: %sMB  WebKit: %sMB  Graphics: %sMB\n" "$TS" "$TOTAL" "$WEBKIT" "$GPU"
    sleep 0.25
  done

elif [[ "$BROWSER" == "chrome" ]]; then
  if [[ -n "$GPU_PID" ]]; then
    echo "timestamp,combined_mb,renderer_mb,gpu_mb,tag16_mb,graphics_mb" > "$OUT"
    echo "Sampling Chrome renderer PID $PID + GPU PID $GPU_PID for ${DURATION_SECONDS}s → $OUT"
    while (( $(date +%s) < END_TIME )); do
      TS=$(date +"%Y-%m-%d %H:%M:%S")
      RRAW=$(footprint -p "$PID"     2>/dev/null)
      GRAW=$(footprint -p "$GPU_PID" 2>/dev/null)
      RENDERER=$(parse_footprint_total "$RRAW")
      GPU_TOTAL=$(parse_footprint_total "$GRAW")
      COMBINED=$(( ${RENDERER:-0} + ${GPU_TOTAL:-0} ))
      TAG16=$(parse_mb "$RRAW" "app-specific tag 16")
      GRAPHICS=$(sum_graphics_mb "$GRAW")
      echo "$TS,$COMBINED,$RENDERER,$GPU_TOTAL,$TAG16,$GRAPHICS" >> "$OUT"
      printf "%s  Combined: %sMB  Renderer: %sMB  GPU proc: %sMB  Tag16(V8/WASM): %sMB  Graphics: %sMB\n" \
        "$TS" "$COMBINED" "$RENDERER" "$GPU_TOTAL" "$TAG16" "$GRAPHICS"
      sleep 0.25
    done
  else
    echo "timestamp,total_mb,tag16_mb,vm_allocate_mb" > "$OUT"
    echo "Sampling Chrome renderer PID $PID (no GPU pid) for ${DURATION_SECONDS}s → $OUT"
    echo "Tip: pass --gpu-pid <GPU_PID> to also capture WebGPU graphics memory"
    while (( $(date +%s) < END_TIME )); do
      TS=$(date +"%Y-%m-%d %H:%M:%S")
      RAW=$(footprint -p "$PID" 2>/dev/null)
      TOTAL=$(parse_footprint_total "$RAW")
      TAG16=$(parse_mb "$RAW" "app-specific tag 16")
      VMALLOC=$(parse_mb "$RAW" "untagged \(VM_ALLOCATE\)")
      echo "$TS,$TOTAL,$TAG16,$VMALLOC" >> "$OUT"
      printf "%s  Total: %sMB  Tag16(V8/WASM): %sMB  VM_Alloc: %sMB\n" "$TS" "$TOTAL" "$TAG16" "$VMALLOC"
      sleep 0.25
    done
  fi

else
  echo "Error: unknown --browser '$BROWSER'; expected safari or chrome" >&2
  exit 1
fi
