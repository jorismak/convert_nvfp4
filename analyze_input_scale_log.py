#!/usr/bin/env python3
"""
Parse NVFP4 input_scale logs and produce per-layer summary stats.

Input format (tab-separated):
  mode\tlayer\tscale\tamax\tshape
Example:
  dynamic\tblocks.0.self_attn.q\t0.008405\t22.593750\t(33440, 3072)

Outputs:
  - CSV summary (per layer)
  - JSON summary (per layer)

Usage:
  python analyze_input_scale_log.py \
      --input nvfp4_scales.txt \
      --csv nvfp4_scales_summary.csv \
      --json nvfp4_scales_summary.json \
      --percentiles 50,90,95,99 \
      --mode dynamic
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_percentiles(value: str) -> List[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    percentiles = []
    for p in parts:
        try:
            pct = int(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid percentile: {p}") from exc
        if pct < 0 or pct > 100:
            raise argparse.ArgumentTypeError(f"Percentile out of range: {pct}")
        percentiles.append(pct)
    if not percentiles:
        raise argparse.ArgumentTypeError("At least one percentile required")
    return sorted(set(percentiles))


def _percentile(sorted_vals: List[float], pct: int) -> float:
    if not sorted_vals:
        return float("nan")
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    lines = [",".join(fields)]
    for row in rows:
        values = []
        for f in fields:
            val = row.get(f, "")
            if isinstance(val, float):
                values.append(f"{val:.6f}")
            else:
                values.append(str(val))
        lines.append(",".join(values))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize NVFP4 input_scale logs")
    parser.add_argument("--input", required=True, help="Path to nvfp4_scales.txt")
    parser.add_argument("--csv", required=True, help="Output CSV path")
    parser.add_argument("--json", required=True, help="Output JSON path")
    parser.add_argument(
        "--percentiles",
        default="50,90,95,99",
        type=_parse_percentiles,
        help="Comma-separated percentiles (default: 50,90,95,99)",
    )
    parser.add_argument(
        "--mode",
        choices=["dynamic", "provided", "all"],
        default="dynamic",
        help="Filter by log mode (default: dynamic)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input log not found: {input_path}")

    scale_by_layer: Dict[str, List[float]] = defaultdict(list)
    amax_by_layer: Dict[str, List[float]] = defaultdict(list)

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            mode, layer, scale_str, amax_str, _shape = parts[:5]
            if args.mode != "all" and mode != args.mode:
                continue
            try:
                scale = float(scale_str)
                amax = float(amax_str)
            except ValueError:
                continue
            scale_by_layer[layer].append(scale)
            amax_by_layer[layer].append(amax)

    rows: List[Dict[str, object]] = []

    for layer in sorted(scale_by_layer.keys()):
        scales = scale_by_layer[layer]
        amaxes = amax_by_layer.get(layer, [])
        scales_sorted = sorted(scales)
        amax_sorted = sorted(amaxes)

        row: Dict[str, object] = {
            "layer": layer,
            "count": len(scales_sorted),
            "scale_min": scales_sorted[0],
            "scale_max": scales_sorted[-1],
            "scale_mean": sum(scales_sorted) / len(scales_sorted),
        }
        for pct in args.percentiles:
            row[f"scale_p{pct}"] = _percentile(scales_sorted, pct)

        if amax_sorted:
            row.update(
                {
                    "amax_min": amax_sorted[0],
                    "amax_max": amax_sorted[-1],
                    "amax_mean": sum(amax_sorted) / len(amax_sorted),
                }
            )
            for pct in args.percentiles:
                row[f"amax_p{pct}"] = _percentile(amax_sorted, pct)

        rows.append(row)

    # Write CSV
    fields = [
        "layer",
        "count",
        "scale_min",
        "scale_max",
        "scale_mean",
    ] + [f"scale_p{p}" for p in args.percentiles]

    if rows and "amax_min" in rows[0]:
        fields += [
            "amax_min",
            "amax_max",
            "amax_mean",
        ] + [f"amax_p{p}" for p in args.percentiles]

    _write_csv(Path(args.csv), rows, fields)

    # Write JSON
    json_out = {"mode": args.mode, "percentiles": args.percentiles, "layers": rows}
    Path(args.json).write_text(json.dumps(json_out, indent=2), encoding="utf-8")

    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote JSON: {args.json}")


if __name__ == "__main__":
    main()
