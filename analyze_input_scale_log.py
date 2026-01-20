#!/usr/bin/env python3
"""
Parse input_scale logs and produce per-layer summary stats.

Input format (tab-separated):
    format\tmode\tlayer\tscale\tamax\tshape
Example:
    nvfp4\tdynamic\tblocks.0.self_attn.q\t0.008405\t22.593750\t(33440, 3072)

Legacy input format (tab-separated):
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
      --percentiles 50,90,95,99,99.9 \
      --mode dynamic
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_percentiles(value: str) -> List[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    percentiles: List[float] = []
    for p in parts:
        try:
            pct = float(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid percentile: {p}") from exc
        if pct < 0 or pct > 100:
            raise argparse.ArgumentTypeError(f"Percentile out of range: {pct}")
        percentiles.append(pct)
    if not percentiles:
        raise argparse.ArgumentTypeError("At least one percentile required")
    return sorted(set(percentiles))


def _percentile(sorted_vals: List[float], pct: float) -> float:
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


def _pct_key(pct: float) -> str:
    pct_str = str(pct).rstrip("0").rstrip(".")
    return pct_str.replace(".", "_")


def _step_percentile_stats(
    values: List[float], steps: int, cfg_passes: int, pct: float
) -> Tuple[float, float, float, int]:
    """
    Compute per-step percentile across all prompts/passes, then summarize.

    Returns:
        (mean, min, max, step_count)
    """
    if steps <= 0 or cfg_passes <= 0 or not values:
        return float("nan"), float("nan"), float("nan"), 0
    step_buckets: List[List[float]] = [[] for _ in range(steps)]
    for i, v in enumerate(values):
        step_idx = (i // cfg_passes) % steps
        step_buckets[step_idx].append(v)
    step_percentiles: List[float] = []
    for bucket in step_buckets:
        if not bucket:
            continue
        step_percentiles.append(_percentile(sorted(bucket), pct))
    if not step_percentiles:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        sum(step_percentiles) / len(step_percentiles),
        min(step_percentiles),
        max(step_percentiles),
        len(step_percentiles),
    )


def _stddev(values: List[float]) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _kurtosis(values: List[float]) -> float:
    """
    Compute excess kurtosis (Fisher) using population moments.

    Returns:
        Excess kurtosis (0 for normal), or NaN if undefined.
    """
    if not values:
        return float("nan")
    if len(values) < 2:
        return float("nan")
    mean = sum(values) / len(values)
    m2 = sum((v - mean) ** 2 for v in values) / len(values)
    if m2 == 0.0:
        return float("nan")
    m4 = sum((v - mean) ** 4 for v in values) / len(values)
    return (m4 / (m2 * m2)) - 3.0


def _outlier_frac(values: List[float], k: float = 3.0) -> float:
    """
    Fraction of values with |x - mean| > k * std.
    Returns NaN if std is 0 or values are empty.
    """
    if not values:
        return float("nan")
    if len(values) < 2:
        return float("nan")
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    if std == 0.0:
        return float("nan")
    threshold = k * std
    count = sum(1 for v in values if abs(v - mean) > threshold)
    return count / len(values)


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
    parser = argparse.ArgumentParser(description="Summarize input_scale logs")
    parser.add_argument("--input", required=True, help="Path to nvfp4_scales.txt")
    parser.add_argument("--csv", help="Output CSV path (optional)")
    parser.add_argument("--json", required=True, help="Output JSON path")
    parser.add_argument(
        "--format",
        default="nvfp4",
        help="Quant format to include (default: nvfp4). Use 'all' to skip filtering.",
    )
    parser.add_argument(
        "--percentiles",
        default="50,90,95,99,99.9",
        type=_parse_percentiles,
        help="Comma-separated percentiles (supports decimals, e.g. 99.9)",
    )
    parser.add_argument(
        "--mode",
        choices=["dynamic", "provided", "all"],
        default="dynamic",
        help="Filter by log mode (default: dynamic)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of diffusion steps (enables step-aware stats)",
    )
    parser.add_argument(
        "--cfg-passes",
        type=int,
        default=1,
        help="Number of passes per step (e.g. 2 for CFG: cond+uncond)",
    )
    parser.add_argument(
        "--step-percentile",
        type=float,
        default=99.9,
        help="Percentile to compute per step (default: 99.9)",
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

            # New format: format, mode, layer, scale, amax, shape
            if len(parts) >= 6:
                fmt, mode, layer, scale_str, amax_str, _shape = parts[:6]
                if args.format != "all" and fmt != args.format:
                    continue
            else:
                # Legacy format: mode, layer, scale, amax, shape
                fmt = "unknown"
                mode, layer, scale_str, amax_str, _shape = parts[:5]
                if args.format != "all" and args.format != "nvfp4":
                    continue
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

    step_pct_key = None
    if args.steps is not None:
        pct_str = str(args.step_percentile).rstrip("0").rstrip(".")
        pct_str = pct_str.replace(".", "_")
        step_pct_key = f"scale_step_p{pct_str}"

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
            "scale_std": _stddev(scales_sorted),
            "scale_kurtosis": _kurtosis(scales_sorted),
            "scale_outlier_frac_3std": _outlier_frac(scales_sorted, 3.0),
        }
        for pct in args.percentiles:
            row[f"scale_p{_pct_key(pct)}"] = _percentile(scales_sorted, pct)

        if args.steps is not None:
            mean_p, min_p, max_p, step_count = _step_percentile_stats(
                scales, args.steps, args.cfg_passes, args.step_percentile
            )
            row[f"{step_pct_key}_mean"] = mean_p
            row[f"{step_pct_key}_min"] = min_p
            row[f"{step_pct_key}_max"] = max_p
            row[f"{step_pct_key}_steps"] = step_count

        if amax_sorted:
            row.update(
                {
                    "amax_min": amax_sorted[0],
                    "amax_max": amax_sorted[-1],
                    "amax_mean": sum(amax_sorted) / len(amax_sorted),
                    "amax_std": _stddev(amax_sorted),
                    "amax_kurtosis": _kurtosis(amax_sorted),
                    "amax_outlier_frac_3std": _outlier_frac(amax_sorted, 3.0),
                }
            )
            for pct in args.percentiles:
                row[f"amax_p{_pct_key(pct)}"] = _percentile(amax_sorted, pct)

        rows.append(row)

    # Write CSV
    fields = [
        "layer",
        "count",
        "scale_min",
        "scale_max",
        "scale_mean",
        "scale_std",
        "scale_kurtosis",
        "scale_outlier_frac_3std",
    ] + [f"scale_p{_pct_key(p)}" for p in args.percentiles]

    if step_pct_key:
        fields += [
            f"{step_pct_key}_mean",
            f"{step_pct_key}_min",
            f"{step_pct_key}_max",
            f"{step_pct_key}_steps",
        ]

    if rows and "amax_min" in rows[0]:
        fields += [
            "amax_min",
            "amax_max",
            "amax_mean",
            "amax_std",
            "amax_kurtosis",
            "amax_outlier_frac_3std",
        ] + [f"amax_p{_pct_key(p)}" for p in args.percentiles]

    if args.csv:
        _write_csv(Path(args.csv), rows, fields)

    # Write JSON
    json_out = {
        "format": args.format,
        "mode": args.mode,
        "percentiles": args.percentiles,
        "layers": rows,
    }
    Path(args.json).write_text(json.dumps(json_out, indent=2), encoding="utf-8")

    if args.csv:
        print(f"Wrote CSV: {args.csv}")
    print(f"Wrote JSON: {args.json}")


if __name__ == "__main__":
    main()
