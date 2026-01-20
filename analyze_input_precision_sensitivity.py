#!/usr/bin/env python3
"""
Analyze input precision sensitivity for NVFP4 layers.

Runs a FP16/FP32 model forward pass and estimates how much
input quantization (FP4 with NVFP4-style scaling) would distort
the activations per Linear layer. Produces a ranked list of layers
most sensitive to input precision.

Usage:
  python analyze_input_precision_sensitivity.py \
    --model-type wan22_5b \
      --fp16-model D:\\ComfyUI\\ComfyUI\\models\\diffusion_models\\wan2.2_ti2v_5B_fp16.safetensors \
      --samples 6 \
      --input-scale-summary-json nvfp4_scales_summary.json \
      --input-scale-summary-percentile 99 \
      --output-json sensitivity.json \
      --output-layers full_precision_layers.txt \
      --top-k 30
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

import torch

from nvfp4_calibration import (
    MODEL_TYPES,
    QWEN_MODEL_TYPES,
    WAN_MODEL_TYPES,
    build_qwen_model_for_calibration,
    build_wan_model_for_calibration,
    run_qwen_calibration_passes,
    run_wan_calibration_passes,
)

REPO_ROOT = Path(__file__).resolve().parent
CK_PATH = REPO_ROOT / "comfy-kitchen"
if CK_PATH.exists():
    sys.path.insert(0, str(CK_PATH))

try:
    from comfy_kitchen.float_utils import (
        F4_E2M1_MAX,
        F8_E4M3_MAX,
        _f32_to_floatx_unpacked,
        _floatx_unpacked_to_f32,
    )
except Exception as exc:
    raise ImportError(
        "Failed to import comfy_kitchen.float_utils. Ensure comfy-kitchen is present."
    ) from exc

NVFP4_BLOCK_SIZE = 16


def _resolve_comfyui_root(user_root: Optional[str]) -> Path:
    if user_root:
        return Path(user_root)
    return REPO_ROOT.parents[0]


def _load_state_dict_for_calibration(model_path: str) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open

    path = Path(model_path)
    if path.suffix == ".safetensors":
        state_dict: Dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict

    return torch.load(path, map_location="cpu")


def _build_model_for_calibration(
    model_path: str,
    model_type: str,
    device: str,
    comfyui_root: Optional[str],
):
    state_dict = _load_state_dict_for_calibration(model_path)
    if model_type in WAN_MODEL_TYPES:
        model = build_wan_model_for_calibration(
            state_dict, device, comfyui_root, model_type
        )
    else:
        model = build_qwen_model_for_calibration(
            state_dict, device, comfyui_root, model_type
        )

    return model


def _to_float8_e4m3(t: torch.Tensor) -> torch.Tensor:
    try:
        return t.to(torch.float8_e4m3fn).to(torch.float32)
    except Exception:
        return t.to(torch.float32)


def _quantize_input_fp4(
    x: torch.Tensor, input_scale: float
) -> tuple[torch.Tensor, torch.Tensor, float]:
    x2d = x.reshape(-1, x.shape[-1]).float()
    orig_cols = x2d.shape[1]

    pad = 0
    if orig_cols % NVFP4_BLOCK_SIZE != 0:
        pad = NVFP4_BLOCK_SIZE - (orig_cols % NVFP4_BLOCK_SIZE)
        x2d = torch.nn.functional.pad(x2d, (0, pad))

    x_blocks = x2d.reshape(x2d.shape[0], -1, NVFP4_BLOCK_SIZE)

    per_tensor_scale = max(float(input_scale), 1e-12)
    block_amax = x_blocks.abs().amax(dim=-1)
    block_scale = (block_amax / F4_E2M1_MAX) / per_tensor_scale
    block_scale = torch.clamp(block_scale, max=F8_E4M3_MAX)
    block_scale = _to_float8_e4m3(block_scale)

    scale = (per_tensor_scale * block_scale).unsqueeze(-1)
    x_scaled = x_blocks / scale
    clip_rate = (x_scaled.abs() > F4_E2M1_MAX).float().mean().item()

    x_fp4 = _f32_to_floatx_unpacked(x_scaled, 2, 1)
    x_deq = _floatx_unpacked_to_f32(x_fp4, 2, 1)
    x_deq = x_deq.view(x_blocks.shape) * scale
    x_deq = x_deq.reshape(x2d.shape)

    if pad:
        x2d = x2d[:, :orig_cols]
        x_deq = x_deq[:, :orig_cols]

    return x2d, x_deq, clip_rate


def _compute_input_scale_from_tensor(x: torch.Tensor) -> float:
    amax = float(x.abs().amax().item())
    return max(amax / (F8_E4M3_MAX * F4_E2M1_MAX), 1e-12)


def _pct_key(pct: float) -> str:
    pct_str = str(pct).rstrip("0").rstrip(".")
    return pct_str.replace(".", "_")


def _load_summary_scales(
    path: str, percentile: float, multiplier: float
) -> Dict[str, float]:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    key = f"scale_p{_pct_key(percentile)}"
    out: Dict[str, float] = {}
    for row in summary.get("layers", []):
        layer = row.get("layer")
        if not layer:
            continue
        if key in row:
            val = float(row[key])
        elif "scale_max" in row:
            val = float(row["scale_max"])
        elif "scale_mean" in row:
            val = float(row["scale_mean"])
        else:
            continue
        out[layer] = val * multiplier
    return out


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def _plot_histogram(values: List[float], title: str, out_path: Path, bins: int) -> None:
    if plt is None:
        print(f"Warning: matplotlib not available; skipping plot: {out_path}")
        return
    if not values:
        print(f"Warning: no values for plot: {out_path}")
        return

    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6

    tick_count = 20
    step = (vmax - vmin) / (tick_count - 1)
    ticks = [vmin + i * step for i in range(tick_count)]

    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins, color="#f58518", alpha=0.85)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("Count")
    plt.xticks(ticks, rotation=45, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze input precision sensitivity")
    parser.add_argument(
        "--model-type",
        required=True,
        choices=MODEL_TYPES,
        help="Model type for calibration (required)",
    )
    parser.add_argument(
        "--fp16-model",
        required=True,
        help="Path to FP16/FP32 model safetensors",
    )
    parser.add_argument(
        "--input-scale-summary-json",
        default=None,
        metavar="PATH",
        help="Per-layer input_scale summary JSON (optional)",
    )
    parser.add_argument(
        "--input-scale-summary-percentile",
        type=float,
        default=99.0,
        help="Percentile to use from summary JSON (default: 99; supports 99.9)",
    )
    parser.add_argument(
        "--input-scale-summary-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for summary-derived input_scale values (default: 1.0)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=6,
        help="Number of calibration samples (default: 6)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for calibration (default: cuda)",
    )
    parser.add_argument(
        "--comfyui-root",
        default=None,
        metavar="PATH",
        help="Path to ComfyUI root for model loading (optional)",
    )
    parser.add_argument(
        "--output-json",
        default="input_precision_sensitivity.json",
        help="Output JSON path (default: input_precision_sensitivity.json)",
    )
    parser.add_argument(
        "--plots-dir",
        default="images",
        help="Output directory for histogram plots (default: images)",
    )
    parser.add_argument(
        "--output-layers",
        default=None,
        help="Optional output layer list (for --full-precision-mm-layers)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Write top K layers to output list (0 disables)",
    )
    parser.add_argument(
        "--min-rel-rmse",
        type=float,
        default=0.0,
        help="Write layers with mean rel-RMSE >= threshold (default: 0.0)",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    input_scale_map: Dict[str, float] = {}
    if args.input_scale_summary_json:
        input_scale_map = _load_summary_scales(
            args.input_scale_summary_json,
            args.input_scale_summary_percentile,
            args.input_scale_summary_multiplier,
        )
        if input_scale_map:
            vals = list(input_scale_map.values())
            print(
                f"Loaded input_scale summary: count={len(vals)}, min={min(vals):.6f}, "
                f"max={max(vals):.6f}, mean={(sum(vals)/len(vals)):.6f}"
            )

    model = _build_model_for_calibration(
        args.fp16_model, args.model_type, args.device, args.comfyui_root
    )

    import torch.nn as nn

    stats: Dict[str, Dict[str, float]] = {}

    def make_hook(name: str):
        def hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            if x is None or not isinstance(x, torch.Tensor):
                return
            if x.dim() < 2:
                return

            if name in input_scale_map:
                input_scale = input_scale_map[name]
            else:
                input_scale = _compute_input_scale_from_tensor(x)

            x_ref, x_q, clip_rate = _quantize_input_fp4(x, input_scale)

            err = (x_ref - x_q)
            mse = float(err.pow(2).mean().item())
            denom = float(x_ref.pow(2).mean().item())
            rel_rmse = math.sqrt(mse) / (math.sqrt(denom) + 1e-12)
            mean_abs = float(err.abs().mean().item())
            max_abs = float(err.abs().max().item())

            s = stats.setdefault(
                name,
                {
                    "count": 0.0,
                    "rel_rmse_sum": 0.0,
                    "mean_abs_sum": 0.0,
                    "clip_rate_sum": 0.0,
                    "max_abs_max": 0.0,
                },
            )
            s["count"] += 1.0
            s["rel_rmse_sum"] += rel_rmse
            s["mean_abs_sum"] += mean_abs
            s["clip_rate_sum"] += clip_rate
            s["max_abs_max"] = max(s["max_abs_max"], max_abs)

        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    print(f"Registered {len(hooks)} activation hooks")

    if args.model_type in WAN_MODEL_TYPES:
        in_dim = model.patch_embedding.weight.shape[1]
        run_wan_calibration_passes(model, in_dim, args.samples, args.device)
    else:
        run_qwen_calibration_passes(model, args.samples, args.device)

    for h in hooks:
        h.remove()

    rows: List[Dict[str, float]] = []
    for layer, s in stats.items():
        count = s.get("count", 0.0)
        if count <= 0:
            continue
        rows.append(
            {
                "layer": layer,
                "count": int(count),
                "rel_rmse_mean": _safe_div(s["rel_rmse_sum"], count),
                "mean_abs_mean": _safe_div(s["mean_abs_sum"], count),
                "clip_rate_mean": _safe_div(s["clip_rate_sum"], count),
                "max_abs": s["max_abs_max"],
            }
        )

    rows.sort(key=lambda r: r["rel_rmse_mean"], reverse=True)
    out_json = {
        "model": args.fp16_model,
        "model_type": args.model_type,
        "samples": args.samples,
        "layers": rows,
    }
    Path(args.output_json).write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output_json}")

    # Histograms
    if rows:
        bins = len(rows)
        plots_dir = Path(args.plots_dir)
        rel_rmse_values = [r["rel_rmse_mean"] for r in rows if math.isfinite(r["rel_rmse_mean"])]
        clip_rate_values = [r["clip_rate_mean"] for r in rows if math.isfinite(r["clip_rate_mean"])]

        _plot_histogram(
            rel_rmse_values,
            "rel_rmse_mean",
            plots_dir / f"rel_rmse_mean_hist_{bins}.png",
            bins,
        )
        _plot_histogram(
            clip_rate_values,
            "clip_rate_mean",
            plots_dir / f"clip_rate_mean_hist_{bins}.png",
            bins,
        )

    if args.output_layers:
        selected: List[str] = []
        if args.top_k > 0:
            selected = [r["layer"] for r in rows[: args.top_k]]
        if args.min_rel_rmse > 0:
            selected = [
                r["layer"] for r in rows if r["rel_rmse_mean"] >= args.min_rel_rmse
            ]
        if args.top_k > 0 and args.min_rel_rmse > 0:
            by_k = set(r["layer"] for r in rows[: args.top_k])
            by_thr = set(
                r["layer"] for r in rows if r["rel_rmse_mean"] >= args.min_rel_rmse
            )
            selected = sorted(by_k.union(by_thr))

        Path(args.output_layers).write_text("\n".join(selected), encoding="utf-8")
        print(f"Wrote: {args.output_layers} (layers={len(selected)})")


if __name__ == "__main__":
    main()
