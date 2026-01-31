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

    if path.suffixes[-2:] == [".safetensors", ".index"] or path.name.endswith(
        ".safetensors.index.json"
    ):
        import json as _json

        index = _json.loads(path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        state_dict: Dict[str, torch.Tensor] = {}
        base_dir = path.parent
        for key, rel_path in weight_map.items():
            shard_path = base_dir / rel_path
            with safe_open(shard_path, framework="pt", device="cpu") as f:
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


class _SafeTensorLoader:
    def __init__(self, model_path: str) -> None:
        from safetensors import safe_open

        self._safe_open = safe_open
        self.path = Path(model_path)
        self._handles = {}

        if self.path.name.endswith(".safetensors.index.json"):
            import json as _json

            index = _json.loads(self.path.read_text(encoding="utf-8"))
            self.weight_map = index.get("weight_map", {})
            self.base_dir = self.path.parent
            self._keys = list(self.weight_map.keys())
            self._single_file = None
        elif self.path.suffix == ".safetensors":
            self.weight_map = None
            self.base_dir = self.path.parent
            self._single_file = self.path
            with self._safe_open(self._single_file, framework="pt", device="cpu") as f:
                self._keys = list(f.keys())
        else:
            raise ValueError("Low-mem loader only supports .safetensors or .safetensors.index.json")

    def keys(self):
        return self._keys

    def __contains__(self, key: str) -> bool:
        if self.weight_map is None:
            return key in self._keys
        return key in self.weight_map

    def _get_handle(self, path: Path):
        handle = self._handles.get(path)
        if handle is None:
            handle = self._safe_open(path, framework="pt", device="cpu")
            self._handles[path] = handle
        return handle

    def get_tensor(self, key: str) -> torch.Tensor:
        if self.weight_map is None:
            handle = self._get_handle(self._single_file)
            return handle.get_tensor(key)

        rel_path = self.weight_map.get(key)
        if rel_path is None:
            raise KeyError(f"Missing tensor key '{key}' in safetensors index")
        shard_path = self.base_dir / rel_path
        handle = self._get_handle(shard_path)
        return handle.get_tensor(key)

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
        self._handles.clear()


def _stream_load_state_dict(
    model, loader: _SafeTensorLoader, device: str
) -> None:
    param_map = dict(model.named_parameters())
    buffer_map = dict(model.named_buffers())
    missing = []

    with torch.no_grad():
        for key in model.state_dict().keys():
            if key not in loader:
                missing.append(key)
                continue
            tensor = loader.get_tensor(key)
            if key in param_map:
                target = param_map[key]
                target.copy_(tensor.to(dtype=target.dtype, device=target.device))
            elif key in buffer_map:
                target = buffer_map[key]
                target.copy_(tensor.to(dtype=target.dtype, device=target.device))

    if missing:
        print(f"  Missing keys during low-mem load: {len(missing)}")
    if device != "cpu":
        model.to(device=device)


def _build_wan_model_from_loader(
    loader: _SafeTensorLoader,
    device: str,
    comfyui_root: Optional[str],
    model_type: str,
):
    root = _resolve_comfyui_root(comfyui_root)
    sys.path.insert(0, str(root))

    from comfy.ldm.wan.model import WanModel
    import comfy.ops as ops

    keys = loader.keys()
    block_indices = [
        int(k.split(".")[1])
        for k in keys
        if k.startswith("blocks.") and k.split(".")[1].isdigit()
    ]
    num_blocks = (max(block_indices) + 1) if block_indices else 0

    hidden_size = loader.get_tensor("blocks.0.self_attn.q.weight").shape[0]
    ffn_dim = loader.get_tensor("blocks.0.ffn.0.weight").shape[0]
    text_dim = loader.get_tensor("text_embedding.0.weight").shape[1]
    patch_weight = loader.get_tensor("patch_embedding.weight")
    in_dim = patch_weight.shape[1]
    head_out = loader.get_tensor("head.head.weight").shape[0]
    out_dim = head_out // 4
    patch_size = tuple(int(v) for v in patch_weight.shape[-3:])

    flf_pos_embed_token_number = None
    if "img_emb.emb_pos" in loader:
        flf_pos_embed_token_number = int(loader.get_tensor("img_emb.emb_pos").shape[1])

    in_dim_ref_conv = None
    if "ref_conv.weight" in loader:
        in_dim_ref_conv = int(loader.get_tensor("ref_conv.weight").shape[1])

    resolved_model_type = {
        "wan22_5b": "i2v",
        "wan22_i2v_lownoise": "i2v",
        "wan22_i2v_highnoise": "i2v",
        "wan22_t2v_lownoise": "t2v",
        "wan22_t2v_highnoise": "t2v",
        "wan21_i2v_480p": "i2v",
    }[model_type]

    model = WanModel(
        model_type=resolved_model_type,
        patch_size=patch_size,
        in_dim=in_dim,
        dim=hidden_size,
        ffn_dim=ffn_dim,
        text_dim=text_dim,
        out_dim=out_dim,
        num_heads=hidden_size // 128,
        num_layers=num_blocks,
        flf_pos_embed_token_number=flf_pos_embed_token_number,
        in_dim_ref_conv=in_dim_ref_conv,
        dtype=torch.bfloat16,
        device="cpu",
        operations=ops.manual_cast,
    )

    model.eval()
    return model


def _build_qwen_model_from_loader(
    loader: _SafeTensorLoader,
    device: str,
    comfyui_root: Optional[str],
    model_type: str,
):
    root = _resolve_comfyui_root(comfyui_root)
    sys.path.insert(0, str(root))

    from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel
    import comfy.ops as ops

    img_in_weight = loader.get_tensor("img_in.weight")
    inner_dim, in_channels = img_in_weight.shape
    txt_in_weight = loader.get_tensor("txt_in.weight")
    joint_attention_dim = txt_in_weight.shape[1]

    norm_q_key = None
    for key in loader.keys():
        if key.endswith("attn.norm_q.weight"):
            norm_q_key = key
            break
    if norm_q_key is None:
        raise KeyError("Missing attn.norm_q.weight in Qwen model")

    attention_head_dim = loader.get_tensor(norm_q_key).shape[0]
    num_attention_heads = int(inner_dim // attention_head_dim)

    layer_indices = [
        int(k.split(".")[1])
        for k in loader.keys()
        if k.startswith("transformer_blocks.") and k.split(".")[1].isdigit()
    ]
    num_layers = (max(layer_indices) + 1) if layer_indices else 0

    proj_out_weight = loader.get_tensor("proj_out.weight") if "proj_out.weight" in loader else None
    if proj_out_weight is not None and proj_out_weight.ndim == 2:
        out_features = proj_out_weight.shape[0]
        patch_size, out_channels = 2, in_channels
        for ps in (2, 4, 1):
            denom = ps * ps
            if out_features % denom == 0:
                patch_size = ps
                out_channels = out_features // denom
                break
    else:
        patch_size, out_channels = 2, in_channels

    use_additional_t_cond = "time_text_embed.addition_t_embedding.weight" in loader
    default_ref_method = "index_timestep_zero" if "__index_timestep_zero__" in loader else "index"

    model = QwenImageTransformer2DModel(
        patch_size=patch_size,
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        num_layers=num_layers,
        attention_head_dim=int(attention_head_dim),
        num_attention_heads=num_attention_heads,
        joint_attention_dim=int(joint_attention_dim),
        pooled_projection_dim=768,
        default_ref_method=default_ref_method,
        use_additional_t_cond=use_additional_t_cond,
        dtype=torch.bfloat16,
        device="cpu",
        operations=ops.manual_cast,
    )

    model.eval()
    return model


def _build_model_for_calibration_low_mem(
    model_path: str,
    model_type: str,
    device: str,
    comfyui_root: Optional[str],
):
    loader = _SafeTensorLoader(model_path)
    try:
        if model_type in WAN_MODEL_TYPES:
            model = _build_wan_model_from_loader(loader, device, comfyui_root, model_type)
        else:
            model = _build_qwen_model_from_loader(loader, device, comfyui_root, model_type)

        _stream_load_state_dict(model, loader, device)
        model.eval()
        return model
    finally:
        loader.close()


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


def _select_clip_value(
    row: Dict,
    pct_key: str,
    ratio_threshold: float = 1.5,
    frac_threshold: float = 0.1,
) -> float:
    channel_max = row.get("channel_amax_max")
    channel_pct = row.get(f"channel_amax_p{pct_key}")
    if isinstance(channel_max, list) and isinstance(channel_pct, list):
        if len(channel_max) == len(channel_pct) and len(channel_max) > 0:
            hits = 0
            for max_val, pct_val in zip(channel_max, channel_pct):
                denom = max(float(pct_val), 1e-12)
                if float(max_val) / denom > ratio_threshold:
                    hits += 1
            frac = hits / len(channel_max)
            return 4.0 if frac >= frac_threshold else 6.0

    global_max = row.get("global_amax_max")
    global_pct = row.get(f"global_amax_p{pct_key}")
    if isinstance(global_max, (int, float)) and isinstance(global_pct, (int, float)):
        denom = max(float(global_pct), 1e-12)
        ratio = float(global_max) / denom
        return 4.0 if ratio > ratio_threshold else 6.0

    return 6.0


def _load_summary_scales(
    path: str, percentile: float, multiplier: float, clip: str
) -> Dict[str, float]:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    pct_key = _pct_key(percentile)
    channel_key = f"channel_amax_p{pct_key}"
    global_key = f"global_amax_p{pct_key}"
    out: Dict[str, float] = {}
    for row in summary.get("layers", []):
        layer = row.get("layer")
        if not layer:
            continue
        if clip == "auto":
            clip_value = _select_clip_value(row, pct_key)
        else:
            clip_value = float(clip)

        if channel_key in row and isinstance(row[channel_key], list):
            vals = [float(v) for v in row[channel_key] if v is not None]
            if not vals:
                continue
            amax = sum(vals) / len(vals)
            scale = amax / (F8_E4M3_MAX * clip_value)
        elif global_key in row:
            amax = float(row[global_key])
            scale = amax / (F8_E4M3_MAX * clip_value)
        elif "global_amax_max" in row:
            amax = float(row["global_amax_max"])
            scale = amax / (F8_E4M3_MAX * clip_value)
        elif "scale_max" in row:
            scale = float(row["scale_max"])
        elif "scale_mean" in row:
            scale = float(row["scale_mean"])
        else:
            continue
        out[layer] = scale * multiplier
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
        "--input-scale-clip",
        choices=["4", "6", "auto"],
        default="6",
        help="Activation clip max for input_scale (4 or 6, default: 6)",
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
        "--low-mem",
        action="store_true",
        help="Stream weights from safetensors to reduce peak RAM (supports .safetensors or .safetensors.index.json)",
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
        default="plots",
        help="Output directory for histogram plots (default: plots)",
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
            args.input_scale_clip,
        )
        if input_scale_map:
            vals = list(input_scale_map.values())
            print(
                f"Loaded input_scale summary: count={len(vals)}, min={min(vals):.6f}, "
                f"max={max(vals):.6f}, mean={(sum(vals)/len(vals)):.6f}"
            )

    if args.low_mem:
        model = _build_model_for_calibration_low_mem(
            args.fp16_model, args.model_type, args.device, args.comfyui_root
        )
    else:
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
