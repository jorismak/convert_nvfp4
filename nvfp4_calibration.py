#!/usr/bin/env python3
"""
Calibration helpers for NVFP4 conversion.

This module contains WAN activation-based input_scale calibration utilities
that are kept separate from the main converter to reduce clutter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch

F4_E2M1_MAX = 6.0
F8_E4M3_MAX = 448.0


class ActivationCollector:
    """Collects input activation statistics for linear layers."""

    def __init__(self) -> None:
        self.amax_values: Dict[str, List[float]] = {}
        self.hooks = []

    def make_hook(self, name: str):
        def hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            if x is not None and isinstance(x, torch.Tensor):
                amax = x.abs().amax().item()
                self.amax_values.setdefault(name, []).append(amax)

        return hook

    def register_hooks(self, model, layer_names: Set[str]) -> None:
        import torch.nn as nn

        for name, module in model.named_modules():
            if name in layer_names and isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self.make_hook(name))
                self.hooks.append(hook)
        print(f"Registered {len(self.hooks)} activation hooks")

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_input_scales(self, method: str = "max") -> Dict[str, float]:
        input_scales = {}
        for name, amaxes in self.amax_values.items():
            if not amaxes:
                continue
            if method == "max":
                amax = max(amaxes)
            elif method == "mean":
                amax = sum(amaxes) / len(amaxes)
            elif method == "percentile_99":
                sorted_amaxes = sorted(amaxes)
                idx = int(len(sorted_amaxes) * 0.99)
                amax = sorted_amaxes[min(idx, len(sorted_amaxes) - 1)]
            else:
                raise ValueError(f"Unknown input_scale method: {method}")

            scale = amax / (F8_E4M3_MAX * F4_E2M1_MAX)
            input_scales[name] = max(scale, 1e-12)

        return input_scales


def _resolve_comfyui_root(user_root: Optional[str]) -> Path:
    if user_root:
        return Path(user_root)
    return Path(__file__).resolve().parents[1]


def build_wan_model_for_calibration(
    state_dict: Dict[str, torch.Tensor],
    device: str,
    comfyui_root: Optional[str],
):
    root = _resolve_comfyui_root(comfyui_root)
    sys.path.insert(0, str(root))

    from comfy.ldm.wan.model import WanModel
    import comfy.ops as ops

    hidden_size = state_dict["blocks.0.self_attn.q.weight"].shape[0]
    num_blocks = (
        max(int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("blocks."))
        + 1
    )
    text_dim = state_dict["text_embedding.0.weight"].shape[1]
    ffn_dim = state_dict["blocks.0.ffn.0.weight"].shape[0]
    in_dim = state_dict["patch_embedding.weight"].shape[1]
    head_out = state_dict["head.head.weight"].shape[0]
    out_dim = head_out // 4

    model = WanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        in_dim=in_dim,
        dim=hidden_size,
        ffn_dim=ffn_dim,
        text_dim=text_dim,
        out_dim=out_dim,
        num_heads=hidden_size // 128,
        num_layers=num_blocks,
        dtype=torch.bfloat16,
        device="cpu",
        operations=ops.manual_cast,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys during calibration load: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys during calibration load: {len(unexpected)}")

    model = model.to(device=device)
    model.eval()
    return model


def run_wan_calibration_passes(
    model,
    in_dim: int,
    num_samples: int,
    device: str,
    batch_size: int = 1,
):
    text_dim = model.text_dim
    text_len = 512

    latent_channels = in_dim
    latent_t = 5
    latent_h = 30
    latent_w = 52

    for i in range(num_samples):
        x = torch.randn(
            batch_size,
            latent_channels,
            latent_t,
            latent_h,
            latent_w,
            device=device,
            dtype=torch.bfloat16,
        )
        timestep = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)
        context = torch.randn(
            batch_size, text_len, text_dim, device=device, dtype=torch.bfloat16
        )

        with torch.no_grad():
            try:
                _ = model(x, timestep, context)
            except Exception as e:
                print(f"Calibration forward error: {e}")
                break

        if i % 2 == 0 and device == "cuda":
            torch.cuda.empty_cache()


def compute_input_scales_from_fp16_state_dict(
    state_dict: Dict[str, torch.Tensor],
    layer_names: Set[str],
    method: str,
    num_samples: int,
    device: str,
    comfyui_root: Optional[str],
) -> Dict[str, torch.Tensor]:
    print("Computing input_scale from FP16/FP32 WAN model activations...")
    model = build_wan_model_for_calibration(state_dict, device, comfyui_root)

    collector = ActivationCollector()
    collector.register_hooks(model, layer_names)

    in_dim = model.patch_embedding.weight.shape[1]
    run_wan_calibration_passes(model, in_dim, num_samples, device)

    collector.remove_hooks()
    input_scales = collector.get_input_scales(method=method)

    if not input_scales:
        print("Warning: No activation scales collected from FP16 model.")
        return {}

    scale_values = list(input_scales.values())
    print(
        f"input_scale range: {min(scale_values):.6f} - {max(scale_values):.6f} "
        f"(mean {sum(scale_values) / len(scale_values):.6f})"
    )

    return {k: torch.tensor([v], dtype=torch.float32) for k, v in input_scales.items()}


def calibrate_layer(
    weight: torch.Tensor,
    num_steps: int = 8,
    batch_size: int = 4,
    seq_len: int = 1024,
) -> torch.Tensor:
    """
    Basic calibration to estimate input_scale for a Linear layer using random inputs.

    Args:
        weight: Weight tensor of shape (out_features, in_features)
        num_steps: Number of calibration steps
        batch_size: Batch size for calibration
        seq_len: Sequence length for calibration

    Returns:
        input_scale tensor (scalar)
    """
    in_features = weight.shape[1]
    device = weight.device
    dtype = weight.dtype

    amax = torch.tensor(0.0, device=device, dtype=torch.float32)

    for _ in range(num_steps):
        x = torch.randn(batch_size, seq_len, in_features, device=device, dtype=dtype)
        amax = torch.maximum(amax, torch.amax(x.abs()))

    input_scale = amax / (F8_E4M3_MAX * F4_E2M1_MAX)
    input_scale = torch.clamp(input_scale, min=1e-12)

    return input_scale
