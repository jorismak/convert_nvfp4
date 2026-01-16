#!/usr/bin/env python3
"""
Analyze WAN NVFP4 models to compare with Flux structure
"""

import safetensors.torch
import torch
from pathlib import Path
import numpy as np

models = [
    (
        "CALIBRATED",
        r"D:\ComfyUI\ComfyUI\models\diffusion_models\wan2.2-ti2v-5b-nvfp4-CALIBRATED.safetensors",
    ),
    (
        "WITH-INPUT-SCALE",
        r"D:\ComfyUI\ComfyUI\models\diffusion_models\wan2.2-ti2v-5b-nvfp4-WITH-INPUT-SCALE.safetensors",
    ),
]

for name, path in models:
    print("=" * 80)
    print(f"ANALYZING: {name}")
    print("=" * 80)
    print()

    if not Path(path).exists():
        print(f"Model not found: {path}")
        print()
        continue

    state = safetensors.torch.load_file(path)

    # Categorize tensors
    weight_scale_2_tensors = []
    input_scale_tensors = []

    for key in state.keys():
        if key.endswith(".weight_scale_2"):
            weight_scale_2_tensors.append(key)
        elif key.endswith(".input_scale"):
            input_scale_tensors.append(key)

    print(f"Tensor counts:")
    print(f"  Weight scale 2 (.weight_scale_2): {len(weight_scale_2_tensors)}")
    print(f"  Input scales (.input_scale):      {len(input_scale_tensors)}")
    print()

    if len(input_scale_tensors) == 0:
        print("No input_scale tensors found!")
        print()
        continue

    # Extract base names
    quantized_layers = set()
    for key in weight_scale_2_tensors:
        base = key.replace(".weight_scale_2", "")
        quantized_layers.add(base)

    # Check completeness
    missing_input = []
    for base in quantized_layers:
        if f"{base}.input_scale" not in state:
            missing_input.append(base)

    if missing_input:
        print(f"WARNING: {len(missing_input)} layers missing input_scale!")
        for layer in missing_input[:5]:
            print(f"  {layer}")
        if len(missing_input) > 5:
            print(f"  ... and {len(missing_input) - 5} more")
    else:
        print("[OK] All quantized layers have input_scale")
    print()

    # Sample first 3 layers
    print("=" * 80)
    print("SAMPLE LAYER ANALYSIS (First 3 quantized layers)")
    print("=" * 80)
    print()

    for i, base in enumerate(sorted(quantized_layers)[:3]):
        print(f"Layer {i + 1}: {base}")
        print("-" * 80)

        weight_scale_2 = state[f"{base}.weight_scale_2"]
        input_scale = state[f"{base}.input_scale"]

        print(
            f"  weight_scale_2: shape={weight_scale_2.shape}, dtype={weight_scale_2.dtype}, value={weight_scale_2.item():.6f}"
        )
        print(
            f"  input_scale:   shape={input_scale.shape}, dtype={input_scale.dtype}, value={input_scale.item():.6f}"
        )

        ratio = input_scale.item() / weight_scale_2.item()
        print(f"  Ratio (input/weight_scale_2): {ratio:.2f}x")
        print()

    # Statistical analysis
    print("=" * 80)
    print("STATISTICAL ANALYSIS (All quantized layers)")
    print("=" * 80)
    print()

    weight_scale_2_values = []
    input_scale_values = []
    ratios = []

    for base in quantized_layers:
        ws2 = state[f"{base}.weight_scale_2"].item()
        inp = state[f"{base}.input_scale"].item()
        weight_scale_2_values.append(ws2)
        input_scale_values.append(inp)
        ratios.append(inp / ws2)

    weight_scale_2_values = np.array(weight_scale_2_values)
    input_scale_values = np.array(input_scale_values)
    ratios = np.array(ratios)

    print("weight_scale_2 distribution:")
    print(f"  Min:    {weight_scale_2_values.min():.6f}")
    print(f"  Max:    {weight_scale_2_values.max():.6f}")
    print(f"  Mean:   {weight_scale_2_values.mean():.6f}")
    print(f"  Median: {np.median(weight_scale_2_values):.6f}")
    print(f"  Std:    {weight_scale_2_values.std():.6f}")
    print()

    print("input_scale distribution:")
    print(f"  Min:    {input_scale_values.min():.6f}")
    print(f"  Max:    {input_scale_values.max():.6f}")
    print(f"  Mean:   {input_scale_values.mean():.6f}")
    print(f"  Median: {np.median(input_scale_values):.6f}")
    print(f"  Std:    {input_scale_values.std():.6f}")
    print()

    print("Ratio (input_scale / weight_scale_2) distribution:")
    print(f"  Min:    {ratios.min():.1f}x")
    print(f"  Max:    {ratios.max():.1f}x")
    print(f"  Mean:   {ratios.mean():.1f}x")
    print(f"  Median: {np.median(ratios):.1f}x")
    print(f"  Std:    {ratios.std():.1f}x")
    print()

    # Compare to Flux
    print("=" * 80)
    print("COMPARISON TO FLUX")
    print("=" * 80)
    print()

    flux_ratio_mean = 765.0
    flux_ws2_mean = 0.000136
    flux_input_mean = 0.097332

    print(f"Flux ratio mean:   {flux_ratio_mean:.1f}x")
    print(f"WAN ratio mean:    {ratios.mean():.1f}x")
    print(f"Difference:        {flux_ratio_mean / ratios.mean():.1f}x too small")
    print()

    print(f"Flux weight_scale_2 mean:   {flux_ws2_mean:.6f}")
    print(f"WAN weight_scale_2 mean:    {weight_scale_2_values.mean():.6f}")
    print(
        f"WAN/Flux ratio:             {weight_scale_2_values.mean() / flux_ws2_mean:.2f}x"
    )
    print()

    print(f"Flux input_scale mean:      {flux_input_mean:.6f}")
    print(f"WAN input_scale mean:       {input_scale_values.mean():.6f}")
    print(
        f"WAN/Flux ratio:             {input_scale_values.mean() / flux_input_mean:.2f}x"
    )
    print()

print("=" * 80)
print("Analysis complete!")
print("=" * 80)
