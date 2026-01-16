#!/usr/bin/env python3
"""
Analyze Flux NVFP4 model structure in detail to understand:
1. How input_scale relates to weight_scale_2
2. What the actual quantization structure looks like
3. Whether our assumptions about the format are correct
"""

import safetensors.torch
import torch
from pathlib import Path
import numpy as np

flux_path = Path(
    r"D:\ComfyUI\ComfyUI\models\diffusion_models\flux-2-klein-9b-nvfp4.safetensors"
)
print(f"Loading Flux NVFP4 model from: {flux_path}")
print()

# Load the model
flux_state = safetensors.torch.load_file(flux_path)

print("=" * 80)
print("FLUX NVFP4 STRUCTURE ANALYSIS")
print("=" * 80)
print()

# Categorize tensors
weight_tensors = []
weight_scale_tensors = []
weight_scale_2_tensors = []
input_scale_tensors = []
bias_tensors = []
other_tensors = []

for key in flux_state.keys():
    if key.endswith(".weight") and not any(x in key for x in ["scale", "input"]):
        weight_tensors.append(key)
    elif key.endswith(".weight_scale"):
        weight_scale_tensors.append(key)
    elif key.endswith(".weight_scale_2"):
        weight_scale_2_tensors.append(key)
    elif key.endswith(".input_scale"):
        input_scale_tensors.append(key)
    elif key.endswith(".bias"):
        bias_tensors.append(key)
    else:
        other_tensors.append(key)

print(f"Tensor counts:")
print(f"  Weights (.weight):          {len(weight_tensors)}")
print(f"  Weight scales (.weight_scale):    {len(weight_scale_tensors)}")
print(f"  Weight scale 2 (.weight_scale_2): {len(weight_scale_2_tensors)}")
print(f"  Input scales (.input_scale):      {len(input_scale_tensors)}")
print(f"  Biases (.bias):             {len(bias_tensors)}")
print(f"  Other:                      {len(other_tensors)}")
print()

# Check if all quantized layers have all required tensors
print("=" * 80)
print("QUANTIZED LAYER COMPLETENESS")
print("=" * 80)
print()

# Extract base names from weight_scale_2 (these are the quantized layers)
quantized_layers = set()
for key in weight_scale_2_tensors:
    base = key.replace(".weight_scale_2", "")
    quantized_layers.add(base)

print(f"Found {len(quantized_layers)} quantized layers")
print()

# Check each quantized layer has all 4 required tensors
missing_components = []
for base in sorted(quantized_layers):
    has_weight = f"{base}.weight" in flux_state
    has_scale = f"{base}.weight_scale" in flux_state
    has_scale2 = f"{base}.weight_scale_2" in flux_state
    has_input = f"{base}.input_scale" in flux_state

    if not (has_weight and has_scale and has_scale2 and has_input):
        missing_components.append(
            {
                "layer": base,
                "weight": has_weight,
                "scale": has_scale,
                "scale2": has_scale2,
                "input": has_input,
            }
        )

if missing_components:
    print("WARNING: Some quantized layers are missing components!")
    for item in missing_components[:5]:  # Show first 5
        print(
            f"  {item['layer']}: W={item['weight']} S={item['scale']} S2={item['scale2']} I={item['input']}"
        )
    if len(missing_components) > 5:
        print(f"  ... and {len(missing_components) - 5} more")
else:
    print(
        "[OK] All quantized layers have all 4 required tensors (weight, weight_scale, weight_scale_2, input_scale)"
    )
print()

# Sample a few layers to understand the structure
print("=" * 80)
print("SAMPLE LAYER ANALYSIS (First 3 quantized layers)")
print("=" * 80)
print()

for i, base in enumerate(sorted(quantized_layers)[:3]):
    print(f"Layer {i + 1}: {base}")
    print("-" * 80)

    weight = flux_state[f"{base}.weight"]
    weight_scale = flux_state[f"{base}.weight_scale"]
    weight_scale_2 = flux_state[f"{base}.weight_scale_2"]
    input_scale = flux_state[f"{base}.input_scale"]

    print(
        f"  weight:        shape={weight.shape}, dtype={weight.dtype}, size={weight.numel()}"
    )
    print(f"  weight_scale:  shape={weight_scale.shape}, dtype={weight_scale.dtype}")
    print(
        f"  weight_scale_2: shape={weight_scale_2.shape}, dtype={weight_scale_2.dtype}, value={weight_scale_2.item():.6f}"
    )
    print(
        f"  input_scale:   shape={input_scale.shape}, dtype={input_scale.dtype}, value={input_scale.item():.6f}"
    )

    # Calculate ratio
    ratio = input_scale.item() / weight_scale_2.item()
    print(f"  Ratio (input/weight_scale_2): {ratio:.2f}x")

    # Check if weight is packed (uint8)
    if weight.dtype == torch.uint8:
        print(f"  Weight is packed uint8 (2 FP4 values per byte)")
        # Calculate expected unpacked size
        unpacked_size = weight.numel() * 2
        print(f"  Unpacked size would be: {unpacked_size} FP4 values")
        # Calculate expected scale blocks
        block_size = 16
        num_blocks = unpacked_size // block_size
        print(f"  Expected weight_scale blocks (size/16): {num_blocks}")
        print(f"  Actual weight_scale size: {weight_scale.numel()}")
        if weight_scale.numel() == num_blocks:
            print(f"  [OK] Scale blocks match expected count")
        else:
            print(
                f"  [ERROR] Scale mismatch! Expected {num_blocks}, got {weight_scale.numel()}"
            )

    print()

# Statistical analysis of all quantized layers
print("=" * 80)
print("STATISTICAL ANALYSIS (All quantized layers)")
print("=" * 80)
print()

weight_scale_2_values = []
input_scale_values = []
ratios = []

for base in quantized_layers:
    ws2 = flux_state[f"{base}.weight_scale_2"].item()
    inp = flux_state[f"{base}.input_scale"].item()
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

# Check for patterns
print("=" * 80)
print("PATTERN ANALYSIS")
print("=" * 80)
print()

# Are input_scale values constant or varying?
unique_input = len(np.unique(np.round(input_scale_values, 6)))
print(
    f"Unique input_scale values (rounded to 6 decimals): {unique_input} out of {len(input_scale_values)}"
)
if unique_input == 1:
    print("  -> Input scales are CONSTANT across all layers")
elif unique_input < len(input_scale_values) * 0.1:
    print("  -> Input scales have LOW variation (mostly same values)")
else:
    print("  -> Input scales VARY per layer")
print()

# Are weight_scale_2 values constant or varying?
unique_ws2 = len(np.unique(np.round(weight_scale_2_values, 6)))
print(
    f"Unique weight_scale_2 values (rounded to 6 decimals): {unique_ws2} out of {len(weight_scale_2_values)}"
)
if unique_ws2 == 1:
    print("  -> Weight scale 2 values are CONSTANT across all layers")
elif unique_ws2 < len(weight_scale_2_values) * 0.1:
    print("  -> Weight scale 2 values have LOW variation")
else:
    print("  -> Weight scale 2 values VARY per layer")
print()

# Check if there's a formula relationship
print("Testing formula relationships:")
print(f"  If input_scale = weight_scale_2 * constant:")
mean_ratio = ratios.mean()
predicted = weight_scale_2_values * mean_ratio
error = np.abs(predicted - input_scale_values).mean()
print(f"    Mean ratio: {mean_ratio:.1f}x, Mean error: {error:.6f}")
if error < 0.001:
    print(
        f"    [OK] Strong linear relationship! input_scale ≈ weight_scale_2 * {mean_ratio:.1f}"
    )
else:
    print(f"    [ERROR] Not a simple linear relationship")
print()

# Check layer naming patterns
print("=" * 80)
print("LAYER NAMING PATTERNS")
print("=" * 80)
print()

# Group by layer type
layer_types = {}
for base in quantized_layers:
    # Extract layer type (e.g., "double_blocks.0.img_attn.qkv")
    parts = base.split(".")
    if len(parts) >= 2:
        layer_type = ".".join(parts[-2:]) if not parts[-1].isdigit() else parts[-1]
    else:
        layer_type = parts[-1]

    if layer_type not in layer_types:
        layer_types[layer_type] = []
    layer_types[layer_type].append(base)

print(f"Found {len(layer_types)} different layer type patterns:")
for ltype, layers in sorted(layer_types.items(), key=lambda x: len(x[1]), reverse=True)[
    :10
]:
    print(f"  {ltype}: {len(layers)} layers")
    # Show ratio distribution for this type
    type_ratios = []
    for base in layers:
        ws2 = flux_state[f"{base}.weight_scale_2"].item()
        inp = flux_state[f"{base}.input_scale"].item()
        type_ratios.append(inp / ws2)
    type_ratios = np.array(type_ratios)
    print(
        f"    Ratio range: {type_ratios.min():.1f}x - {type_ratios.max():.1f}x (mean: {type_ratios.mean():.1f}x)"
    )

print()
print("Analysis complete!")
