#!/usr/bin/env python3
"""
Create 3 test WAN models with corrected input_scale values:
1. Constant Flux median (0.024) for all layers
2. Scaled calibrated (multiply existing by 44x)
3. Flux-ratio matched (scale to match Flux's ratio distribution)
"""

import safetensors.torch
import torch
from pathlib import Path

# Load the CALIBRATED model as base
calibrated_path = Path(
    r"D:\ComfyUI\ComfyUI\models\diffusion_models\wan2.2-ti2v-5b-nvfp4-CALIBRATED.safetensors"
)
print(f"Loading CALIBRATED model from: {calibrated_path}")
state = safetensors.torch.load_file(calibrated_path)
print(f"Loaded {len(state)} tensors")
print()

# Flux reference values
FLUX_MEDIAN_INPUT_SCALE = 0.023940
FLUX_MEAN_RATIO = 765.0
WAN_CALIBRATED_MEAN_RATIO = 22.7
SCALE_MULTIPLIER = FLUX_MEAN_RATIO / WAN_CALIBRATED_MEAN_RATIO  # ~33.7x

print(f"Flux median input_scale: {FLUX_MEDIAN_INPUT_SCALE:.6f}")
print(f"Flux mean ratio: {FLUX_MEAN_RATIO:.1f}x")
print(f"WAN calibrated mean ratio: {WAN_CALIBRATED_MEAN_RATIO:.1f}x")
print(f"Scale multiplier: {SCALE_MULTIPLIER:.1f}x")
print()

# Find all input_scale tensors
input_scale_keys = [k for k in state.keys() if k.endswith(".input_scale")]
print(f"Found {len(input_scale_keys)} input_scale tensors")
print()

# ============================================================================
# MODEL 1: Constant Flux Median
# ============================================================================

print("=" * 80)
print("MODEL 1: CONSTANT FLUX MEDIAN")
print("=" * 80)
print()

state1 = dict(state)  # Copy
constant_value = torch.tensor([FLUX_MEDIAN_INPUT_SCALE], dtype=torch.float32)

for key in input_scale_keys:
    state1[key] = constant_value.clone()

# Verify
sample_key = input_scale_keys[0]
print(f"Sample layer: {sample_key}")
print(f"  Old value: {state[sample_key].item():.6f}")
print(f"  New value: {state1[sample_key].item():.6f}")
print(f"  Multiplier: {state1[sample_key].item() / state[sample_key].item():.1f}x")
print()

output1 = Path(
    r"D:\ComfyUI\ComfyUI\models\diffusion_models\wan2.2-ti2v-5b-nvfp4-FLUX-MEDIAN.safetensors"
)
print(f"Saving to: {output1}")
safetensors.torch.save_file(state1, output1)
print("Saved!")
print()

# ============================================================================
# MODEL 2: Scaled Calibrated (33.7x multiplier to match Flux ratio)
# ============================================================================

print("=" * 80)
print("MODEL 2: SCALED CALIBRATED (33.7x)")
print("=" * 80)
print()

state2 = dict(state)  # Copy

for key in input_scale_keys:
    old_value = state[key]
    new_value = old_value * SCALE_MULTIPLIER
    state2[key] = new_value

# Verify
print(f"Sample layer: {sample_key}")
print(f"  Old value: {state[sample_key].item():.6f}")
print(f"  New value: {state2[sample_key].item():.6f}")
print(f"  Multiplier: {state2[sample_key].item() / state[sample_key].item():.1f}x")
print()

output2 = Path(
    r"D:\ComfyUI\ComfyUI\models\diffusion_models\wan2.2-ti2v-5b-nvfp4-SCALED-CALIBRATED.safetensors"
)
print(f"Saving to: {output2}")
safetensors.torch.save_file(state2, output2)
print("Saved!")
print()

# ============================================================================
# MODEL 3: Per-layer ratio matching (scale each layer to target ratio)
# ============================================================================

print("=" * 80)
print("MODEL 3: PER-LAYER FLUX RATIO MATCHING")
print("=" * 80)
print()

state3 = dict(state)  # Copy

# For each layer, compute weight_scale_2 and scale input_scale to achieve target ratio
target_ratio = FLUX_MEAN_RATIO

for key in input_scale_keys:
    base_key = key.replace(".input_scale", "")
    ws2_key = f"{base_key}.weight_scale_2"

    if ws2_key in state:
        weight_scale_2 = state[ws2_key].item()
        # Target: input_scale / weight_scale_2 = target_ratio
        # Therefore: input_scale = weight_scale_2 * target_ratio
        new_input_scale = torch.tensor(
            [weight_scale_2 * target_ratio], dtype=torch.float32
        )
        state3[key] = new_input_scale
    else:
        print(f"WARNING: No weight_scale_2 for {base_key}, using constant")
        state3[key] = torch.tensor([FLUX_MEDIAN_INPUT_SCALE], dtype=torch.float32)

# Verify
print(f"Sample layer: {sample_key}")
base_key = sample_key.replace(".input_scale", "")
ws2_key = f"{base_key}.weight_scale_2"
ws2_value = state[ws2_key].item()
print(f"  weight_scale_2: {ws2_value:.6f}")
print(f"  Old input_scale: {state[sample_key].item():.6f}")
print(f"  Old ratio: {state[sample_key].item() / ws2_value:.1f}x")
print(f"  New input_scale: {state3[sample_key].item():.6f}")
print(f"  New ratio: {state3[sample_key].item() / ws2_value:.1f}x")
print()

output3 = Path(
    r"D:\ComfyUI\ComfyUI\models\diffusion_models\wan2.2-ti2v-5b-nvfp4-RATIO-MATCHED.safetensors"
)
print(f"Saving to: {output3}")
safetensors.torch.save_file(state3, output3)
print("Saved!")
print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print("Created 3 test models:")
print()
print(f"1. FLUX-MEDIAN: {output1.name}")
print(f"   - All input_scale = {FLUX_MEDIAN_INPUT_SCALE:.6f} (constant)")
print(f"   - Simple heuristic approach")
print()
print(f"2. SCALED-CALIBRATED: {output2.name}")
print(f"   - Multiply all input_scale by {SCALE_MULTIPLIER:.1f}x")
print(f"   - Preserves per-layer variation")
print()
print(f"3. RATIO-MATCHED: {output3.name}")
print(f"   - Each layer scaled to achieve ratio = {target_ratio:.1f}x")
print(f"   - Matches Flux's mean ratio exactly")
print()

print("RECOMMENDATION: Test RATIO-MATCHED first (most likely to work)")
print("It matches Flux's input_scale/weight_scale_2 ratio exactly while preserving")
print("per-layer weight_scale_2 values.")
print()
