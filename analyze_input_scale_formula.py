#!/usr/bin/env python3
"""
Understand the input_scale formula by working backwards from Flux values
"""

import numpy as np

# Constants
F8_E4M3_MAX = 448.0
F4_E2M1_MAX = 6.0
NVFP4_SCALE_DIVISOR = F8_E4M3_MAX * F4_E2M1_MAX  # 2688.0

print("=" * 80)
print("REVERSE ENGINEERING FLUX INPUT_SCALE VALUES")
print("=" * 80)
print()

# Flux statistics from our analysis
flux_input_scale_min = 0.003962
flux_input_scale_max = 2.271429
flux_input_scale_mean = 0.097332
flux_input_scale_median = 0.023940

# WAN calibrated statistics
wan_input_scale_min = 0.002046
wan_input_scale_max = 0.002453
wan_input_scale_mean = 0.002208
wan_input_scale_median = 0.002197

print("FLUX INPUT_SCALE VALUES:")
print(f"  Min:    {flux_input_scale_min:.6f}")
print(f"  Max:    {flux_input_scale_max:.6f}")
print(f"  Mean:   {flux_input_scale_mean:.6f}")
print(f"  Median: {flux_input_scale_median:.6f}")
print()

print("WAN CALIBRATED INPUT_SCALE VALUES:")
print(f"  Min:    {wan_input_scale_min:.6f}")
print(f"  Max:    {wan_input_scale_max:.6f}")
print(f"  Mean:   {wan_input_scale_mean:.6f}")
print(f"  Median: {wan_input_scale_median:.6f}")
print()

print("=" * 80)
print("WORKING BACKWARDS: WHAT AMAX VALUES PRODUCED THESE SCALES?")
print("=" * 80)
print()

print("Formula: input_scale = amax / 2688.0")
print("Therefore: amax = input_scale * 2688.0")
print()

print("FLUX IMPLIED ACTIVATION RANGES (amax):")
flux_amax_min = flux_input_scale_min * NVFP4_SCALE_DIVISOR
flux_amax_max = flux_input_scale_max * NVFP4_SCALE_DIVISOR
flux_amax_mean = flux_input_scale_mean * NVFP4_SCALE_DIVISOR
flux_amax_median = flux_input_scale_median * NVFP4_SCALE_DIVISOR

print(f"  Min:    {flux_amax_min:.2f}")
print(f"  Max:    {flux_amax_max:.2f}")
print(f"  Mean:   {flux_amax_mean:.2f}")
print(f"  Median: {flux_amax_median:.2f}")
print()

print("WAN IMPLIED ACTIVATION RANGES (amax):")
wan_amax_min = wan_input_scale_min * NVFP4_SCALE_DIVISOR
wan_amax_max = wan_input_scale_max * NVFP4_SCALE_DIVISOR
wan_amax_mean = wan_input_scale_mean * NVFP4_SCALE_DIVISOR
wan_amax_median = wan_input_scale_median * NVFP4_SCALE_DIVISOR

print(f"  Min:    {wan_amax_min:.2f}")
print(f"  Max:    {wan_amax_max:.2f}")
print(f"  Mean:   {wan_amax_mean:.2f}")
print(f"  Median: {wan_amax_median:.2f}")
print()

print("=" * 80)
print("COMPARISON")
print("=" * 80)
print()

ratio_mean = flux_amax_mean / wan_amax_mean
ratio_median = flux_amax_median / wan_amax_median

print(f"Flux amax / WAN amax (mean):   {ratio_mean:.1f}x")
print(f"Flux amax / WAN amax (median): {ratio_median:.1f}x")
print()

print("INTERPRETATION:")
print()
if ratio_mean > 30:
    print(
        f"Flux expects activations with amax ~{flux_amax_mean:.1f} (range: {flux_amax_min:.1f} - {flux_amax_max:.1f})"
    )
    print(
        f"WAN calibration measured amax ~{wan_amax_mean:.1f} (range: {wan_amax_min:.1f} - {wan_amax_max:.1f})"
    )
    print()
    print(
        f"Our WAN calibration is finding activations {ratio_mean:.1f}x SMALLER than Flux expects!"
    )
    print()
    print("POSSIBLE EXPLANATIONS:")
    print("  1. Our calibration is using wrong/too-small inputs (e.g., noise, zeros)")
    print("  2. Flux's input_scale was NOT calibrated but uses a heuristic/constant")
    print("  3. Flux's input_scale serves a different purpose than we think")
    print("  4. WAN architecture has genuinely smaller activation ranges than Flux")

print()
print("=" * 80)
print("TESTING ALTERNATIVE HYPOTHESES")
print("=" * 80)
print()

# Hypothesis 1: Flux uses FP8-only for inputs (not NVFP4)
print("HYPOTHESIS 1: Flux quantizes inputs to FP8 (not NVFP4)")
print("  Formula would be: input_scale = amax / 448.0")
print()

flux_amax_if_fp8_mean = flux_input_scale_mean * F8_E4M3_MAX
wan_amax_if_fp8_mean = wan_input_scale_mean * F8_E4M3_MAX

print(f"  Flux implied amax (FP8 formula): {flux_amax_if_fp8_mean:.2f}")
print(f"  WAN implied amax (FP8 formula):  {wan_amax_if_fp8_mean:.2f}")
print(f"  Ratio: {flux_amax_if_fp8_mean / wan_amax_if_fp8_mean:.1f}x")
print()

if abs(flux_amax_if_fp8_mean / wan_amax_if_fp8_mean - ratio_mean) < 0.1:
    print("  -> This doesn't change the ratio! Still the same problem.")
else:
    print("  -> Different result, but ratio still large")
print()

# Hypothesis 2: Flux uses a constant/heuristic
print("HYPOTHESIS 2: Flux uses constant input_scale (not calibrated)")
print()

flux_input_constant = flux_input_scale_median
print(f"  If Flux used constant median value: {flux_input_constant:.6f}")
print(
    f"  Implied activation range: amax = {flux_input_constant * NVFP4_SCALE_DIVISOR:.2f}"
)
print()

wan_with_flux_constant = flux_input_constant
wan_amax_needed = wan_with_flux_constant * NVFP4_SCALE_DIVISOR
print(f"  If we use this for WAN: input_scale = {wan_with_flux_constant:.6f}")
print(f"  This assumes WAN activations have amax ~{wan_amax_needed:.2f}")
print()

# Hypothesis 3: Use a multiplier on our calibrated values
print("HYPOTHESIS 3: Scale our calibrated values by Flux/WAN ratio")
print()

scale_multiplier = ratio_mean
wan_scaled_mean = wan_input_scale_mean * scale_multiplier
wan_scaled_median = wan_input_scale_median * scale_multiplier

print(f"  Multiplier: {scale_multiplier:.1f}x")
print(f"  WAN scaled mean: {wan_scaled_mean:.6f}")
print(f"  WAN scaled median: {wan_scaled_median:.6f}")
print()
print(
    f"  This would give activation range: amax = {wan_scaled_mean * NVFP4_SCALE_DIVISOR:.2f}"
)
print()

print("=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print()

print("Based on this analysis:")
print()
print("1. TEST: Use Flux median input_scale (~0.024) as constant for ALL WAN layers")
print(f"   Create model with input_scale = {flux_input_scale_median:.6f} everywhere")
print()
print(f"2. TEST: Scale our calibrated values by {scale_multiplier:.1f}x")
print(f"   Multiply all input_scale by {scale_multiplier:.1f}")
print()
print("3. INVESTIGATE: Check our calibration - are we using realistic inputs?")
print("   - Print actual activation ranges during calibration")
print("   - Compare to what ComfyUI would actually feed the model")
print()
