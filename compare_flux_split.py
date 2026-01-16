"""
Compare Flux FP8 vs NVFP4 - load one at a time
"""

from safetensors import safe_open
import numpy as np
import json

fp8_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/flux-2-klein-9b-fp8.safetensors"
nvfp4_path = (
    "D:/ComfyUI/ComfyUI/models/diffusion_models/flux-2-klein-9b-nvfp4.safetensors"
)

# FP4 E2M1 lookup table
FP4_VALUES = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


def fp8_to_float(arr):
    """Convert FP8 E4M3 bytes to float32"""
    arr = arr.view(np.uint8)
    sign = (arr >> 7) & 1
    exp = (arr >> 3) & 0x0F
    mantissa = arr & 0x07

    result = np.zeros_like(arr, dtype=np.float32)

    # Normal numbers
    normal_mask = (exp > 0) & (exp < 15)
    result[normal_mask] = (
        ((-1.0) ** sign[normal_mask])
        * (2.0 ** (exp[normal_mask].astype(np.float32) - 7))
        * (1.0 + mantissa[normal_mask].astype(np.float32) / 8.0)
    )

    # Subnormal
    subnormal_mask = (exp == 0) & (mantissa > 0)
    result[subnormal_mask] = (
        ((-1.0) ** sign[subnormal_mask])
        * (2.0**-6)
        * (mantissa[subnormal_mask].astype(np.float32) / 8.0)
    )

    return result


test_layer = "double_blocks.0.img_attn.proj"

print("=" * 100)
print(f"LAYER: {test_layer}")
print("=" * 100)

# Load FP8 first
print("\n1. FP8 MODEL")
print("-" * 50)

with safe_open(fp8_path, framework="np") as f:
    fp8_weight = f.get_tensor(test_layer + ".weight")
    fp8_scale = f.get_tensor(test_layer + ".weight_scale")

    print(f"Weight shape: {fp8_weight.shape}, dtype: {fp8_weight.dtype}")
    print(f"Scale: {fp8_scale.item():.8f}")

    # Sample first 32 rows, 64 cols
    sample_rows = 32
    sample_cols = 64

    fp8_sample = fp8_weight[:sample_rows, :sample_cols]
    fp8_float = fp8_to_float(fp8_sample)
    fp8_dequant = fp8_float * fp8_scale.item()

    print(f"FP8 raw sample range: [{fp8_float.min():.2f}, {fp8_float.max():.2f}]")
    print(
        f"FP8 dequant sample range: [{fp8_dequant.min():.6f}, {fp8_dequant.max():.6f}]"
    )
    print(f"FP8 dequant mean: {fp8_dequant.mean():.6f}, std: {fp8_dequant.std():.6f}")

    # Save for comparison
    fp8_first_row = fp8_dequant[0, :16].copy()

del fp8_weight, fp8_sample, fp8_float
print("FP8 data unloaded")

# Load NVFP4 next
print("\n2. NVFP4 MODEL")
print("-" * 50)

with safe_open(nvfp4_path, framework="np") as f:
    nvfp4_weight = f.get_tensor(test_layer + ".weight")
    nvfp4_block_scale = f.get_tensor(test_layer + ".weight_scale")
    nvfp4_tensor_scale = f.get_tensor(test_layer + ".weight_scale_2")

    print(f"Weight shape: {nvfp4_weight.shape}, dtype: {nvfp4_weight.dtype}")
    print(
        f"Block scale shape: {nvfp4_block_scale.shape}, dtype: {nvfp4_block_scale.dtype}"
    )
    print(f"Tensor scale: {nvfp4_tensor_scale.item():.8f}")

    # Sample - note NVFP4 weight is half size due to packing
    # 64 cols in FP8 = 32 packed bytes in NVFP4
    nvfp4_sample = nvfp4_weight[:sample_rows, : sample_cols // 2]

    # Unpack FP4
    low = nvfp4_sample & 0x0F
    high = (nvfp4_sample >> 4) & 0x0F

    unpacked = np.zeros((sample_rows, sample_cols), dtype=np.int32)
    unpacked[:, 0::2] = low
    unpacked[:, 1::2] = high

    fp4_values = FP4_VALUES[unpacked]

    # Get block scales for this sample
    # 64 values = 4 blocks of 16
    num_blocks = sample_cols // 16
    block_scale_sample = nvfp4_block_scale[:sample_rows, :num_blocks]
    block_scale_float = fp8_to_float(block_scale_sample)

    print(f"Block scales for sample: {block_scale_float[0].tolist()}")

    # Apply block scales
    nvfp4_dequant = np.zeros_like(fp4_values)
    for b in range(num_blocks):
        start = b * 16
        end = start + 16
        nvfp4_dequant[:, start:end] = (
            fp4_values[:, start:end] * block_scale_float[:, b : b + 1]
        )

    # Apply tensor scale
    nvfp4_dequant = nvfp4_dequant * nvfp4_tensor_scale.item()

    print(
        f"NVFP4 dequant sample range: [{nvfp4_dequant.min():.6f}, {nvfp4_dequant.max():.6f}]"
    )
    print(
        f"NVFP4 dequant mean: {nvfp4_dequant.mean():.6f}, std: {nvfp4_dequant.std():.6f}"
    )

    nvfp4_first_row = nvfp4_dequant[0, :16].copy()

print("\n3. COMPARISON")
print("=" * 100)

print(f"\nFirst row, first 16 values:")
print(f"  FP8:   {[f'{x:.5f}' for x in fp8_first_row]}")
print(f"  NVFP4: {[f'{x:.5f}' for x in nvfp4_first_row]}")

# Correlation of first row
corr = np.corrcoef(fp8_first_row, nvfp4_first_row)[0, 1]
print(f"\nCorrelation (first row): {corr:.6f}")

# Full sample correlation
fp8_flat = fp8_dequant.flatten()
nvfp4_flat = nvfp4_dequant.flatten()

corr_full = np.corrcoef(fp8_flat, nvfp4_flat)[0, 1]
print(f"Correlation (full sample): {corr_full:.6f}")

# Error metrics
diff = np.abs(fp8_flat - nvfp4_flat)
print(f"Mean absolute error: {diff.mean():.6f}")
print(f"Max absolute error: {diff.max():.6f}")

# Scale comparison
print(f"\nScale comparison:")
print(f"  FP8 scale: {fp8_scale.item():.8f}")
print(f"  NVFP4 tensor scale: {nvfp4_tensor_scale.item():.8f}")
print(f"  Ratio (FP8/NVFP4): {fp8_scale.item() / nvfp4_tensor_scale.item():.2f}")

print("\n" + "=" * 100)
print("CONCLUSION")
print("=" * 100)
if corr_full > 0.95:
    print("HIGH CORRELATION - Both quantizations preserve same underlying weights")
elif corr_full > 0.7:
    print("MODERATE CORRELATION - Some differences in quantization")
else:
    print(
        "LOW CORRELATION - Significant differences, may be different source models or quantization issues"
    )
