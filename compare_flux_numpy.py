"""
Compare Flux FP8 vs NVFP4 using numpy to avoid PyTorch segfaults
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
    # FP8 E4M3FN: 1 sign, 4 exponent, 3 mantissa
    # Simple conversion: treat as raw representation
    arr = arr.view(np.uint8)
    sign = (arr >> 7) & 1
    exp = (arr >> 3) & 0x0F
    mantissa = arr & 0x07

    # Handle special cases
    result = np.zeros_like(arr, dtype=np.float32)

    # Normal numbers: (-1)^sign * 2^(exp-7) * (1 + mantissa/8)
    normal_mask = (exp > 0) & (exp < 15)
    result[normal_mask] = (
        ((-1.0) ** sign[normal_mask])
        * (2.0 ** (exp[normal_mask].astype(np.float32) - 7))
        * (1.0 + mantissa[normal_mask].astype(np.float32) / 8.0)
    )

    # Subnormal: (-1)^sign * 2^-6 * (mantissa/8)
    subnormal_mask = (exp == 0) & (mantissa > 0)
    result[subnormal_mask] = (
        ((-1.0) ** sign[subnormal_mask])
        * (2.0**-6)
        * (mantissa[subnormal_mask].astype(np.float32) / 8.0)
    )

    return result


print("=" * 100)
print("FLUX FP8 vs NVFP4 COMPARISON (using numpy)")
print("=" * 100)

# Load metadata and keys
print("\n1. LOADING METADATA")
print("=" * 100)

with safe_open(fp8_path, framework="np") as f:
    fp8_keys = list(f.keys())
    fp8_meta = f.metadata()
    fp8_qmeta = json.loads(fp8_meta.get("_quantization_metadata", "{}"))

with safe_open(nvfp4_path, framework="np") as f:
    nvfp4_keys = list(f.keys())
    nvfp4_meta = f.metadata()
    nvfp4_qmeta = json.loads(nvfp4_meta.get("_quantization_metadata", "{}"))

print(f"FP8 total tensors: {len(fp8_keys)}")
print(f"FP8 quantized layers: {len(fp8_qmeta.get('layers', {}))}")
print(f"NVFP4 total tensors: {len(nvfp4_keys)}")
print(f"NVFP4 quantized layers: {len(nvfp4_qmeta.get('layers', {}))}")

# Find common quantized layers
fp8_layers = set(fp8_qmeta.get("layers", {}).keys())
nvfp4_layers = set(nvfp4_qmeta.get("layers", {}).keys())
common_layers = fp8_layers & nvfp4_layers

print(f"\nCommon quantized layers: {len(common_layers)}")

# Compare tensor structure for one layer
print("\n2. TENSOR STRUCTURE COMPARISON")
print("=" * 100)

test_layer = "double_blocks.0.img_attn.proj"
print(f"\nTest layer: {test_layer}")

# FP8 tensors for this layer
fp8_layer_tensors = [k for k in fp8_keys if k.startswith(test_layer)]
nvfp4_layer_tensors = [k for k in nvfp4_keys if k.startswith(test_layer)]

print(f"\nFP8 tensors for this layer:")
for t in sorted(fp8_layer_tensors):
    print(f"  {t}")

print(f"\nNVFP4 tensors for this layer:")
for t in sorted(nvfp4_layer_tensors):
    print(f"  {t}")

# Load and compare actual tensor values
print("\n3. DEQUANTIZATION COMPARISON")
print("=" * 100)

with (
    safe_open(fp8_path, framework="np") as f_fp8,
    safe_open(nvfp4_path, framework="np") as f_nvfp4,
):
    weight_key = test_layer + ".weight"

    # FP8 weight
    fp8_weight = f_fp8.get_tensor(weight_key)
    fp8_scale = f_fp8.get_tensor(test_layer + ".weight_scale")

    print(f"\nFP8:")
    print(f"  Weight shape: {fp8_weight.shape}, dtype: {fp8_weight.dtype}")
    print(f"  Scale shape: {fp8_scale.shape}, value: {fp8_scale.item():.8f}")

    # Convert FP8 to float and dequantize
    fp8_float = fp8_to_float(fp8_weight)
    fp8_dequant = fp8_float * fp8_scale.item()

    print(f"  FP8 raw range: [{fp8_float.min():.2f}, {fp8_float.max():.2f}]")
    print(f"  Dequantized range: [{fp8_dequant.min():.6f}, {fp8_dequant.max():.6f}]")
    print(f"  Dequantized mean: {fp8_dequant.mean():.6f}, std: {fp8_dequant.std():.6f}")

    # NVFP4 weight
    nvfp4_weight = f_nvfp4.get_tensor(weight_key)
    nvfp4_block_scale = f_nvfp4.get_tensor(test_layer + ".weight_scale")
    nvfp4_tensor_scale = f_nvfp4.get_tensor(test_layer + ".weight_scale_2")

    print(f"\nNVFP4:")
    print(f"  Weight shape: {nvfp4_weight.shape}, dtype: {nvfp4_weight.dtype}")
    print(
        f"  Block scale shape: {nvfp4_block_scale.shape}, dtype: {nvfp4_block_scale.dtype}"
    )
    print(f"  Tensor scale: {nvfp4_tensor_scale.item():.8f}")

    # Convert block scales from FP8 to float
    nvfp4_block_scale_float = fp8_to_float(nvfp4_block_scale)
    print(
        f"  Block scale range: [{nvfp4_block_scale_float.min():.2f}, {nvfp4_block_scale_float.max():.2f}]"
    )

    # Dequantize NVFP4 (sample - first 64 rows, first 64 packed cols = 128 values)
    sample_rows = min(64, nvfp4_weight.shape[0])
    sample_packed_cols = min(64, nvfp4_weight.shape[1])

    packed_sample = nvfp4_weight[:sample_rows, :sample_packed_cols]

    # Unpack FP4
    low_nibbles = packed_sample & 0x0F
    high_nibbles = (packed_sample >> 4) & 0x0F

    # Interleave
    unpacked = np.zeros((sample_rows, sample_packed_cols * 2), dtype=np.int32)
    unpacked[:, 0::2] = low_nibbles
    unpacked[:, 1::2] = high_nibbles

    # Convert to FP4 values
    fp4_values = FP4_VALUES[unpacked]

    # Apply block scales (16 values per block)
    num_blocks = sample_packed_cols * 2 // 16
    block_scale_sample = nvfp4_block_scale_float[:sample_rows, :num_blocks]

    nvfp4_dequant = np.zeros_like(fp4_values)
    for block_idx in range(num_blocks):
        start = block_idx * 16
        end = start + 16
        nvfp4_dequant[:, start:end] = (
            fp4_values[:, start:end] * block_scale_sample[:, block_idx : block_idx + 1]
        )

    # Apply tensor scale
    nvfp4_dequant = nvfp4_dequant * nvfp4_tensor_scale.item()

    print(f"\n  Sample ({sample_rows} x {sample_packed_cols * 2}) dequantized:")
    print(f"    Range: [{nvfp4_dequant.min():.6f}, {nvfp4_dequant.max():.6f}]")
    print(f"    Mean: {nvfp4_dequant.mean():.6f}, std: {nvfp4_dequant.std():.6f}")

    # Compare with FP8 sample
    fp8_sample = fp8_dequant[:sample_rows, : sample_packed_cols * 2]

    print(f"\n4. CORRELATION ANALYSIS")
    print("=" * 100)
    print(f"  FP8 sample range: [{fp8_sample.min():.6f}, {fp8_sample.max():.6f}]")
    print(
        f"  NVFP4 sample range: [{nvfp4_dequant.min():.6f}, {nvfp4_dequant.max():.6f}]"
    )

    # Correlation
    fp8_flat = fp8_sample.flatten()
    nvfp4_flat = nvfp4_dequant.flatten()

    correlation = np.corrcoef(fp8_flat, nvfp4_flat)[0, 1]
    print(f"\n  Correlation: {correlation:.6f}")

    # Error metrics
    diff = np.abs(fp8_flat - nvfp4_flat)
    rel_error = diff / (np.abs(fp8_flat) + 1e-8)

    print(f"  Mean absolute error: {diff.mean():.6f}")
    print(f"  Max absolute error: {diff.max():.6f}")
    print(f"  Mean relative error: {rel_error.mean():.2%}")

    # Show first few values
    print(f"\n  First 8 values:")
    print(f"    FP8:   {fp8_sample[0, :8].tolist()}")
    print(f"    NVFP4: {nvfp4_dequant[0, :8].tolist()}")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print("""
KEY FINDINGS:
1. FP8 format: weight (fp8) + weight_scale (f32 scalar)
2. NVFP4 format: weight (uint8 packed) + weight_scale (fp8 block array) + weight_scale_2 (f32 scalar)
3. Both use _quantization_metadata with format field
4. No marker tensors needed (unlike scaled_fp8 for WAN)

If correlation is high (>0.95): Both represent same underlying weights
If correlation is low: Different source models or quantization issues
""")
