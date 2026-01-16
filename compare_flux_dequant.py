"""
Deep comparison of Flux FP8 vs NVFP4 - dequantize and compare actual values
"""

from safetensors import safe_open
import torch
import json
import numpy as np

fp8_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/flux-2-klein-9b-fp8.safetensors"
nvfp4_path = (
    "D:/ComfyUI/ComfyUI/models/diffusion_models/flux-2-klein-9b-nvfp4.safetensors"
)

# FP4 E2M1 lookup table (values 0-15)
FP4_E2M1_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,  # Positive: 0-7
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,  # Negative: 8-15
]


def unpack_fp4(packed_uint8):
    """Unpack uint8 to two FP4 values (low nibble first, then high nibble)"""
    low_nibble = packed_uint8 & 0x0F
    high_nibble = (packed_uint8 >> 4) & 0x0F
    return low_nibble, high_nibble


def dequantize_nvfp4_block(packed_weight, block_scales, tensor_scale, block_size=16):
    """
    Dequantize NVFP4 weights.

    packed_weight: uint8 tensor, shape [out_features, in_features/2]
    block_scales: fp8 tensor, shape [out_features, num_blocks] where num_blocks = in_features/block_size
    tensor_scale: float32 scalar
    """
    out_features, packed_in = packed_weight.shape
    in_features = packed_in * 2  # 2 FP4 values per byte

    # Convert to numpy for easier manipulation
    packed_np = packed_weight.numpy()
    block_scales_np = block_scales.float().numpy()

    # Output tensor
    output = np.zeros((out_features, in_features), dtype=np.float32)

    # Unpack and dequantize
    for i in range(out_features):
        for j in range(packed_in):
            low, high = unpack_fp4(packed_np[i, j])

            # Position in unpacked tensor
            pos_low = j * 2
            pos_high = j * 2 + 1

            # Block index
            block_idx_low = pos_low // block_size
            block_idx_high = pos_high // block_size

            # Get block scales
            bs_low = (
                block_scales_np[i, block_idx_low]
                if block_idx_low < block_scales_np.shape[1]
                else 1.0
            )
            bs_high = (
                block_scales_np[i, block_idx_high]
                if block_idx_high < block_scales_np.shape[1]
                else 1.0
            )

            # Dequantize: fp4_value * block_scale * tensor_scale
            output[i, pos_low] = FP4_E2M1_VALUES[low] * bs_low * tensor_scale
            output[i, pos_high] = FP4_E2M1_VALUES[high] * bs_high * tensor_scale

    return torch.from_numpy(output)


def dequantize_fp8(weight, scale):
    """Dequantize FP8 weight: weight * scale"""
    return weight.float() * scale


print("=" * 100)
print("DEQUANTIZATION COMPARISON: FLUX FP8 vs NVFP4")
print("=" * 100)

with (
    safe_open(fp8_path, framework="pt", device="cpu") as f_fp8,
    safe_open(nvfp4_path, framework="pt", device="cpu") as f_nvfp4,
):
    fp8_keys = list(f_fp8.keys())
    nvfp4_keys = list(f_nvfp4.keys())

    # Find common quantized layers
    fp8_weights = [k for k in fp8_keys if k.endswith(".weight") and "scale" not in k]
    nvfp4_weights = [
        k for k in nvfp4_keys if k.endswith(".weight") and "scale" not in k
    ]

    common_layers = sorted(set(fp8_weights) & set(nvfp4_weights))
    print(f"Common weight layers: {len(common_layers)}")

    # Compare a few layers
    layers_to_compare = common_layers[:5]

    for layer_key in layers_to_compare:
        print(f"\n{'=' * 100}")
        print(f"Layer: {layer_key}")
        print("=" * 100)

        base_name = layer_key.replace(".weight", "")

        # Load FP8 version
        fp8_weight = f_fp8.get_tensor(layer_key)
        fp8_scale_key = base_name + ".weight_scale"

        if fp8_scale_key not in fp8_keys:
            print(f"  FP8: Not quantized (no weight_scale)")
            continue

        fp8_scale = f_fp8.get_tensor(fp8_scale_key)

        print(f"  FP8 weight: shape={list(fp8_weight.shape)}, dtype={fp8_weight.dtype}")
        print(f"  FP8 scale: {fp8_scale.item():.8f}")

        # Dequantize FP8
        fp8_dequant = dequantize_fp8(fp8_weight, fp8_scale.item())
        print(
            f"  FP8 dequantized range: [{fp8_dequant.min().item():.6f}, {fp8_dequant.max().item():.6f}]"
        )
        print(
            f"  FP8 dequantized mean: {fp8_dequant.mean().item():.6f}, std: {fp8_dequant.std().item():.6f}"
        )

        # Load NVFP4 version
        nvfp4_weight = f_nvfp4.get_tensor(layer_key)
        nvfp4_scale_key = base_name + ".weight_scale"
        nvfp4_scale2_key = base_name + ".weight_scale_2"

        if nvfp4_scale_key not in nvfp4_keys or nvfp4_scale2_key not in nvfp4_keys:
            print(f"  NVFP4: Not quantized (missing scales)")
            continue

        nvfp4_block_scale = f_nvfp4.get_tensor(nvfp4_scale_key)
        nvfp4_tensor_scale = f_nvfp4.get_tensor(nvfp4_scale2_key)

        print(
            f"  NVFP4 weight: shape={list(nvfp4_weight.shape)}, dtype={nvfp4_weight.dtype}"
        )
        print(
            f"  NVFP4 block_scale: shape={list(nvfp4_block_scale.shape)}, dtype={nvfp4_block_scale.dtype}"
        )
        print(f"  NVFP4 tensor_scale: {nvfp4_tensor_scale.item():.8f}")

        # Only dequantize small layers to avoid memory issues
        if nvfp4_weight.numel() > 10_000_000:
            print(
                f"  Skipping full dequantization (too large: {nvfp4_weight.numel()} elements)"
            )

            # Sample a small portion
            sample_rows = 32
            sample_packed_cols = 128

            nvfp4_sample = nvfp4_weight[:sample_rows, :sample_packed_cols]
            block_scale_sample = nvfp4_block_scale[
                :sample_rows, : sample_packed_cols // 8
            ]  # 16 values per block, 2 per byte = 8 bytes per block

            nvfp4_dequant_sample = dequantize_nvfp4_block(
                nvfp4_sample, block_scale_sample, nvfp4_tensor_scale.item()
            )

            fp8_sample = fp8_dequant[:sample_rows, : sample_packed_cols * 2]

            print(f"  Sample comparison ({sample_rows}x{sample_packed_cols * 2}):")
            print(
                f"    FP8 sample range: [{fp8_sample.min().item():.6f}, {fp8_sample.max().item():.6f}]"
            )
            print(
                f"    NVFP4 sample range: [{nvfp4_dequant_sample.min().item():.6f}, {nvfp4_dequant_sample.max().item():.6f}]"
            )

            # Correlation
            corr = torch.corrcoef(
                torch.stack([fp8_sample.flatten(), nvfp4_dequant_sample.flatten()])
            )[0, 1]
            print(f"    Correlation: {corr.item():.6f}")

            # Relative error
            diff = (fp8_sample - nvfp4_dequant_sample).abs()
            rel_error = diff / (fp8_sample.abs() + 1e-8)
            print(f"    Mean absolute error: {diff.mean().item():.6f}")
            print(f"    Mean relative error: {rel_error.mean().item():.4%}")
        else:
            # Full dequantization for smaller layers
            nvfp4_dequant = dequantize_nvfp4_block(
                nvfp4_weight, nvfp4_block_scale, nvfp4_tensor_scale.item()
            )

            print(
                f"  NVFP4 dequantized range: [{nvfp4_dequant.min().item():.6f}, {nvfp4_dequant.max().item():.6f}]"
            )
            print(
                f"  NVFP4 dequantized mean: {nvfp4_dequant.mean().item():.6f}, std: {nvfp4_dequant.std().item():.6f}"
            )

            # Correlation
            corr = torch.corrcoef(
                torch.stack([fp8_dequant.flatten(), nvfp4_dequant.flatten()])
            )[0, 1]
            print(f"  Correlation: {corr.item():.6f}")

            # Relative error
            diff = (fp8_dequant - nvfp4_dequant).abs()
            rel_error = diff / (fp8_dequant.abs() + 1e-8)
            print(f"  Mean absolute error: {diff.mean().item():.6f}")
            print(f"  Mean relative error: {rel_error.mean().item():.4%}")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print("""
If correlation is high (>0.95) and relative error is low (<10%):
  -> Both quantization methods preserve the same underlying weights
  -> The format/algorithm is correct

If correlation is low or error is high:
  -> Something is different between how they were quantized
  -> Could be source model difference, or quantization algorithm issue
""")
