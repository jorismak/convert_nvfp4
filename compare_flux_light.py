"""
Light comparison of Flux FP8 vs NVFP4 - sample small portions only
"""

from safetensors import safe_open
import torch
import json

fp8_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/flux-2-klein-9b-fp8.safetensors"
nvfp4_path = (
    "D:/ComfyUI/ComfyUI/models/diffusion_models/flux-2-klein-9b-nvfp4.safetensors"
)

# FP4 E2M1 lookup table (values 0-15)
FP4_E2M1_VALUES = torch.tensor(
    [
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
)


def dequant_nvfp4_sample(
    packed_weight, block_scales, tensor_scale, num_rows=64, num_packed_cols=64
):
    """Dequantize a small sample of NVFP4 weights"""
    # Sample dimensions
    packed_sample = packed_weight[:num_rows, :num_packed_cols].clone()

    # Block size is 16, so for num_packed_cols bytes = num_packed_cols*2 values
    # Number of blocks per row = (num_packed_cols * 2) / 16 = num_packed_cols / 8
    num_blocks_per_row = num_packed_cols // 8
    block_scales_sample = block_scales[:num_rows, :num_blocks_per_row].float()

    # Unpack
    low_nibbles = packed_sample & 0x0F
    high_nibbles = (packed_sample >> 4) & 0x0F

    # Interleave: for each byte position j, low goes to 2j, high goes to 2j+1
    unpacked = torch.zeros(num_rows, num_packed_cols * 2, dtype=torch.long)
    unpacked[:, 0::2] = low_nibbles
    unpacked[:, 1::2] = high_nibbles

    # Convert to FP4 values
    fp4_values = FP4_E2M1_VALUES[unpacked]

    # Apply block scales (16 values per block)
    output = torch.zeros_like(fp4_values)
    for block_idx in range(num_blocks_per_row):
        start = block_idx * 16
        end = start + 16
        if end <= fp4_values.shape[1]:
            output[:, start:end] = (
                fp4_values[:, start:end]
                * block_scales_sample[:, block_idx : block_idx + 1]
            )

    # Apply tensor scale
    output = output * tensor_scale

    return output


print("=" * 100)
print("LIGHT DEQUANTIZATION COMPARISON: FLUX FP8 vs NVFP4")
print("=" * 100)

with (
    safe_open(fp8_path, framework="pt", device="cpu") as f_fp8,
    safe_open(nvfp4_path, framework="pt", device="cpu") as f_nvfp4,
):
    fp8_keys = list(f_fp8.keys())
    nvfp4_keys = list(f_nvfp4.keys())

    # Find a quantized layer that exists in both
    # Pick a smaller layer for testing
    test_layers = [
        "single_blocks.0.linear1.weight",
        "single_blocks.0.linear2.weight",
        "double_blocks.0.img_attn.proj.weight",
    ]

    for layer_key in test_layers:
        if layer_key not in fp8_keys or layer_key not in nvfp4_keys:
            print(f"Skipping {layer_key} - not in both models")
            continue

        print(f"\n{'=' * 100}")
        print(f"Layer: {layer_key}")
        print("=" * 100)

        base_name = layer_key.replace(".weight", "")

        # Check if both are quantized
        fp8_scale_key = base_name + ".weight_scale"
        nvfp4_scale_key = base_name + ".weight_scale"
        nvfp4_scale2_key = base_name + ".weight_scale_2"

        if fp8_scale_key not in fp8_keys:
            print(f"  FP8: Not quantized")
            continue
        if nvfp4_scale_key not in nvfp4_keys or nvfp4_scale2_key not in nvfp4_keys:
            print(f"  NVFP4: Not quantized")
            continue

        # Load FP8
        fp8_weight = f_fp8.get_tensor(layer_key)
        fp8_scale = f_fp8.get_tensor(fp8_scale_key)

        print(
            f"  FP8 weight shape: {list(fp8_weight.shape)}, dtype: {fp8_weight.dtype}"
        )
        print(f"  FP8 scale: {fp8_scale.item():.8f}")

        # Load NVFP4
        nvfp4_weight = f_nvfp4.get_tensor(layer_key)
        nvfp4_block_scale = f_nvfp4.get_tensor(nvfp4_scale_key)
        nvfp4_tensor_scale = f_nvfp4.get_tensor(nvfp4_scale2_key)

        print(
            f"  NVFP4 weight shape: {list(nvfp4_weight.shape)}, dtype: {nvfp4_weight.dtype}"
        )
        print(
            f"  NVFP4 block_scale shape: {list(nvfp4_block_scale.shape)}, dtype: {nvfp4_block_scale.dtype}"
        )
        print(f"  NVFP4 tensor_scale: {nvfp4_tensor_scale.item():.8f}")

        # Sample comparison
        num_rows = min(64, fp8_weight.shape[0])
        num_packed_cols = min(64, nvfp4_weight.shape[1])
        num_cols = num_packed_cols * 2  # unpacked

        print(f"\n  Comparing sample: {num_rows} x {num_cols}")

        # FP8 dequant (sample)
        fp8_sample = fp8_weight[:num_rows, :num_cols].float() * fp8_scale.item()
        print(
            f"  FP8 sample range: [{fp8_sample.min().item():.6f}, {fp8_sample.max().item():.6f}]"
        )
        print(
            f"  FP8 sample mean: {fp8_sample.mean().item():.6f}, std: {fp8_sample.std().item():.6f}"
        )

        # NVFP4 dequant (sample)
        nvfp4_sample = dequant_nvfp4_sample(
            nvfp4_weight,
            nvfp4_block_scale,
            nvfp4_tensor_scale.item(),
            num_rows=num_rows,
            num_packed_cols=num_packed_cols,
        )
        print(
            f"  NVFP4 sample range: [{nvfp4_sample.min().item():.6f}, {nvfp4_sample.max().item():.6f}]"
        )
        print(
            f"  NVFP4 sample mean: {nvfp4_sample.mean().item():.6f}, std: {nvfp4_sample.std().item():.6f}"
        )

        # Correlation
        corr = torch.corrcoef(
            torch.stack([fp8_sample.flatten(), nvfp4_sample.flatten()])
        )[0, 1]
        print(f"\n  Correlation: {corr.item():.6f}")

        # Error metrics
        diff = (fp8_sample - nvfp4_sample).abs()
        rel_error = diff / (fp8_sample.abs() + 1e-8)
        print(f"  Mean absolute error: {diff.mean().item():.6f}")
        print(f"  Max absolute error: {diff.max().item():.6f}")
        print(f"  Mean relative error: {rel_error.mean().item():.2%}")

        # Show some actual values
        print(f"\n  First 5 values comparison:")
        print(f"    FP8:   {fp8_sample[0, :5].tolist()}")
        print(f"    NVFP4: {nvfp4_sample[0, :5].tolist()}")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
