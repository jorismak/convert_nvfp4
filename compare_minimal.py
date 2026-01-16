"""
Minimal comparison - just load one layer at a time
"""

from safetensors import safe_open
import torch

fp8_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/flux-2-klein-9b-fp8.safetensors"
nvfp4_path = (
    "D:/ComfyUI/ComfyUI/models/diffusion_models/flux-2-klein-9b-nvfp4.safetensors"
)

print("Loading FP8 keys...")
with safe_open(fp8_path, framework="pt", device="cpu") as f:
    fp8_keys = list(f.keys())
print(f"FP8 has {len(fp8_keys)} tensors")

print("\nLoading NVFP4 keys...")
with safe_open(nvfp4_path, framework="pt", device="cpu") as f:
    nvfp4_keys = list(f.keys())
print(f"NVFP4 has {len(nvfp4_keys)} tensors")

# Find smallest quantized layer
print("\nFinding smallest quantized layer...")

test_layer = "single_blocks.0.linear2.weight"
base_name = test_layer.replace(".weight", "")

print(f"\nLoading FP8 layer: {test_layer}")
with safe_open(fp8_path, framework="pt", device="cpu") as f:
    fp8_weight = f.get_tensor(test_layer)
    fp8_scale = f.get_tensor(base_name + ".weight_scale")
    print(f"  Weight: {fp8_weight.shape}, {fp8_weight.dtype}")
    print(f"  Scale: {fp8_scale.item():.8f}")

    # Dequantize first row
    fp8_row = fp8_weight[0, :32].float() * fp8_scale.item()
    print(f"  First row (dequant): {fp8_row[:8].tolist()}")

print(f"\nLoading NVFP4 layer: {test_layer}")
with safe_open(nvfp4_path, framework="pt", device="cpu") as f:
    nvfp4_weight = f.get_tensor(test_layer)
    nvfp4_scale = f.get_tensor(base_name + ".weight_scale")
    nvfp4_scale2 = f.get_tensor(base_name + ".weight_scale_2")
    print(f"  Weight: {nvfp4_weight.shape}, {nvfp4_weight.dtype}")
    print(f"  Block scale: {nvfp4_scale.shape}, {nvfp4_scale.dtype}")
    print(f"  Tensor scale: {nvfp4_scale2.item():.8f}")

    # Get first row block scales
    block_scales_row0 = nvfp4_scale[0, :4].float()
    print(f"  First 4 block scales: {block_scales_row0.tolist()}")

    # FP4 lookup
    FP4_VALUES = [
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
    ]

    # Unpack first 16 bytes (32 FP4 values = first 2 blocks)
    packed = nvfp4_weight[0, :16].tolist()
    unpacked = []
    for byte in packed:
        low = byte & 0x0F
        high = (byte >> 4) & 0x0F
        unpacked.extend([FP4_VALUES[low], FP4_VALUES[high]])

    print(f"  First 8 FP4 values (raw): {unpacked[:8]}")

    # Apply scales
    # First 16 values use block_scales[0], next 16 use block_scales[1]
    tensor_scale = nvfp4_scale2.item()
    dequant = []
    for i, v in enumerate(unpacked[:32]):
        block_idx = i // 16
        bs = (
            block_scales_row0[block_idx].item()
            if block_idx < len(block_scales_row0)
            else 1.0
        )
        dequant.append(v * bs * tensor_scale)

    print(f"  First row (dequant): {dequant[:8]}")

print("\nComparison:")
print(f"  FP8 first 8:   {[f'{x:.4f}' for x in fp8_row[:8].tolist()]}")
print(f"  NVFP4 first 8: {[f'{x:.4f}' for x in dequant[:8]]}")
