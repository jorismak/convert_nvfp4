"""
Check source FP32 values and manually compute what scale SHOULD be
"""

from safetensors import safe_open
import torch

# Load source
source_path = "D:/comfy2/ComfyUI/nvfp4-conv/wan2.2-ti2v-5b/diffusion_pytorch_model-00001-of-00003.safetensors"
test_layer = "blocks.0.cross_attn.k.weight"

print("=" * 80)
print(f"Analyzing {test_layer}")
print("=" * 80)

with safe_open(source_path, framework="pt", device="cpu") as f:
    weight = f.get_tensor(test_layer)

    print(f"\nSource FP32:")
    print(f"  Shape: {list(weight.shape)}")
    print(f"  Dtype: {weight.dtype}")
    print(f"  First 20 values: {weight.flatten()[:20].tolist()}")
    print(f"  Min: {weight.min().item():.6f}")
    print(f"  Max: {weight.max().item():.6f}")
    print(f"  Abs max: {weight.abs().max().item():.6f}")

    # Compute scale
    amax = weight.abs().max().item()
    FP8_MAX = 448.0
    scale = amax / FP8_MAX

    print(f"\nManual quantization:")
    print(f"  amax = {amax:.6f}")
    print(f"  FP8_MAX = {FP8_MAX}")
    print(f"  scale = amax / FP8_MAX = {scale:.6f}")

    # Quantize
    scaled = weight.float() / scale
    print(f"\nScaled values (before FP8 conversion):")
    print(f"  First 20: {scaled.flatten()[:20].tolist()}")
    print(f"  Min: {scaled.min().item():.2f}")
    print(f"  Max: {scaled.max().item():.2f}")

    fp8_weight = scaled.to(torch.float8_e4m3fn)
    print(f"\nAfter FP8 conversion:")
    print(f"  First 20: {fp8_weight.flatten()[:20].tolist()}")

# Now check KJ's model
print("\n" + "=" * 80)
print("KJ's model for comparison")
print("=" * 80)

kj_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors"
with safe_open(kj_path, framework="pt", device="cpu") as f:
    kj_weight = f.get_tensor(test_layer)
    kj_scale = f.get_tensor(test_layer.replace(".weight", ".scale_weight"))

    print(f"\nKJ FP8 weight:")
    print(f"  First 20: {kj_weight.flatten()[:20].tolist()}")
    print(f"  Scale: {kj_scale.item():.6f}")

    # Dequantize to see original
    dequant = kj_weight.float() * kj_scale.item()
    print(f"\nKJ dequantized (weight * scale):")
    print(f"  First 20: {dequant.flatten()[:20].tolist()}")

# Check our broken model
print("\n" + "=" * 80)
print("Our broken FP8 model")
print("=" * 80)

ours_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/wan2.2-ti2v-5b-fp8-simple-kj-format.safetensors"
with safe_open(ours_path, framework="pt", device="cpu") as f:
    ours_weight = f.get_tensor(test_layer)
    ours_scale = f.get_tensor(test_layer.replace(".weight", ".scale_weight"))

    print(f"\nOurs FP8 weight:")
    print(f"  First 20: {ours_weight.flatten()[:20].tolist()}")
    print(f"  Scale: {ours_scale.item():.6f}")

    # Dequantize
    dequant = ours_weight.float() * ours_scale.item()
    print(f"\nOurs dequantized (weight * scale):")
    print(f"  First 20: {dequant.flatten()[:20].tolist()}")
