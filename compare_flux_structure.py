"""
Compare Flux FP8 vs NVFP4 - focus on scale values and structure only
(Avoid loading full weight tensors due to memory/dtype issues)
"""

from safetensors import safe_open
import numpy as np
import json

fp8_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/flux-2-klein-9b-fp8.safetensors"
nvfp4_path = (
    "D:/ComfyUI/ComfyUI/models/diffusion_models/flux-2-klein-9b-nvfp4.safetensors"
)

print("=" * 100)
print("FLUX FP8 vs NVFP4 STRUCTURE COMPARISON")
print("=" * 100)

# Load FP8 metadata and scale tensors
print("\n1. FP8 MODEL STRUCTURE")
print("-" * 50)

with safe_open(fp8_path, framework="np") as f:
    fp8_keys = list(f.keys())
    fp8_meta = f.metadata()
    fp8_qmeta = json.loads(fp8_meta.get("_quantization_metadata", "{}"))

    print(f"Total tensors: {len(fp8_keys)}")
    print(f"Quantized layers (from metadata): {len(fp8_qmeta.get('layers', {}))}")

    # Categorize
    weights = [k for k in fp8_keys if k.endswith(".weight") and "scale" not in k]
    weight_scales = [k for k in fp8_keys if k.endswith(".weight_scale")]
    weight_scale_2s = [k for k in fp8_keys if k.endswith(".weight_scale_2")]
    input_scales = [k for k in fp8_keys if k.endswith(".input_scale")]

    print(f"\n  .weight: {len(weights)}")
    print(f"  .weight_scale: {len(weight_scales)}")
    print(f"  .weight_scale_2: {len(weight_scale_2s)}")
    print(f"  .input_scale: {len(input_scales)}")

    # Load some scale tensors to see their values
    if weight_scales:
        test_scale_key = weight_scales[0]
        scale = f.get_tensor(test_scale_key)
        print(f"\n  Example scale ({test_scale_key}):")
        print(f"    Shape: {scale.shape}, dtype: {scale.dtype}")
        print(f"    Value: {scale}")

    if input_scales:
        test_iscale_key = input_scales[0]
        iscale = f.get_tensor(test_iscale_key)
        print(f"\n  Example input_scale ({test_iscale_key}):")
        print(f"    Shape: {iscale.shape}, dtype: {iscale.dtype}")
        print(f"    Value: {iscale}")

# Load NVFP4 metadata and scale tensors
print("\n2. NVFP4 MODEL STRUCTURE")
print("-" * 50)

with safe_open(nvfp4_path, framework="np") as f:
    nvfp4_keys = list(f.keys())
    nvfp4_meta = f.metadata()
    nvfp4_qmeta = json.loads(nvfp4_meta.get("_quantization_metadata", "{}"))

    print(f"Total tensors: {len(nvfp4_keys)}")
    print(f"Quantized layers (from metadata): {len(nvfp4_qmeta.get('layers', {}))}")

    # Categorize
    weights = [k for k in nvfp4_keys if k.endswith(".weight") and "scale" not in k]
    weight_scales = [k for k in nvfp4_keys if k.endswith(".weight_scale")]
    weight_scale_2s = [k for k in nvfp4_keys if k.endswith(".weight_scale_2")]
    input_scales = [k for k in nvfp4_keys if k.endswith(".input_scale")]

    print(f"\n  .weight: {len(weights)}")
    print(f"  .weight_scale: {len(weight_scales)}")
    print(f"  .weight_scale_2: {len(weight_scale_2s)}")
    print(f"  .input_scale: {len(input_scales)}")

    # Load scale_2 tensors (these are float32 scalars)
    if weight_scale_2s:
        test_scale2_key = weight_scale_2s[0]
        scale2 = f.get_tensor(test_scale2_key)
        print(f"\n  Example weight_scale_2 ({test_scale2_key}):")
        print(f"    Shape: {scale2.shape}, dtype: {scale2.dtype}")
        print(f"    Value: {scale2}")

    if input_scales:
        test_iscale_key = input_scales[0]
        iscale = f.get_tensor(test_iscale_key)
        print(f"\n  Example input_scale ({test_iscale_key}):")
        print(f"    Shape: {iscale.shape}, dtype: {iscale.dtype}")
        print(f"    Value: {iscale}")

# Compare metadata structure
print("\n3. METADATA COMPARISON")
print("=" * 100)

print("\nFP8 _quantization_metadata sample:")
fp8_layers = fp8_qmeta.get("layers", {})
for i, (k, v) in enumerate(list(fp8_layers.items())[:5]):
    print(f"  {k}: {v}")

print("\nNVFP4 _quantization_metadata sample:")
nvfp4_layers = nvfp4_qmeta.get("layers", {})
for i, (k, v) in enumerate(list(nvfp4_layers.items())[:5]):
    print(f"  {k}: {v}")

# Compare scale values for same layers
print("\n4. SCALE VALUE COMPARISON")
print("=" * 100)

# Find common layers
common_layers = set(fp8_layers.keys()) & set(nvfp4_layers.keys())
print(f"\nCommon quantized layers: {len(common_layers)}")

# Load and compare scales for a few layers
test_layers = list(common_layers)[:5]

with safe_open(fp8_path, framework="np") as f_fp8:
    with safe_open(nvfp4_path, framework="np") as f_nvfp4:
        for layer in test_layers:
            print(f"\n  Layer: {layer}")

            fp8_scale_key = layer + ".weight_scale"
            nvfp4_scale2_key = layer + ".weight_scale_2"

            if fp8_scale_key in fp8_keys:
                fp8_scale = f_fp8.get_tensor(fp8_scale_key)
                print(f"    FP8 weight_scale: {fp8_scale.item():.8f}")

            if nvfp4_scale2_key in nvfp4_keys:
                nvfp4_scale2 = f_nvfp4.get_tensor(nvfp4_scale2_key)
                print(f"    NVFP4 weight_scale_2: {nvfp4_scale2.item():.8f}")

                if fp8_scale_key in fp8_keys:
                    ratio = (
                        fp8_scale.item() / nvfp4_scale2.item()
                        if nvfp4_scale2.item() != 0
                        else float("inf")
                    )
                    print(f"    Ratio (FP8/NVFP4): {ratio:.2f}")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print("""
KEY STRUCTURAL DIFFERENCES:

FP8 Format:
  - .weight: float8_e4m3fn tensor (full dimensions)
  - .weight_scale: float32 scalar (per-tensor scale)
  - .input_scale: float32 scalar (runtime activation scale)
  - metadata format: {"format": "float8_e4m3fn"}

NVFP4 Format:
  - .weight: uint8 packed tensor (half dimensions, 2 FP4 per byte)
  - .weight_scale: float8_e4m3fn array (block scales, shape matches blocks)
  - .weight_scale_2: float32 scalar (per-tensor scale)
  - .input_scale: float32 scalar (runtime activation scale)
  - metadata format: {"format": "nvfp4"}

KEY INSIGHT:
- FP8 uses single per-tensor scale
- NVFP4 uses TWO-LEVEL scaling: block scales (fp8 array) + tensor scale (f32 scalar)
- Both use _quantization_metadata to signal format to ComfyUI
- Neither needs marker tensors like scaled_fp8
""")
