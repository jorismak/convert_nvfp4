"""
Analyze working Flux NVFP4 model to understand how it loads in ComfyUI
"""

from safetensors import safe_open
import json

flux_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/flux1-dev-nvfp4.safetensors"

print("=" * 80)
print("FLUX NVFP4 MODEL ANALYSIS")
print("=" * 80)

with safe_open(flux_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    metadata = f.metadata()

    print(f"\n1. METADATA")
    print("=" * 80)
    for k, v in sorted(metadata.items()):
        if len(str(v)) < 200:
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: <{len(v)} bytes>")
            if k == "_quantization_metadata":
                try:
                    qmeta = json.loads(v)
                    print(f"    Format version: {qmeta.get('format_version')}")
                    print(f"    Layers count: {len(qmeta.get('layers', {}))}")
                    # Show first layer
                    layers = qmeta.get("layers", {})
                    if layers:
                        first_layer = list(layers.keys())[0]
                        print(f"    First layer: {first_layer}")
                        print(f"      Config: {layers[first_layer]}")
                except:
                    pass

    print(f"\n2. TENSOR ANALYSIS")
    print("=" * 80)
    print(f"Total tensors: {len(keys)}")

    # Find quantized layers
    quantized_weights = []
    weight_scales = []
    weight_scale_2s = []

    for k in keys:
        if k.endswith(".weight") and "scale" not in k:
            tensor = f.get_tensor(k)
            if tensor.dtype == torch.uint8:  # Packed NVFP4
                quantized_weights.append(k)
        elif k.endswith(".weight_scale"):
            weight_scales.append(k)
        elif k.endswith(".weight_scale_2"):
            weight_scale_2s.append(k)

    print(f"\nQuantized (uint8) weights: {len(quantized_weights)}")
    print(f".weight_scale tensors: {len(weight_scales)}")
    print(f".weight_scale_2 tensors: {len(weight_scale_2s)}")

    if quantized_weights:
        print(f"\nFirst 5 quantized layers:")
        for i, qw in enumerate(quantized_weights[:5]):
            weight = f.get_tensor(qw)
            scale_key = qw.replace(".weight", ".weight_scale")
            scale2_key = qw.replace(".weight", ".weight_scale_2")

            has_scale = scale_key in keys
            has_scale2 = scale2_key in keys

            print(f"  {i + 1}. {qw}")
            print(f"     Shape: {list(weight.shape)}, dtype: {weight.dtype}")
            print(f"     Has .weight_scale: {has_scale}")
            print(f"     Has .weight_scale_2: {has_scale2}")

            if has_scale:
                scale = f.get_tensor(scale_key)
                print(f"     Scale shape: {list(scale.shape)}, dtype: {scale.dtype}")

            if has_scale2:
                scale2 = f.get_tensor(scale2_key)
                print(
                    f"     Scale2 shape: {list(scale2.shape)}, dtype: {scale2.dtype}, value: {scale2.item()}"
                )

    # Check for any marker tensors
    print(f"\n3. POTENTIAL MARKER TENSORS")
    print("=" * 80)
    marker_candidates = [k for k in keys if len(k) < 20 and "." not in k]
    if marker_candidates:
        print(f"Found {len(marker_candidates)} short tensor names (potential markers):")
        for mk in marker_candidates:
            tensor = f.get_tensor(mk)
            print(
                f"  {mk}: shape={list(tensor.shape)}, dtype={tensor.dtype}, numel={tensor.numel()}"
            )
    else:
        print("No marker tensors found")

import torch

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
Key questions:
1. Does Flux NVFP4 have _quantization_metadata?
2. What format does the metadata use?
3. Are there any marker tensors like 'scaled_fp8'?
4. How does ComfyUI know to use mixed-ops for this model?
""")
