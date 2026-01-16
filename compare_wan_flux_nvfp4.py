"""
Compare our broken WAN NVFP4 model with working Flux NVFP4 model
"""

from safetensors import safe_open
import json

# Working reference
flux_nvfp4_path = (
    "D:/ComfyUI/ComfyUI/models/diffusion_models/flux-2-klein-9b-nvfp4.safetensors"
)

# Our broken model (find the latest one)
import os

wan_models_dir = "D:/ComfyUI/ComfyUI/models/diffusion_models/"
wan_nvfp4_candidates = [
    f
    for f in os.listdir(wan_models_dir)
    if "wan" in f.lower() and "nvfp4" in f.lower() and f.endswith(".safetensors")
]
print(f"Found WAN NVFP4 candidates: {wan_nvfp4_candidates}")

# Use the most recent one or specify
wan_nvfp4_path = wan_models_dir + "wan2.2-ti2v-5b-nvfp4-quant-test.safetensors"
if not os.path.exists(wan_nvfp4_path):
    # Try another
    for c in wan_nvfp4_candidates:
        wan_nvfp4_path = wan_models_dir + c
        if os.path.exists(wan_nvfp4_path):
            break

print(f"\nUsing WAN model: {wan_nvfp4_path}")
print(f"File exists: {os.path.exists(wan_nvfp4_path)}")

print("\n" + "=" * 100)
print("COMPARISON: Working Flux NVFP4 vs Our WAN NVFP4")
print("=" * 100)

# Analyze Flux NVFP4 (working)
print("\n1. FLUX NVFP4 (WORKING)")
print("-" * 50)

with safe_open(flux_nvfp4_path, framework="np") as f:
    flux_keys = list(f.keys())
    flux_meta = f.metadata()

    print(f"Total tensors: {len(flux_keys)}")
    print(f"\nMetadata keys: {list(flux_meta.keys())}")

    # Quantization metadata
    flux_qmeta = json.loads(flux_meta.get("_quantization_metadata", "{}"))
    print(f"\n_quantization_metadata:")
    print(f"  format_version: {flux_qmeta.get('format_version')}")
    print(f"  layers count: {len(flux_qmeta.get('layers', {}))}")

    # Show format for first few layers
    flux_layers = flux_qmeta.get("layers", {})
    print(f"\n  First 3 layer configs:")
    for i, (k, v) in enumerate(list(flux_layers.items())[:3]):
        print(f"    {k}: {v}")

    # Tensor categories
    weights = [k for k in flux_keys if k.endswith(".weight") and "scale" not in k]
    weight_scales = [k for k in flux_keys if k.endswith(".weight_scale")]
    weight_scale_2s = [k for k in flux_keys if k.endswith(".weight_scale_2")]
    input_scales = [k for k in flux_keys if k.endswith(".input_scale")]

    print(f"\n  Tensor counts:")
    print(f"    .weight: {len(weights)}")
    print(f"    .weight_scale: {len(weight_scales)}")
    print(f"    .weight_scale_2: {len(weight_scale_2s)}")
    print(f"    .input_scale: {len(input_scales)}")

    # Sample a quantized layer
    if weight_scale_2s:
        sample_layer = weight_scale_2s[0].replace(".weight_scale_2", "")
        print(f"\n  Sample quantized layer: {sample_layer}")

        scale2 = f.get_tensor(sample_layer + ".weight_scale_2")
        print(f"    weight_scale_2: {scale2.item():.8f}")

        if sample_layer + ".input_scale" in flux_keys:
            iscale = f.get_tensor(sample_layer + ".input_scale")
            print(f"    input_scale: {iscale.item():.8f}")

# Analyze our WAN NVFP4 (broken)
print("\n\n2. OUR WAN NVFP4 (BROKEN)")
print("-" * 50)

if os.path.exists(wan_nvfp4_path):
    with safe_open(wan_nvfp4_path, framework="np") as f:
        wan_keys = list(f.keys())
        wan_meta = f.metadata()

        print(f"Total tensors: {len(wan_keys)}")
        print(f"\nMetadata keys: {list(wan_meta.keys())}")

        # Quantization metadata
        wan_qmeta_raw = wan_meta.get("_quantization_metadata", "{}")
        print(f"\n_quantization_metadata raw length: {len(wan_qmeta_raw)} bytes")

        wan_qmeta = json.loads(wan_qmeta_raw)
        print(f"\n_quantization_metadata:")
        print(f"  format_version: {wan_qmeta.get('format_version')}")
        print(f"  layers count: {len(wan_qmeta.get('layers', {}))}")

        # Show format for first few layers
        wan_layers = wan_qmeta.get("layers", {})
        print(f"\n  First 3 layer configs:")
        for i, (k, v) in enumerate(list(wan_layers.items())[:3]):
            print(f"    {k}: {v}")

        # Tensor categories
        weights = [k for k in wan_keys if k.endswith(".weight") and "scale" not in k]
        weight_scales = [k for k in wan_keys if k.endswith(".weight_scale")]
        weight_scale_2s = [k for k in wan_keys if k.endswith(".weight_scale_2")]
        input_scales = [k for k in wan_keys if k.endswith(".input_scale")]

        print(f"\n  Tensor counts:")
        print(f"    .weight: {len(weights)}")
        print(f"    .weight_scale: {len(weight_scales)}")
        print(f"    .weight_scale_2: {len(weight_scale_2s)}")
        print(f"    .input_scale: {len(input_scales)}")

        # Sample a quantized layer
        if weight_scale_2s:
            sample_layer = weight_scale_2s[0].replace(".weight_scale_2", "")
            print(f"\n  Sample quantized layer: {sample_layer}")

            scale2 = f.get_tensor(sample_layer + ".weight_scale_2")
            print(f"    weight_scale_2: {scale2.item():.8f}")

            if sample_layer + ".input_scale" in wan_keys:
                iscale = f.get_tensor(sample_layer + ".input_scale")
                print(f"    input_scale: {iscale.item():.8f}")
            else:
                print(f"    input_scale: NOT PRESENT")

        # Check for any differences in format
        print(f"\n3. KEY DIFFERENCES")
        print("-" * 50)

        # Check layer format values
        flux_formats = set(v.get("format") for v in flux_layers.values())
        wan_formats = set(v.get("format") for v in wan_layers.values())

        print(f"\nFlux layer formats: {flux_formats}")
        print(f"WAN layer formats: {wan_formats}")

        # Check if input_scale is present for all quantized layers
        wan_quantized = set(wan_layers.keys())
        wan_has_input_scale = set(k.replace(".input_scale", "") for k in input_scales)

        missing_input_scale = wan_quantized - wan_has_input_scale
        print(f"\nWAN layers missing input_scale: {len(missing_input_scale)}")
        if missing_input_scale:
            print(f"  Examples: {list(missing_input_scale)[:5]}")

else:
    print(f"ERROR: WAN model not found at {wan_nvfp4_path}")

print("\n" + "=" * 100)
print("ANALYSIS")
print("=" * 100)
