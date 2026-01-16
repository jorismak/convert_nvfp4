"""
Compare Flux FP8 and NVFP4 models to understand format differences
"""

from safetensors import safe_open
import torch
import json

fp8_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/flux-2-klein-9b-fp8.safetensors"
nvfp4_path = (
    "D:/ComfyUI/ComfyUI/models/diffusion_models/flux-2-klein-9b-nvfp4.safetensors"
)

print("=" * 100)
print("FLUX FP8 MODEL ANALYSIS")
print("=" * 100)

with safe_open(fp8_path, framework="pt", device="cpu") as f:
    fp8_keys = list(f.keys())
    fp8_metadata = f.metadata()

    print(f"\n1. METADATA")
    print("=" * 100)
    for k, v in sorted(fp8_metadata.items()):
        if len(str(v)) < 200:
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: <{len(v)} bytes>")
            if k == "_quantization_metadata":
                try:
                    qmeta = json.loads(v)
                    print(f"    Format version: {qmeta.get('format_version')}")
                    print(f"    Layers count: {len(qmeta.get('layers', {}))}")
                    first_layer = (
                        list(qmeta.get("layers", {}).keys())[0]
                        if qmeta.get("layers")
                        else None
                    )
                    if first_layer:
                        print(f"    First layer example: {first_layer}")
                        print(f"      Config: {qmeta['layers'][first_layer]}")
                except Exception as e:
                    print(f"    Error parsing: {e}")

    print(f"\n2. TENSOR COUNTS")
    print("=" * 100)
    print(f"Total tensors: {len(fp8_keys)}")

    # Categorize tensors
    weights = [k for k in fp8_keys if k.endswith(".weight") and "scale" not in k]
    scale_weights = [k for k in fp8_keys if k.endswith(".scale_weight")]
    weight_scales = [k for k in fp8_keys if k.endswith(".weight_scale")]
    weight_scale_2s = [k for k in fp8_keys if k.endswith(".weight_scale_2")]
    input_scales = [k for k in fp8_keys if k.endswith(".input_scale")]
    biases = [k for k in fp8_keys if k.endswith(".bias")]
    norms = [k for k in fp8_keys if ".norm" in k]
    other = [
        k
        for k in fp8_keys
        if k
        not in weights
        + scale_weights
        + weight_scales
        + weight_scale_2s
        + input_scales
        + biases
        + norms
    ]

    print(f"  .weight tensors: {len(weights)}")
    print(f"  .scale_weight tensors: {len(scale_weights)}")
    print(f"  .weight_scale tensors: {len(weight_scales)}")
    print(f"  .weight_scale_2 tensors: {len(weight_scale_2s)}")
    print(f"  .input_scale tensors: {len(input_scales)}")
    print(f"  .bias tensors: {len(biases)}")
    print(f"  norm tensors: {len(norms)}")
    print(f"  other tensors: {len(other)}")

    # Check for marker tensors
    print(f"\n3. MARKER TENSORS")
    print("=" * 100)
    marker_candidates = [k for k in fp8_keys if len(k) < 20 and "." not in k]
    if marker_candidates:
        print(f"Found {len(marker_candidates)} potential marker tensors:")
        for mk in marker_candidates:
            tensor = f.get_tensor(mk)
            print(
                f"  {mk}: shape={list(tensor.shape)}, dtype={tensor.dtype}, numel={tensor.numel()}"
            )
    else:
        print("No marker tensors found (no short names without dots)")

    # Sample a quantized layer
    print(f"\n4. SAMPLE QUANTIZED LAYER")
    print("=" * 100)
    if weights:
        sample_weight_key = weights[0]
        sample_weight = f.get_tensor(sample_weight_key)
        print(f"Layer: {sample_weight_key}")
        print(
            f"  .weight shape: {list(sample_weight.shape)}, dtype: {sample_weight.dtype}"
        )

        # Check for associated scale tensors
        base_name = sample_weight_key.replace(".weight", "")
        scale_weight_key = base_name + ".scale_weight"
        weight_scale_key = base_name + ".weight_scale"
        weight_scale_2_key = base_name + ".weight_scale_2"
        input_scale_key = base_name + ".input_scale"

        if scale_weight_key in fp8_keys:
            scale = f.get_tensor(scale_weight_key)
            print(
                f"  .scale_weight: shape={list(scale.shape)}, dtype={scale.dtype}, value={scale.item() if scale.numel() == 1 else 'array'}"
            )

        if weight_scale_key in fp8_keys:
            scale = f.get_tensor(weight_scale_key)
            print(f"  .weight_scale: shape={list(scale.shape)}, dtype={scale.dtype}")

        if weight_scale_2_key in fp8_keys:
            scale2 = f.get_tensor(weight_scale_2_key)
            print(
                f"  .weight_scale_2: shape={list(scale2.shape)}, dtype={scale2.dtype}, value={scale2.item() if scale2.numel() == 1 else 'array'}"
            )

        if input_scale_key in fp8_keys:
            iscale = f.get_tensor(input_scale_key)
            print(
                f"  .input_scale: shape={list(iscale.shape)}, dtype={iscale.dtype}, value={iscale.item() if iscale.numel() == 1 else 'array'}"
            )

        # Check if weight is FP8
        if sample_weight.dtype == torch.float8_e4m3fn:
            print(f"  Weight is FP8 E4M3FN")
            # Convert to float to get range
            weight_float = sample_weight.float()
            print(
                f"  Value range: [{weight_float.min().item():.6f}, {weight_float.max().item():.6f}]"
            )

print("\n" + "=" * 100)
print("FLUX NVFP4 MODEL ANALYSIS")
print("=" * 100)

with safe_open(nvfp4_path, framework="pt", device="cpu") as f:
    nvfp4_keys = list(f.keys())
    nvfp4_metadata = f.metadata()

    print(f"\n1. METADATA")
    print("=" * 100)
    for k, v in sorted(nvfp4_metadata.items()):
        if len(str(v)) < 200:
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: <{len(v)} bytes>")
            if k == "_quantization_metadata":
                try:
                    qmeta = json.loads(v)
                    print(f"    Format version: {qmeta.get('format_version')}")
                    print(f"    Layers count: {len(qmeta.get('layers', {}))}")
                    first_layer = (
                        list(qmeta.get("layers", {}).keys())[0]
                        if qmeta.get("layers")
                        else None
                    )
                    if first_layer:
                        print(f"    First layer example: {first_layer}")
                        print(f"      Config: {qmeta['layers'][first_layer]}")
                except Exception as e:
                    print(f"    Error parsing: {e}")

    print(f"\n2. TENSOR COUNTS")
    print("=" * 100)
    print(f"Total tensors: {len(nvfp4_keys)}")

    # Categorize tensors
    weights = [k for k in nvfp4_keys if k.endswith(".weight") and "scale" not in k]
    scale_weights = [k for k in nvfp4_keys if k.endswith(".scale_weight")]
    weight_scales = [k for k in nvfp4_keys if k.endswith(".weight_scale")]
    weight_scale_2s = [k for k in nvfp4_keys if k.endswith(".weight_scale_2")]
    input_scales = [k for k in nvfp4_keys if k.endswith(".input_scale")]
    biases = [k for k in nvfp4_keys if k.endswith(".bias")]
    norms = [k for k in nvfp4_keys if ".norm" in k]
    other = [
        k
        for k in nvfp4_keys
        if k
        not in weights
        + scale_weights
        + weight_scales
        + weight_scale_2s
        + input_scales
        + biases
        + norms
    ]

    print(f"  .weight tensors: {len(weights)}")
    print(f"  .scale_weight tensors: {len(scale_weights)}")
    print(f"  .weight_scale tensors: {len(weight_scales)}")
    print(f"  .weight_scale_2 tensors: {len(weight_scale_2s)}")
    print(f"  .input_scale tensors: {len(input_scales)}")
    print(f"  .bias tensors: {len(biases)}")
    print(f"  norm tensors: {len(norms)}")
    print(f"  other tensors: {len(other)}")

    # Check for marker tensors
    print(f"\n3. MARKER TENSORS")
    print("=" * 100)
    marker_candidates = [k for k in nvfp4_keys if len(k) < 20 and "." not in k]
    if marker_candidates:
        print(f"Found {len(marker_candidates)} potential marker tensors:")
        for mk in marker_candidates:
            tensor = f.get_tensor(mk)
            print(
                f"  {mk}: shape={list(tensor.shape)}, dtype={tensor.dtype}, numel={tensor.numel()}"
            )
    else:
        print("No marker tensors found (no short names without dots)")

    # Sample a quantized layer
    print(f"\n4. SAMPLE QUANTIZED LAYER")
    print("=" * 100)
    if weights:
        sample_weight_key = weights[0]
        sample_weight = f.get_tensor(sample_weight_key)
        print(f"Layer: {sample_weight_key}")
        print(
            f"  .weight shape: {list(sample_weight.shape)}, dtype: {sample_weight.dtype}"
        )

        # Check for associated scale tensors
        base_name = sample_weight_key.replace(".weight", "")
        scale_weight_key = base_name + ".scale_weight"
        weight_scale_key = base_name + ".weight_scale"
        weight_scale_2_key = base_name + ".weight_scale_2"
        input_scale_key = base_name + ".input_scale"

        if scale_weight_key in nvfp4_keys:
            scale = f.get_tensor(scale_weight_key)
            print(
                f"  .scale_weight: shape={list(scale.shape)}, dtype={scale.dtype}, value={scale.item() if scale.numel() == 1 else 'array'}"
            )

        if weight_scale_key in nvfp4_keys:
            scale = f.get_tensor(weight_scale_key)
            print(f"  .weight_scale: shape={list(scale.shape)}, dtype={scale.dtype}")
            scale_float = scale.float() if scale.dtype == torch.float8_e4m3fn else scale
            print(f"    First 5 values: {scale_float.flatten()[:5].tolist()}")
            print(
                f"    Range: [{scale_float.min().item():.6f}, {scale_float.max().item():.6f}]"
            )

        if weight_scale_2_key in nvfp4_keys:
            scale2 = f.get_tensor(weight_scale_2_key)
            print(
                f"  .weight_scale_2: shape={list(scale2.shape)}, dtype={scale2.dtype}, value={scale2.item() if scale2.numel() == 1 else 'array'}"
            )

        if input_scale_key in nvfp4_keys:
            iscale = f.get_tensor(input_scale_key)
            print(
                f"  .input_scale: shape={list(iscale.shape)}, dtype={iscale.dtype}, value={iscale.item() if iscale.numel() == 1 else 'array'}"
            )

        # Check if weight is uint8 (packed NVFP4)
        if sample_weight.dtype == torch.uint8:
            print(f"  Weight is UINT8 (packed NVFP4)")
            print(f"  Packed weight is half size of expected (2 FP4 per byte)")

print("\n" + "=" * 100)
print("COMPARISON")
print("=" * 100)

print("\n1. TENSOR NAMING DIFFERENCES")
with (
    safe_open(fp8_path, framework="pt", device="cpu") as f1,
    safe_open(nvfp4_path, framework="pt", device="cpu") as f2,
):
    fp8_set = set(fp8_keys)
    nvfp4_set = set(nvfp4_keys)

    # Get base layer names (without scale suffixes)
    def get_base_layers(keys):
        bases = set()
        for k in keys:
            if k.endswith(".weight") and "scale" not in k:
                bases.add(k.replace(".weight", ""))
        return bases

    fp8_layers = get_base_layers(fp8_keys)
    nvfp4_layers = get_base_layers(nvfp4_keys)

    print(f"FP8 quantized layers: {len(fp8_layers)}")
    print(f"NVFP4 quantized layers: {len(nvfp4_layers)}")

    # Sample one layer from each and show what tensors exist
    if fp8_layers and nvfp4_layers:
        fp8_sample = list(fp8_layers)[0]
        nvfp4_sample = list(nvfp4_layers)[0]

        print(f"\nFP8 sample layer: {fp8_sample}")
        for suffix in [
            ".weight",
            ".scale_weight",
            ".weight_scale",
            ".weight_scale_2",
            ".input_scale",
        ]:
            exists = (fp8_sample + suffix) in fp8_set
            print(f"  {suffix}: {'YES' if exists else 'NO'}")

        print(f"\nNVFP4 sample layer: {nvfp4_sample}")
        for suffix in [
            ".weight",
            ".scale_weight",
            ".weight_scale",
            ".weight_scale_2",
            ".input_scale",
        ]:
            exists = (nvfp4_sample + suffix) in nvfp4_set
            print(f"  {suffix}: {'YES' if exists else 'NO'}")

print("\n2. DEQUANTIZATION VALUE COMPARISON")
print("=" * 100)

# Try to dequantize and compare a layer that exists in both
with (
    safe_open(fp8_path, framework="pt", device="cpu") as f1,
    safe_open(nvfp4_path, framework="pt", device="cpu") as f2,
):
    # Find a common layer
    fp8_weights = [k for k in fp8_keys if k.endswith(".weight") and "scale" not in k]
    nvfp4_weights = [
        k for k in nvfp4_keys if k.endswith(".weight") and "scale" not in k
    ]

    common_layers = set(fp8_weights) & set(nvfp4_weights)

    if common_layers:
        sample_layer = list(common_layers)[0]
        print(f"Comparing layer: {sample_layer}")

        # Load FP8
        fp8_weight = f1.get_tensor(sample_layer)
        base_name = sample_layer.replace(".weight", "")

        fp8_dequant = None
        if fp8_weight.dtype == torch.float8_e4m3fn:
            scale_key = base_name + ".scale_weight"
            if scale_key in fp8_keys:
                fp8_scale = f1.get_tensor(scale_key)
                fp8_dequant = fp8_weight.float() * fp8_scale.item()
                print(
                    f"FP8 dequantized range: [{fp8_dequant.min().item():.6f}, {fp8_dequant.max().item():.6f}]"
                )
                print(f"FP8 scale value: {fp8_scale.item():.6f}")

        # Load NVFP4
        nvfp4_weight = f2.get_tensor(sample_layer)
        nvfp4_dequant = None

        if nvfp4_weight.dtype == torch.uint8:
            # Need to dequantize NVFP4
            weight_scale_key = base_name + ".weight_scale"
            weight_scale_2_key = base_name + ".weight_scale_2"

            if weight_scale_key in nvfp4_keys and weight_scale_2_key in nvfp4_keys:
                print(
                    f"NVFP4 has block scales and tensor scale - attempting dequantization..."
                )

                # This is a simplified dequantization - real one requires unpacking FP4
                weight_scale = f2.get_tensor(weight_scale_key)
                weight_scale_2 = f2.get_tensor(weight_scale_2_key)

                print(
                    f"NVFP4 weight_scale shape: {list(weight_scale.shape)}, dtype: {weight_scale.dtype}"
                )
                print(f"NVFP4 weight_scale_2: {weight_scale_2.item():.6f}")
                weight_scale_float = (
                    weight_scale.float()
                    if weight_scale.dtype == torch.float8_e4m3fn
                    else weight_scale
                )
                print(
                    f"NVFP4 block scale range: [{weight_scale_float.min().item():.6f}, {weight_scale_float.max().item():.6f}]"
                )

        if fp8_dequant is not None and nvfp4_dequant is not None:
            # Compare
            correlation = torch.corrcoef(
                torch.stack([fp8_dequant.flatten(), nvfp4_dequant.flatten()])
            )[0, 1]
            print(f"\nCorrelation between FP8 and NVFP4 dequantized: {correlation:.6f}")
        else:
            print(
                f"\nCannot fully compare - need proper NVFP4 dequantization implementation"
            )

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
