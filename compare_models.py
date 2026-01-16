"""
Compare metadata and structure of three models:
1. GGUF Q4_0 (working)
2. KJ's FP8 (working)
3. Our NVFP4 (broken)
"""

import gguf
from safetensors import safe_open
import json

print("=" * 80)
print("1. GGUF Q4_0 Model Analysis")
print("=" * 80)

gguf_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/Wan2.2-TI2V-5B-Q4_0.gguf"
reader = gguf.GGUFReader(gguf_path)

print(f"\nGGUF Metadata:")
for field in reader.fields.values():
    if not field.name.startswith("general"):
        continue
    print(f"  {field.name}: {field.parts[field.data[0]]}")

print(f"\nGGUF Tensors: {len(reader.tensors)} total")
tensor_types = {}
for tensor in reader.tensors:
    ttype = tensor.tensor_type.name
    tensor_types[ttype] = tensor_types.get(ttype, 0) + 1

for ttype, count in sorted(tensor_types.items()):
    print(f"  {ttype}: {count}")

# Sample tensor names
print(f"\nSample tensor names (first 10):")
for i, tensor in enumerate(reader.tensors[:10]):
    print(f"  {tensor.name}: {tensor.tensor_type.name}, shape={tensor.shape}")

print("\n" + "=" * 80)
print("2. KJ's FP8 Model Analysis (WORKING)")
print("=" * 80)

kj_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors"
with safe_open(kj_path, framework="pt", device="cpu") as f:
    kj_keys = list(f.keys())
    kj_meta = f.metadata()

    print(f"\nKJ Metadata keys:")
    for key in sorted(kj_meta.keys()):
        val = kj_meta[key]
        if len(str(val)) < 200:
            print(f"  {key}: {val}")
        else:
            print(f"  {key}: <{len(val)} bytes>")

    print(f"\nKJ Tensors: {len(kj_keys)} total")

    # Group by dtype
    dtypes = {}
    scales = []
    for key in kj_keys:
        tensor = f.get_tensor(key)
        dtype = str(tensor.dtype)
        if "scale" in key:
            scales.append(key)
        dtypes[dtype] = dtypes.get(dtype, 0) + 1

    for dtype, count in sorted(dtypes.items()):
        print(f"  {dtype}: {count}")

    print(f"\nScale tensors: {len(scales)}")
    print(f"Sample scales (first 5):")
    for s in scales[:5]:
        tensor = f.get_tensor(s)
        print(
            f"  {s}: shape={tensor.shape}, dtype={tensor.dtype}, value={tensor.item() if tensor.numel() == 1 else 'multi'}"
        )

    # Sample tensor names
    print(f"\nSample tensor names (first 10):")
    for key in kj_keys[:10]:
        tensor = f.get_tensor(key)
        print(f"  {key}: {tensor.dtype}, shape={list(tensor.shape)}")

print("\n" + "=" * 80)
print("3. Our NVFP4 Model Analysis (BROKEN)")
print("=" * 80)

nvfp4_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/wan2.2-ti2v-5b-nvfp4-quant-test-fixed.safetensors"
with safe_open(nvfp4_path, framework="pt", device="cpu") as f:
    nvfp4_keys = list(f.keys())
    nvfp4_meta = f.metadata()

    print(f"\nOur Metadata keys:")
    for key in sorted(nvfp4_meta.keys()):
        val = nvfp4_meta[key]
        if len(str(val)) < 200:
            print(f"  {key}: {val}")
        else:
            print(f"  {key}: <{len(val)} bytes>")

    print(f"\nOur Tensors: {len(nvfp4_keys)} total")

    # Group by dtype
    dtypes = {}
    weights = []
    scales = []
    scale2s = []
    for key in nvfp4_keys:
        tensor = f.get_tensor(key)
        dtype = str(tensor.dtype)
        if "weight_scale_2" in key:
            scale2s.append(key)
        elif "weight_scale" in key:
            scales.append(key)
        elif ".weight" in key:
            weights.append(key)
        dtypes[dtype] = dtypes.get(dtype, 0) + 1

    for dtype, count in sorted(dtypes.items()):
        print(f"  {dtype}: {count}")

    print(
        f"\nQuantized weights: {len([k for k in weights if any(s.replace('weight_scale', 'weight') == k for s in scales)])}"
    )
    print(f"Block scales (.weight_scale): {len(scales)}")
    print(f"Tensor scales (.weight_scale_2): {len(scale2s)}")

    # Sample scale values
    print(f"\nSample scales (first quantized layer):")
    sample_weight = [k for k in weights if "blocks.0.self_attn.q.weight" in k][0]
    sample_scale = sample_weight.replace(".weight", ".weight_scale")
    sample_scale2 = sample_weight.replace(".weight", ".weight_scale_2")

    if sample_scale in nvfp4_keys:
        scale_tensor = f.get_tensor(sample_scale)
        print(
            f"  {sample_scale}: shape={list(scale_tensor.shape)}, dtype={scale_tensor.dtype}"
        )
        # Convert to float32 for stats
        scale_float = scale_tensor.float()
        print(
            f"    min={scale_float.min().item():.6f}, max={scale_float.max().item():.6f}, mean={scale_float.mean().item():.6f}"
        )

    if sample_scale2 in nvfp4_keys:
        scale2_tensor = f.get_tensor(sample_scale2)
        print(
            f"  {sample_scale2}: shape={list(scale2_tensor.shape)}, dtype={scale2_tensor.dtype}, value={scale2_tensor.item()}"
        )

    # Sample tensor names
    print(f"\nSample tensor names (first 10):")
    for key in nvfp4_keys[:10]:
        tensor = f.get_tensor(key)
        print(f"  {key}: {tensor.dtype}, shape={list(tensor.shape)}")

print("\n" + "=" * 80)
print("4. Key Differences")
print("=" * 80)

print("\nMetadata comparison:")
print(f"  GGUF has metadata: {len(reader.fields)} fields")
print(f"  KJ has metadata: {len(kj_meta)} keys")
print(f"  Ours has metadata: {len(nvfp4_meta)} keys")

print("\nKJ metadata keys not in ours:")
for key in kj_meta.keys():
    if key not in nvfp4_meta:
        print(f"  - {key}")

print("\nOur metadata keys not in KJ:")
for key in nvfp4_meta.keys():
    if key not in kj_meta:
        print(f"  - {key}")
