"""
Compare working Flux NVFP4 model with our broken WAN NVFP4 model
"""

from safetensors import safe_open
import json

print("=" * 80)
print("WORKING: flux1-dev-nvfp4.safetensors")
print("=" * 80)

flux_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/flux1-dev-nvfp4.safetensors"
with safe_open(flux_path, framework="pt", device="cpu") as f:
    flux_keys = list(f.keys())
    flux_meta = f.metadata()

    print(f"\nMetadata keys:")
    for key in sorted(flux_meta.keys()):
        val = flux_meta[key]
        if len(str(val)) < 500:
            print(f"  {key}: {val}")
        else:
            print(f"  {key}: <{len(val)} bytes>")

    print(f"\nTotal tensors: {len(flux_keys)}")

    # Check first quantized layer structure
    sample_keys = [k for k in flux_keys if "double_blocks.0.img_attn.qkv.weight" in k]
    print(f"\nSample quantized layer (double_blocks.0.img_attn.qkv):")
    for key in sorted(sample_keys):
        tensor = f.get_tensor(key)
        print(f"  {key}")
        print(f"    shape: {list(tensor.shape)}, dtype: {tensor.dtype}")
        if tensor.numel() == 1:
            print(f"    value: {tensor.item()}")

print("\n" + "=" * 80)
print("BROKEN: wan2.2-ti2v-5b-nvfp4-quant-test-fixed.safetensors")
print("=" * 80)

wan_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/wan2.2-ti2v-5b-nvfp4-quant-test-fixed.safetensors"
with safe_open(wan_path, framework="pt", device="cpu") as f:
    wan_keys = list(f.keys())
    wan_meta = f.metadata()

    print(f"\nMetadata keys:")
    for key in sorted(wan_meta.keys()):
        val = wan_meta[key]
        if len(str(val)) < 500:
            print(f"  {key}: {val}")
        else:
            print(f"  {key}: <{len(val)} bytes>")

    print(f"\nTotal tensors: {len(wan_keys)}")

    # Check first quantized layer structure
    sample_keys = [k for k in wan_keys if "blocks.0.self_attn.q.weight" in k]
    print(f"\nSample quantized layer (blocks.0.self_attn.q):")
    for key in sorted(sample_keys):
        tensor = f.get_tensor(key)
        print(f"  {key}")
        print(f"    shape: {list(tensor.shape)}, dtype: {tensor.dtype}")
        if tensor.numel() == 1:
            print(f"    value: {tensor.item()}")

print("\n" + "=" * 80)
print("KEY DIFFERENCES")
print("=" * 80)

print("\nMetadata comparison:")
flux_meta_keys = set(flux_meta.keys())
wan_meta_keys = set(wan_meta.keys())

print(f"\nFlux has but WAN doesn't:")
for key in sorted(flux_meta_keys - wan_meta_keys):
    print(f"  - {key}")

print(f"\nWAN has but Flux doesn't:")
for key in sorted(wan_meta_keys - flux_meta_keys):
    print(f"  - {key}")
