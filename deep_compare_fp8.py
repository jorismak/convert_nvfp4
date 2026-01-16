"""
Deep comparison: KJ's working FP8 vs Our broken FP8
Find THE difference that causes ComfyUI to reject our scales
"""

from safetensors import safe_open
import torch
import json

print("=" * 80)
print("DEEP DIVE: KJ (Working) vs Ours (Broken)")
print("=" * 80)

kj_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors"
ours_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/wan2.2-ti2v-5b-fp8-simple-kj-format.safetensors"

print("\n1. METADATA COMPARISON")
print("=" * 80)

with safe_open(kj_path, framework="pt", device="cpu") as fkj:
    kj_meta = fkj.metadata()
    with safe_open(ours_path, framework="pt", device="cpu") as fours:
        ours_meta = fours.metadata()

        print("\nKJ metadata:")
        for k, v in sorted(kj_meta.items()):
            print(f"  {k}: {repr(v)[:100]}")

        print("\nOurs metadata:")
        for k, v in sorted(ours_meta.items()):
            print(f"  {k}: {repr(v)[:100]}")

        print("\nMetadata differences:")
        all_keys = set(kj_meta.keys()) | set(ours_meta.keys())
        for key in sorted(all_keys):
            kj_val = kj_meta.get(key, "<MISSING>")
            ours_val = ours_meta.get(key, "<MISSING>")
            if kj_val != ours_val:
                print(f"  {key}:")
                print(f"    KJ:   {repr(kj_val)[:80]}")
                print(f"    Ours: {repr(ours_val)[:80]}")

print("\n2. TENSOR LIST COMPARISON")
print("=" * 80)

with safe_open(kj_path, framework="pt", device="cpu") as fkj:
    kj_keys = list(fkj.keys())
    with safe_open(ours_path, framework="pt", device="cpu") as fours:
        ours_keys = list(fours.keys())

        print(f"\nKJ has {len(kj_keys)} tensors")
        print(f"Ours has {len(ours_keys)} tensors")

        kj_set = set(kj_keys)
        ours_set = set(ours_keys)

        missing_in_ours = kj_set - ours_set
        extra_in_ours = ours_set - kj_set

        if missing_in_ours:
            print(f"\nTensors in KJ but NOT in ours ({len(missing_in_ours)}):")
            for k in sorted(list(missing_in_ours)[:10]):
                print(f"  - {k}")
            if len(missing_in_ours) > 10:
                print(f"  ... and {len(missing_in_ours) - 10} more")

        if extra_in_ours:
            print(f"\nTensors in ours but NOT in KJ ({len(extra_in_ours)}):")
            for k in sorted(list(extra_in_ours)[:10]):
                print(f"  - {k}")
            if len(extra_in_ours) > 10:
                print(f"  ... and {len(extra_in_ours) - 10} more")

print("\n3. FIRST QUANTIZED LAYER COMPARISON")
print("=" * 80)

test_layer = "blocks.0.cross_attn.k.weight"
test_scale = "blocks.0.cross_attn.k.scale_weight"

with safe_open(kj_path, framework="pt", device="cpu") as fkj:
    kj_weight = fkj.get_tensor(test_layer)
    kj_scale = fkj.get_tensor(test_scale)

    print(f"\nKJ - {test_layer}:")
    print(f"  Shape: {list(kj_weight.shape)}")
    print(f"  Dtype: {kj_weight.dtype}")
    print(f"  First 20 bytes: {kj_weight.flatten()[:20].tolist()}")
    print(f"  Storage offset: {kj_weight.storage_offset()}")
    print(f"  Is contiguous: {kj_weight.is_contiguous()}")

    print(f"\nKJ - {test_scale}:")
    print(f"  Shape: {list(kj_scale.shape)}")
    print(f"  Dtype: {kj_scale.dtype}")
    print(f"  Value: {kj_scale.item()}")
    print(f"  Storage offset: {kj_scale.storage_offset()}")

with safe_open(ours_path, framework="pt", device="cpu") as fours:
    if test_layer in fours.keys() and test_scale in fours.keys():
        ours_weight = fours.get_tensor(test_layer)
        ours_scale = fours.get_tensor(test_scale)

        print(f"\nOurs - {test_layer}:")
        print(f"  Shape: {list(ours_weight.shape)}")
        print(f"  Dtype: {ours_weight.dtype}")
        print(f"  First 20 bytes: {ours_weight.flatten()[:20].tolist()}")
        print(f"  Storage offset: {ours_weight.storage_offset()}")
        print(f"  Is contiguous: {ours_weight.is_contiguous()}")

        print(f"\nOurs - {test_scale}:")
        print(f"  Shape: {list(ours_scale.shape)}")
        print(f"  Dtype: {ours_scale.dtype}")
        print(f"  Value: {ours_scale.item()}")
        print(f"  Storage offset: {ours_scale.storage_offset()}")

        print(f"\n--- COMPARISON ---")
        print(
            f"Weight shapes match: {list(kj_weight.shape) == list(ours_weight.shape)}"
        )
        print(f"Weight dtypes match: {kj_weight.dtype == ours_weight.dtype}")
        print(f"Scale shapes match: {list(kj_scale.shape) == list(ours_scale.shape)}")
        print(f"Scale dtypes match: {kj_scale.dtype == ours_scale.dtype}")
        print(f"Scale values match: {abs(kj_scale.item() - ours_scale.item()) < 1e-6}")
    else:
        print(f"\nOurs: {test_layer} or {test_scale} NOT FOUND")

print("\n4. TENSOR ORDERING")
print("=" * 80)

print("\nFirst 20 tensor names in KJ:")
for i, k in enumerate(kj_keys[:20]):
    print(f"  {i:2d}. {k}")

print("\nFirst 20 tensor names in ours:")
for i, k in enumerate(ours_keys[:20]):
    print(f"  {i:2d}. {k}")

print("\n5. FILE SIZE COMPARISON")
print("=" * 80)

import os

kj_size = os.path.getsize(kj_path)
ours_size = os.path.getsize(ours_path)

print(f"\nKJ file size: {kj_size:,} bytes ({kj_size / 1024**3:.2f} GB)")
print(f"Ours file size: {ours_size:,} bytes ({ours_size / 1024**3:.2f} GB)")
print(
    f"Difference: {abs(kj_size - ours_size):,} bytes ({abs(kj_size - ours_size) / 1024**2:.2f} MB)"
)
