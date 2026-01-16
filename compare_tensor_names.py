"""
Compare exact tensor names and ordering between KJ and ours
"""

from safetensors import safe_open

kj_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors"
ours_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/wan2.2-ti2v-5b-fp8-simple-kj-format.safetensors"

print("=" * 80)
print("Loading both models...")
print("=" * 80)

with safe_open(kj_path, framework="pt", device="cpu") as fkj:
    kj_keys = list(fkj.keys())

with safe_open(ours_path, framework="pt", device="cpu") as fours:
    ours_keys = list(fours.keys())

print(f"\nKJ: {len(kj_keys)} tensors")
print(f"Ours: {len(ours_keys)} tensors")

# Find the extra tensor in KJ
kj_set = set(kj_keys)
ours_set = set(ours_keys)

print(f"\n{'=' * 80}")
print("TENSOR IN KJ BUT NOT IN OURS:")
print("=" * 80)
missing = kj_set - ours_set
for k in sorted(missing):
    print(f"  {k}")

print(f"\n{'=' * 80}")
print("FIRST 30 TENSOR NAMES - SIDE BY SIDE")
print("=" * 80)

print(f"\n{'KJ':<60} | {'OURS':<60}")
print("-" * 125)

for i in range(min(30, len(kj_keys), len(ours_keys))):
    kj_name = kj_keys[i]
    ours_name = ours_keys[i]
    marker = "" if kj_name == ours_name else " <-- DIFFERENT"
    print(f"{kj_name:<60} | {ours_name:<60}{marker}")

# Check if any quantized layer in KJ is missing its scale in ours
print(f"\n{'=' * 80}")
print("CHECKING SCALE PAIRING")
print("=" * 80)

kj_weights = [k for k in kj_keys if k.endswith(".weight") and "scale" not in k]
kj_scales = [k for k in kj_keys if k.endswith(".scale_weight")]

print(f"\nKJ has {len(kj_weights)} .weight tensors (non-scale)")
print(f"KJ has {len(kj_scales)} .scale_weight tensors")

ours_weights = [k for k in ours_keys if k.endswith(".weight") and "scale" not in k]
ours_scales = [k for k in ours_keys if k.endswith(".scale_weight")]

print(f"\nOurs has {len(ours_weights)} .weight tensors (non-scale)")
print(f"Ours has {len(ours_scales)} .scale_weight tensors")

# Check if KJ has any weights that are FP8
print(f"\n{'=' * 80}")
print("CHECKING WHICH LAYERS ARE QUANTIZED")
print("=" * 80)

with safe_open(kj_path, framework="pt", device="cpu") as fkj:
    kj_fp8_weights = []
    for k in kj_keys:
        if k.endswith(".weight") and "scale" not in k:
            tensor = fkj.get_tensor(k)
            if "float8" in str(tensor.dtype):
                kj_fp8_weights.append(k)

    print(f"\nKJ has {len(kj_fp8_weights)} FP8 quantized weights")
    print(f"First 5:")
    for k in kj_fp8_weights[:5]:
        scale_key = k.replace(".weight", ".scale_weight")
        has_scale = scale_key in kj_keys
        print(f"  {k} -> scale exists: {has_scale}")

with safe_open(ours_path, framework="pt", device="cpu") as fours:
    ours_fp8_weights = []
    for k in ours_keys:
        if k.endswith(".weight") and "scale" not in k:
            tensor = fours.get_tensor(k)
            if "float8" in str(tensor.dtype):
                ours_fp8_weights.append(k)

    print(f"\nOurs has {len(ours_fp8_weights)} FP8 quantized weights")
    print(f"First 5:")
    for k in ours_fp8_weights[:5]:
        scale_key = k.replace(".weight", ".scale_weight")
        has_scale = scale_key in ours_keys
        print(f"  {k} -> scale exists: {has_scale}")

# Most important: Are scale names IDENTICAL?
print(f"\n{'=' * 80}")
print("SCALE NAME COMPARISON")
print("=" * 80)

kj_scales_set = set(kj_scales)
ours_scales_set = set(ours_scales)

if kj_scales_set == ours_scales_set:
    print("\nSCALE NAMES ARE IDENTICAL!")
else:
    print("\nSCALE NAMES ARE DIFFERENT!")
    print(f"\nIn KJ but not ours: {kj_scales_set - ours_scales_set}")
    print(f"\nIn ours but not KJ: {ours_scales_set - kj_scales_set}")
