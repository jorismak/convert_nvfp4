"""
Check what the 'scaled_fp8' tensor is in KJ's model
"""

from safetensors import safe_open

kj_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors"

print("Checking 'scaled_fp8' tensor in KJ's model...")

with safe_open(kj_path, framework="pt", device="cpu") as f:
    if "scaled_fp8" in f.keys():
        tensor = f.get_tensor("scaled_fp8")
        print(f"\nFound 'scaled_fp8':")
        print(f"  Shape: {list(tensor.shape)}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Numel: {tensor.numel()}")
        if tensor.numel() == 1:
            print(f"  Value: {tensor.item()}")
        else:
            print(f"  First 20 values: {tensor.flatten()[:20].tolist()}")
    else:
        print("\n'scaled_fp8' NOT found (shouldn't happen)")
