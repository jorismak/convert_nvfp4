"""
Check if working Flux NVFP4 models also have this transpose pattern
"""

from safetensors import safe_open
import torch

flux_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/flux1-dev-nvfp4.safetensors"

print("=" * 80)
print("Flux1-dev NVFP4 Shape Analysis")
print("=" * 80)

with safe_open(flux_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())

    # Find linear weight examples
    linear_weights = [
        k for k in keys if ".weight" in k and "scale" not in k and "norm" not in k
    ][:10]

    print(f"\nTotal tensors: {len(keys)}")
    print(f"\nSample linear weights:")

    for key in linear_weights:
        tensor = f.get_tensor(key)
        print(f"  {key}")
        print(f"    Shape: {list(tensor.shape)}, dtype: {tensor.dtype}")

        # Check if there's a scale
        if key.replace(".weight", ".weight_scale") in keys:
            scale = f.get_tensor(key.replace(".weight", ".weight_scale"))
            print(f"    Scale shape: {list(scale.shape)}")

        # Check dimensions
        if tensor.dtype == torch.uint8:  # Packed NVFP4
            unpacked_dim = list(tensor.shape)
            unpacked_dim[-1] *= 2
            print(f"    Unpacked would be: {unpacked_dim}")

print("\n" + "=" * 80)
print("WAN vs Flux Comparison")
print("=" * 80)
print("""
Key question: Do Flux models also have the same storage format as our WAN models?

If Flux works and has same format → Something else is wrong
If Flux has different format → We need to match their format
""")
