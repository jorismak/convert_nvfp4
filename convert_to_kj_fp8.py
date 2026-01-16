"""
Convert WAN model to simple FP8 format matching KJ's working model:
- Weights: float8_e4m3fn (unpacked, full dimensions)
- Scale: single F32 scalar per layer (`.scale_weight`)
- Everything else: F32
"""

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

print("Loading source model...")
source_path = "D:/comfy2/ComfyUI/nvfp4-conv/wan2.2-ti2v-5b/diffusion_pytorch_model.safetensors.index.json"

import json

with open(source_path) as f:
    index = json.load(f)

# Load all tensors
state_dict = {}
shard_files = set(index["weight_map"].values())
shard_dir = "D:/comfy2/ComfyUI/nvfp4-conv/wan2.2-ti2v-5b"

for shard_file in sorted(shard_files):
    shard_path = f"{shard_dir}/{shard_file}"
    print(f"Loading {shard_file}...")
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

print(f"Loaded {len(state_dict)} tensors")

# Decide which layers to quantize (same as GGUF Q4_0)
skip_patterns = [
    "head.head",
    "text_embedding",
    "time_embedding",
    "time_projection",
    ".norm",
    ".modulation",
    ".bias",
    "patch_embedding",
]


def should_quantize(key):
    if ".weight" not in key:
        return False
    for pattern in skip_patterns:
        if pattern in key:
            return False
    # Only quantize 2D weights (Linear layers)
    return len(state_dict[key].shape) == 2


# Quantize
output_dict = {}
quantized_count = 0

print("\nQuantizing...")
for key in tqdm(sorted(state_dict.keys())):
    tensor = state_dict[key].float()  # Convert to F32 first

    if should_quantize(key):
        # Quantize to FP8 with simple scaling (like KJ's model)
        # KJ uses: scale = amax (NOT amax/448)
        # Then stores: weight / scale (values in [-1, +1] range)
        # Dequant: fp8_value * scale
        amax = tensor.abs().max().item()
        scale = amax if amax > 0 else 1.0  # Just use amax directly

        # Quantize: divide by scale to get values in [-1, +1]
        quantized = (tensor / scale).to(torch.float8_e4m3fn)

        # Store
        output_dict[key] = quantized
        output_dict[key.replace(".weight", ".scale_weight")] = torch.tensor(
            [scale], dtype=torch.float32
        )
        quantized_count += 1
    else:
        # Keep as F32
        output_dict[key] = tensor

print(f"\nQuantized {quantized_count} layers")
print(f"Total tensors in output: {len(output_dict)}")

# Add the 'scaled_fp8' marker tensor (dummy FP8 tensor that signals to ComfyUI this is FP8 scaled)
output_dict["scaled_fp8"] = torch.tensor([0.0, 0.0], dtype=torch.float8_e4m3fn)
print("Added 'scaled_fp8' marker tensor")

# Add metadata matching KJ's model
metadata = {"format": "pt", "model_type": "TI2V-5B"}

# Save
output_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/wan2.2-ti2v-5b-fp8-simple-kj-format.safetensors"
print(f"\nSaving to {output_path}...")
save_file(output_dict, output_path, metadata=metadata)

print("Done!")
print(
    f"\nOutput size: {sum(t.numel() * t.element_size() for t in output_dict.values()) / 1024**3:.2f} GB"
)
