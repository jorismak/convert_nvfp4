"""
NVFP4 Converter using comfy_kitchen's quantization directly.

This uses the exact same quantization kernels that ComfyUI uses at runtime,
ensuring perfect compatibility.
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Import comfy_kitchen
sys.path.insert(0, r"D:\comfy2\ComfyUI")
import comfy_kitchen as ck

# Constants
F8_E4M3_MAX = 448.0
F4_E2M1_MAX = 6.0


def quantize_layer_nvfp4_ck(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a layer to NVFP4 using comfy_kitchen's quantization.

    Returns: (packed_weight, block_scales, tensor_scale)
    """
    # Convert to BF16 (required by comfy_kitchen)
    weight_bf16 = weight.to(torch.bfloat16).cuda()

    # Compute per-tensor scale
    amax = weight_bf16.abs().amax()
    per_tensor_scale = (amax / (F8_E4M3_MAX * F4_E2M1_MAX)).to(torch.float32)

    # Quantize using comfy_kitchen
    packed, block_scales = ck.quantize_nvfp4(
        weight_bf16, per_tensor_scale, pad_16x=True
    )

    return packed.cpu(), block_scales.cpu(), per_tensor_scale.cpu()


def convert_model(
    source_path: str,
    output_path: str,
):
    """Convert FP16/BF16 model to NVFP4 using comfy_kitchen quantization."""

    print(f"Loading model from {source_path}...")

    # Load all tensors
    tensors = {}
    with safe_open(source_path, framework="pt", device="cpu") as f:
        for key in tqdm(f.keys(), desc="Loading tensors"):
            tensors[key] = f.get_tensor(key)

    # Identify linear layers (2D weights)
    linear_layers = []
    for name, tensor in tensors.items():
        if name.endswith(".weight") and tensor.dim() == 2:
            layer_name = name[:-7]  # Remove ".weight"
            linear_layers.append(layer_name)

    print(f"\nFound {len(linear_layers)} linear layers")

    # Quantize layers
    output_tensors = {}
    quantized_configs = {}

    print("\nQuantizing layers...")
    for layer_name in tqdm(linear_layers, desc="Quantizing"):
        weight_key = f"{layer_name}.weight"
        bias_key = f"{layer_name}.bias"

        weight = tensors[weight_key]

        # Quantize
        packed, block_scales, tensor_scale = quantize_layer_nvfp4_ck(weight)

        # Store
        output_tensors[weight_key] = packed
        output_tensors[f"{weight_key}_scale"] = block_scales
        output_tensors[f"{weight_key}_scale_2"] = tensor_scale

        # Track config
        quantized_configs[layer_name] = {"format": "nvfp4"}

        # Copy bias if exists
        if bias_key in tensors:
            output_tensors[bias_key] = tensors[bias_key]

    # Copy non-quantized tensors
    print("\nCopying non-quantized tensors...")
    for name, tensor in tqdm(tensors.items(), desc="Copying"):
        if name not in output_tensors:
            # Convert FP32 to BF16 for non-quantized weights
            if tensor.dtype == torch.float32:
                tensor = tensor.to(torch.bfloat16)
            output_tensors[name] = tensor

    # Build metadata
    metadata = {
        "_quantization_metadata": json.dumps(
            {
                "format_version": "1.0",
                "layers": quantized_configs,
            }
        ),
        "converter": "convert_nvfp4_ck.py (using comfy_kitchen quantization)",
        "quantized_layers": str(len(quantized_configs)),
    }

    # Save
    print(f"\nSaving to {output_path}...")
    save_file(output_tensors, output_path, metadata=metadata)

    # Stats
    total_size = sum(t.numel() * t.element_size() for t in output_tensors.values())
    print(f"\nDone!")
    print(f"Quantized {len(quantized_configs)} layers to NVFP4")
    print(f"Output size: {total_size / 1e9:.2f} GB")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert model to NVFP4 using comfy_kitchen"
    )
    parser.add_argument("source", help="Source model path (FP16/BF16)")
    parser.add_argument("output", help="Output model path")

    args = parser.parse_args()

    convert_model(args.source, args.output)


if __name__ == "__main__":
    main()
