"""
Mini FP8 Converter - Test format and layer naming

This is a minimal FP8 converter to test if our format/layer-naming approach
is correct. If this also produces noisy output, the problem is in our approach.
If it works, the problem is specific to NVFP4.
"""

import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

F8_E4M3_MAX = 448.0


def quantize_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight to FP8 E4M3 with per-tensor scaling."""
    # Compute scale
    amax = torch.amax(weight.abs())
    scale = amax / F8_E4M3_MAX
    scale = torch.clamp(scale, min=1e-12).to(torch.float32)

    # Quantize
    scaled = weight / scale
    scaled = torch.clamp(scaled, -F8_E4M3_MAX, F8_E4M3_MAX)
    fp8_weight = scaled.to(torch.float8_e4m3fn)

    return fp8_weight, scale


def main():
    source_dir = Path(r"D:\comfy2\ComfyUI\nvfp4-conv\wan2.2-ti2v-5b")
    output_path = Path(
        r"D:\ComfyUI\ComfyUI\models\diffusion_models\wan2.2-ti2v-5b-fp8-test.safetensors"
    )

    # Load index
    index_path = source_dir / "diffusion_pytorch_model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    tensor_names = list(weight_map.keys())

    print(f"Found {len(tensor_names)} tensors")

    # Load all tensors
    print("Loading tensors...")
    tensors = {}
    shard_files = set(weight_map.values())

    for shard_file in tqdm(shard_files, desc="Loading shards"):
        shard_path = source_dir / shard_file
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensors[name] = f.get_tensor(name)

    # Identify Linear layers (2D weights)
    linear_layers = set()
    for name in tensor_names:
        if name.endswith(".weight"):
            t = tensors[name]
            if t.dim() == 2:
                layer = name[:-7]  # Remove .weight
                linear_layers.add(layer)

    print(f"Found {len(linear_layers)} Linear layers")

    # Convert to FP8
    output_tensors = {}
    quantized_layer_configs = {}

    print("Quantizing to FP8...")
    for layer in tqdm(sorted(linear_layers), desc="Quantizing"):
        weight_name = f"{layer}.weight"
        bias_name = f"{layer}.bias"

        weight = tensors[weight_name]
        weight_cuda = weight.to(device="cuda", dtype=torch.float32)

        # Quantize
        fp8_weight, scale = quantize_fp8(weight_cuda)

        # Store
        output_tensors[weight_name] = fp8_weight.cpu()
        output_tensors[f"{layer}.weight_scale"] = scale.cpu()

        # Track for metadata
        quantized_layer_configs[layer] = {"format": "float8_e4m3fn"}

        # Copy bias if exists
        if bias_name in tensors:
            output_tensors[bias_name] = tensors[bias_name]

    # Copy non-quantized tensors
    print("Copying non-quantized tensors...")
    for name in tqdm(tensor_names, desc="Copying"):
        if name not in output_tensors:
            tensor = tensors[name]
            # Convert FP32 to BF16
            if tensor.dtype == torch.float32:
                tensor = tensor.to(torch.bfloat16)
            output_tensors[name] = tensor

    # Build metadata
    output_metadata = {
        "_quantization_metadata": json.dumps(
            {
                "format_version": "1.0",
                "layers": quantized_layer_configs,
            }
        ),
        "fp8_converter": "convert_fp8_mini.py",
        "fp8_quantized_layers": str(len(quantized_layer_configs)),
    }

    # Save
    print(f"Saving to {output_path}...")
    save_file(output_tensors, str(output_path), metadata=output_metadata)

    # Stats
    total_size = sum(t.numel() * t.element_size() for t in output_tensors.values())
    print(f"\nDone!")
    print(f"Quantized {len(quantized_layer_configs)} layers to FP8")
    print(f"Output size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
