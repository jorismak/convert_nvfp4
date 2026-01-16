"""
Deep dive into shape issues in our NVFP4 quantization
"""

import gguf
import numpy as np
from safetensors import safe_open
import torch

print("=" * 80)
print("Shape and Layout Analysis")
print("=" * 80)

# Load GGUF
gguf_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/Wan2.2-TI2V-5B-Q4_0.gguf"
reader = gguf.GGUFReader(gguf_path)

# Load our NVFP4
nvfp4_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/wan2.2-ti2v-5b-nvfp4-quant-test.safetensors"

# Load source FP32 to compare - load index to find which shard has each layer
import json

index_path = "D:/comfy2/ComfyUI/nvfp4-conv/wan2.2-ti2v-5b/diffusion_pytorch_model.safetensors.index.json"
with open(index_path, "r") as f:
    index = json.load(f)
    weight_map = index["weight_map"]

shard_dir = "D:/comfy2/ComfyUI/nvfp4-conv/wan2.2-ti2v-5b/"

test_layers = [
    "blocks.0.self_attn.q.weight",
    "blocks.0.self_attn.k.weight",
    "blocks.5.ffn.0.weight",
]

for layer_name in test_layers:
    print(f"\n{'=' * 80}")
    print(f"Layer: {layer_name}")
    print("=" * 80)

    # Find in GGUF
    gguf_tensor = None
    for tensor in reader.tensors:
        if tensor.name == layer_name:
            gguf_tensor = tensor
            break

    if not gguf_tensor:
        print("NOT FOUND in GGUF")
        continue

    print(f"\n1. GGUF Q4_0:")
    print(f"   Shape: {gguf_tensor.shape}")
    print(f"   Type: {gguf_tensor.tensor_type.name}")

    # Load source FP32 - find which shard has this layer
    source_tensor = None
    if layer_name in weight_map:
        shard_file = weight_map[layer_name]
        shard_path = shard_dir + shard_file
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            source_tensor = f.get_tensor(layer_name)
            print(f"\n2. Source FP32 (from {shard_file}):")
            print(f"   Shape: {list(source_tensor.shape)}")
            print(f"   Dtype: {source_tensor.dtype}")
            print(f"   First 10 values: {source_tensor.flatten()[:10].tolist()}")
    else:
        print(f"\n2. Source FP32: NOT IN INDEX")
        source_tensor = None

    # Load our NVFP4
    with safe_open(nvfp4_path, framework="pt", device="cpu") as f:
        if layer_name not in f.keys():
            print(f"\n3. Our NVFP4: NOT FOUND")
            continue

        weight = f.get_tensor(layer_name)
        weight_scale = f.get_tensor(layer_name.replace(".weight", ".weight_scale"))
        weight_scale_2 = f.get_tensor(layer_name.replace(".weight", ".weight_scale_2"))

        print(f"\n3. Our NVFP4 (packed):")
        print(f"   weight shape: {list(weight.shape)} (packed uint8)")
        print(f"   weight_scale shape: {list(weight_scale.shape)} (FP8 block scales)")
        print(f"   weight_scale_2: {weight_scale_2.item()} (F32 tensor scale)")

        # Calculate unpacked shape
        packed_shape = list(weight.shape)
        unpacked_shape = packed_shape.copy()
        unpacked_shape[-1] = unpacked_shape[-1] * 2  # 2 FP4 per byte
        print(f"   Unpacked shape would be: {unpacked_shape}")

        # Check if this matches source or GGUF
        if source_tensor is not None:
            source_shape = list(source_tensor.shape)
            print(f"\n4. Shape Comparison:")
            print(f"   Source FP32: {source_shape}")
            print(f"   GGUF Q4_0:   {list(gguf_tensor.shape)}")
            print(f"   Our unpacked: {unpacked_shape}")

            if source_shape == list(gguf_tensor.shape):
                print(f"   [OK] Source matches GGUF")
            else:
                print(f"   [ERROR] Source DIFFERENT from GGUF!")

            if unpacked_shape == source_shape:
                print(f"   [OK] Our unpacked matches source")
            elif unpacked_shape == source_shape[::-1]:
                print(f"   [ERROR] Our unpacked is TRANSPOSED of source")
            else:
                print(f"   [ERROR] Our unpacked shape is WRONG")

        # Check weight_scale shape
        print(f"\n5. Block Scale Analysis:")
        n_elements = np.prod(unpacked_shape)
        n_blocks = (n_elements + 15) // 16  # 16 values per block
        expected_scale_elements = n_blocks
        actual_scale_elements = np.prod(weight_scale.shape)

        print(f"   Total elements (unpacked): {n_elements}")
        print(f"   Expected blocks (÷16): {n_blocks}")
        print(f"   Actual scale elements: {actual_scale_elements}")

        if actual_scale_elements == expected_scale_elements:
            print(f"   [OK] Scale count matches")
        else:
            print(
                f"   [ERROR] Scale count MISMATCH by {actual_scale_elements - expected_scale_elements}"
            )

            # Try with transposed shape
            transposed_shape = unpacked_shape[::-1]
            n_elements_t = np.prod(transposed_shape)
            n_blocks_t = (n_elements_t + 15) // 16
            if actual_scale_elements == n_blocks_t:
                print(
                    f"   [INFO] Scale count matches TRANSPOSED shape! ({transposed_shape})"
                )

print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)
print("""
If shapes don't match, this explains the correlation ~0.001:
- We're reading weights in the wrong order
- Block scales are aligned to wrong dimensions
- Dequantization produces garbage

Next step: Check convert_nvfp4.py quantization code for transpose issues.
""")
