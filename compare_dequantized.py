"""
Compare dequantized values from GGUF Q4_0 vs our NVFP4
"""

import gguf
import numpy as np
from safetensors import safe_open
import torch

print("=" * 80)
print("1. DEQUANTIZE GGUF Q4_0 and compare with our NVFP4")
print("=" * 80)

# Load GGUF
gguf_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/Wan2.2-TI2V-5B-Q4_0.gguf"
reader = gguf.GGUFReader(gguf_path)

# Load our NVFP4
nvfp4_path = "D:/ComfyUI/ComfyUI/models/diffusion_models/wan2.2-ti2v-5b-nvfp4-quant-test.safetensors"

# Test a few quantized layers
test_layers = [
    "blocks.0.self_attn.q.weight",
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

    if gguf_tensor is None:
        print(f"NOT FOUND in GGUF")
        continue

    print(f"GGUF tensor type: {gguf_tensor.tensor_type.name}")
    print(f"GGUF shape: {gguf_tensor.shape}")

    # Dequantize GGUF using gguf library's method
    gguf_weights_deq = gguf_tensor.data.astype(
        np.float32
    )  # This should auto-dequantize

    # If not, do manual dequantization
    if gguf_tensor.tensor_type.name == "Q4_0":
        # Actually load and dequantize properly
        import struct

        raw_data = np.frombuffer(gguf_tensor.data, dtype=np.uint8)

        # Q4_0 format: blocks of 32 values
        # Each block: 2 bytes scale (FP16) + 16 bytes (32 nibbles)
        block_size = 32
        n_elements = gguf_tensor.n_elements
        n_blocks = (n_elements + block_size - 1) // block_size

        gguf_weights_deq = []

        offset = 0
        for block_idx in range(n_blocks):
            # Read scale (FP16, 2 bytes)
            scale_bytes = bytes(raw_data[offset : offset + 2])
            scale = struct.unpack("<e", scale_bytes)[0]  # half-float
            offset += 2

            # Read 16 bytes (32 4-bit values)
            quant_bytes = raw_data[offset : offset + 16]
            offset += 16

            for byte_val in quant_bytes:
                # Lower nibble
                nibble = int(byte_val) & 0x0F
                signed_val = nibble - 8  # Q4_0 is signed
                gguf_weights_deq.append(signed_val * scale)

                # Upper nibble
                nibble = (int(byte_val) >> 4) & 0x0F
                signed_val = nibble - 8
                gguf_weights_deq.append(signed_val * scale)

        gguf_weights_deq = np.array(gguf_weights_deq[:n_elements], dtype=np.float32)
        gguf_weights_deq = gguf_weights_deq.reshape(gguf_tensor.shape)

    # Load our NVFP4
    with safe_open(nvfp4_path, framework="pt", device="cpu") as f:
        if layer_name not in f.keys():
            print(f"NOT FOUND in our NVFP4")
            continue

        weight = f.get_tensor(layer_name)
        weight_scale = f.get_tensor(layer_name.replace(".weight", ".weight_scale"))
        weight_scale_2 = f.get_tensor(layer_name.replace(".weight", ".weight_scale_2"))

        print(f"\nOur NVFP4:")
        print(f"  weight shape: {list(weight.shape)}, dtype: {weight.dtype}")
        print(
            f"  weight_scale shape: {list(weight_scale.shape)}, dtype: {weight_scale.dtype}"
        )
        print(f"  weight_scale_2: {weight_scale_2.item()}")

        # Dequantize our NVFP4
        packed = weight.numpy()
        block_scale = weight_scale.float().numpy()
        tensor_scale = weight_scale_2.item()

        # FP4 lookup table (E2M1 format)
        fp4_to_float = np.array(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=np.float32,
        )

        shape = gguf_tensor.shape
        n_elements = np.prod(shape)

        dequantized = []
        flat_packed = packed.flatten()
        flat_scales = block_scale.flatten()

        for i, byte_val in enumerate(flat_packed):
            # Lower nibble (first FP4 value)
            idx = i * 2
            if idx < n_elements:
                fp4_val = fp4_to_float[byte_val & 0x0F]
                block_idx = idx // 16
                scale = flat_scales[block_idx] * tensor_scale
                dequantized.append(fp4_val * scale)

            # Upper nibble (second FP4 value)
            idx = i * 2 + 1
            if idx < n_elements:
                fp4_val = fp4_to_float[byte_val >> 4]
                block_idx = idx // 16
                scale = flat_scales[block_idx] * tensor_scale
                dequantized.append(fp4_val * scale)

        our_weights_deq = np.array(dequantized, dtype=np.float32).reshape(shape)

    # Compare
    print(f"\n{'=' * 40}")
    print("COMPARISON")
    print("=" * 40)
    print(f"GGUF dequantized stats:")
    print(f"  shape: {gguf_weights_deq.shape}")
    print(f"  min: {gguf_weights_deq.min():.6f}, max: {gguf_weights_deq.max():.6f}")
    print(f"  mean: {gguf_weights_deq.mean():.6f}, std: {gguf_weights_deq.std():.6f}")

    print(f"\nOur NVFP4 dequantized stats:")
    print(f"  shape: {our_weights_deq.shape}")
    print(f"  min: {our_weights_deq.min():.6f}, max: {our_weights_deq.max():.6f}")
    print(f"  mean: {our_weights_deq.mean():.6f}, std: {our_weights_deq.std():.6f}")

    # Error metrics
    abs_diff = np.abs(gguf_weights_deq - our_weights_deq)
    rel_error = abs_diff / (np.abs(gguf_weights_deq) + 1e-8)

    print(f"\nError metrics:")
    print(f"  Max absolute difference: {abs_diff.max():.6f}")
    print(f"  Mean absolute difference: {abs_diff.mean():.6f}")
    print(f"  Max relative error: {rel_error.max():.6f}")
    print(f"  Mean relative error: {rel_error.mean():.6f}")

    # Correlation
    corr = np.corrcoef(gguf_weights_deq.flatten(), our_weights_deq.flatten())[0, 1]
    print(f"  Pearson correlation: {corr:.8f}")

    # Sample values
    print(f"\nFirst 20 values comparison:")
    flat_gguf = gguf_weights_deq.flatten()
    flat_ours = our_weights_deq.flatten()
    for i in range(min(20, len(flat_gguf))):
        diff = flat_gguf[i] - flat_ours[i]
        print(
            f"  [{i:2d}] GGUF: {flat_gguf[i]:10.6f}  Ours: {flat_ours[i]:10.6f}  Diff: {diff:10.6f}"
        )
