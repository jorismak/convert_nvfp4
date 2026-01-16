"""
Proper calibration script - actually runs the model to measure real activation ranges.

This loads the FP16 model, hooks linear layers, and runs forward passes
with synthetic inputs to measure real activation statistics.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from collections import defaultdict
import gc

# FP8 E4M3 max value
FP8_MAX = 448.0


class ActivationCollector:
    """Collects input activation statistics for linear layers."""

    def __init__(self):
        self.amax_values = defaultdict(list)
        self.hooks = []

    def make_hook(self, name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            if x is not None and isinstance(x, torch.Tensor):
                amax = x.abs().amax().item()
                self.amax_values[name].append(amax)

        return hook

    def register_hooks(self, model, layer_names):
        """Register forward hooks on specified layers."""
        for name, module in model.named_modules():
            if name in layer_names and isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self.make_hook(name))
                self.hooks.append(hook)
        print(f"Registered {len(self.hooks)} hooks")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_input_scales(self, method="max"):
        """Compute input_scale from collected amax values."""
        input_scales = {}
        for name, amaxes in self.amax_values.items():
            if method == "max":
                amax = max(amaxes)
            elif method == "mean":
                amax = sum(amaxes) / len(amaxes)
            elif method == "percentile_99":
                sorted_amaxes = sorted(amaxes)
                idx = int(len(sorted_amaxes) * 0.99)
                amax = sorted_amaxes[min(idx, len(sorted_amaxes) - 1)]
            else:
                raise ValueError(f"Unknown method: {method}")

            input_scales[name] = max(amax / FP8_MAX, 1e-6)

        return input_scales


def build_wan_model(model_path: str, device: str = "cuda"):
    """
    Build a minimal WAN model structure for calibration.
    We only need the forward pass to work, not full functionality.
    """
    import sys

    sys.path.insert(0, r"D:\comfy2\ComfyUI")

    # Try to use ComfyUI's model loading
    from comfy.ldm.wan.model import WanModel
    import comfy.ops as ops

    # Load state dict
    print(f"Loading model from {model_path}...")

    if model_path.endswith(".safetensors"):
        state_dict = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in tqdm(f.keys(), desc="Loading weights"):
                state_dict[key] = f.get_tensor(key)
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    # Infer model config from state dict
    # Check hidden size from a weight
    hidden_size = state_dict["blocks.0.self_attn.q.weight"].shape[0]
    num_blocks = (
        max(int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("blocks."))
        + 1
    )

    # Text embedding input size
    text_dim = state_dict["text_embedding.0.weight"].shape[1]

    # FFN hidden size
    ffn_dim = state_dict["blocks.0.ffn.0.weight"].shape[0]

    # Patch embedding input channels - from patch_embedding.weight shape [out, in, 1, 2, 2]
    in_dim = state_dict["patch_embedding.weight"].shape[1]

    # Output dim from head.head.weight shape [out, hidden] -> out / (patch_size product) = out_dim
    # Actually head.head output is out_dim * patch_size_product = out_dim * 1*2*2 = out_dim * 4
    head_out = state_dict["head.head.weight"].shape[0]
    out_dim = head_out // 4  # patch_size = (1,2,2), product = 4

    print(
        f"Detected: hidden_size={hidden_size}, num_blocks={num_blocks}, text_dim={text_dim}, ffn_dim={ffn_dim}, in_dim={in_dim}, out_dim={out_dim}"
    )

    # Create model - WAN 2.2 T2V 5B config
    model = WanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        in_dim=in_dim,
        dim=hidden_size,
        ffn_dim=ffn_dim,
        text_dim=text_dim,
        out_dim=out_dim,
        num_heads=hidden_size // 128,  # head_dim=128
        num_layers=num_blocks,
        dtype=torch.bfloat16,
        device="cpu",  # Load on CPU first
        operations=ops.manual_cast,  # Use ComfyUI's ops
    )

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

    model = model.to(device=device)
    model.eval()

    return model


def get_linear_layer_names(model):
    """Get names of all Linear layers in the model."""
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.append(name)
    return names


def run_calibration_forward_passes(
    model,
    in_dim: int,
    num_samples: int = 8,
    batch_size: int = 1,
    device: str = "cuda",
):
    """
    Run forward passes with synthetic data to collect activation statistics.

    WanModel expects:
    - x: [B, C, T, H, W] - video latent (before patch embedding)
    - timestep: [B] - diffusion timestep
    - context: [B, text_len, text_dim] - text encoder output
    """
    text_dim = model.text_dim
    text_len = 512  # Default text sequence length

    # Video latent shape for a small test video
    # For 480x832 video with VAE 8x compression and patch_size=(1,2,2):
    # Latent: [B, in_dim, T, H, W] where T=num_frames
    # in_dim is typically 16 for T2V or 48 for TI2V (16 noise + 16 image latent + 16 mask)
    latent_channels = in_dim
    latent_t = 5  # small number of frames
    latent_h = 30  # ~240 pixels / 8
    latent_w = 52  # ~416 pixels / 8

    print(f"Running {num_samples} calibration forward passes...")
    print(
        f"  Latent shape: [{batch_size}, {latent_channels}, {latent_t}, {latent_h}, {latent_w}]"
    )
    print(f"  Context shape: [{batch_size}, {text_len}, {text_dim}]")

    for i in tqdm(range(num_samples), desc="Calibration"):
        # Create synthetic inputs
        # x: [B, C, T, H, W] - video latent
        x = torch.randn(
            batch_size,
            latent_channels,
            latent_t,
            latent_h,
            latent_w,
            device=device,
            dtype=torch.bfloat16,
        )

        # timestep: [B] - random timestep in diffusion range
        timestep = torch.randint(
            0, 1000, (batch_size,), device=device, dtype=torch.long
        )

        # context: [B, text_len, text_dim] - text encoder output
        # Use unit normal - reasonably realistic for normalized embeddings
        context = torch.randn(
            batch_size, text_len, text_dim, device=device, dtype=torch.bfloat16
        )

        # Forward pass
        with torch.no_grad():
            try:
                _ = model(x, timestep, context)
            except Exception as e:
                print(f"Forward pass {i} error: {e}")
                # Even if forward fails partway, we may have collected some activations
                import traceback

                traceback.print_exc()
                break

        # Clear cache periodically
        if i % 2 == 0:
            torch.cuda.empty_cache()


def calibrate_from_fp16_model(
    fp16_model_path: str,
    quantized_model_path: str,
    output_path: str,
    num_samples: int = 8,
    method: str = "max",
    device: str = "cuda",
):
    """
    Full calibration pipeline:
    1. Load FP16 model
    2. Hook linear layers
    3. Run calibration forward passes
    4. Compute input_scales
    5. Add to quantized model
    """
    print(f"=== Calibrating from FP16 model ===\n")

    # Step 1: Build model
    model = build_wan_model(fp16_model_path, device)

    # Step 2: Get layer names from quantized model (only calibrate those)
    print("\nStep 2: Getting quantized layer names...")
    with safe_open(quantized_model_path, framework="pt") as f:
        meta = f.metadata()
        qmeta = json.loads(meta.get("_quantization_metadata", "{}"))
        quantized_layers = set(qmeta.get("layers", {}).keys())
    print(f"  Found {len(quantized_layers)} quantized layers")

    # Step 3: Register hooks
    print("\nStep 3: Registering activation hooks...")
    collector = ActivationCollector()
    collector.register_hooks(model, quantized_layers)

    # Step 4: Run calibration
    print("\nStep 4: Running calibration forward passes...")
    in_dim = model.patch_embedding.weight.shape[1]  # Get actual input channels
    run_calibration_forward_passes(
        model, in_dim=in_dim, num_samples=num_samples, device=device
    )

    # Step 5: Compute input scales
    print("\nStep 5: Computing input_scale values...")
    collector.remove_hooks()
    input_scales = collector.get_input_scales(method=method)

    # Show statistics
    if input_scales:
        scale_values = list(input_scales.values())
        print(f"  Collected scales for {len(input_scales)} layers")
        print(f"  input_scale range: {min(scale_values):.6f} - {max(scale_values):.6f}")
        print(f"  input_scale mean: {sum(scale_values) / len(scale_values):.6f}")
    else:
        print("  WARNING: No scales collected!")

    # Clean up model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 6: Add scales to quantized model
    print("\nStep 6: Adding input_scale to quantized model...")

    with safe_open(quantized_model_path, framework="pt") as f:
        metadata = dict(f.metadata())
        tensors = {}
        for key in tqdm(f.keys(), desc="Loading tensors"):
            tensors[key] = f.get_tensor(key)

    # Add input_scale tensors
    added = 0
    for layer_name, scale in input_scales.items():
        scale_key = f"{layer_name}.input_scale"
        tensors[scale_key] = torch.tensor(scale, dtype=torch.float32)
        added += 1

    print(f"  Added {added} input_scale tensors")

    # For layers we didn't capture (maybe model didn't reach them), use default
    for layer_name in quantized_layers:
        scale_key = f"{layer_name}.input_scale"
        if scale_key not in tensors:
            # Use a reasonable default (median of collected scales or 1.0)
            default_scale = (
                sum(input_scales.values()) / len(input_scales) if input_scales else 0.1
            )
            tensors[scale_key] = torch.tensor(default_scale, dtype=torch.float32)
            print(f"  Using default scale for {layer_name}: {default_scale:.6f}")

    # Save
    print(f"\nSaving to {output_path}...")
    save_file(tensors, output_path, metadata=metadata)

    print(f"\n=== Calibration complete! ===")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrate quantized model using FP16 model"
    )
    parser.add_argument("--fp16", required=True, help="FP16 source model path")
    parser.add_argument(
        "--quantized", required=True, help="Quantized model to add scales to"
    )
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument(
        "--samples", type=int, default=8, help="Number of calibration samples"
    )
    parser.add_argument(
        "--method",
        choices=["max", "mean", "percentile_99"],
        default="max",
        help="Method for computing input_scale",
    )
    parser.add_argument("--device", default="cuda", help="Device for calibration")

    args = parser.parse_args()

    calibrate_from_fp16_model(
        args.fp16,
        args.quantized,
        args.output,
        num_samples=args.samples,
        method=args.method,
        device=args.device,
    )


if __name__ == "__main__":
    main()
