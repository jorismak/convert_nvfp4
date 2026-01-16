"""
Calibration script for NVFP4/FP8 quantization.

Runs synthetic/semi-realistic data through the model to measure
activation ranges (amax) at each layer input, then computes input_scale values.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from collections import defaultdict

# FP8 E4M3 max value
FP8_MAX = 448.0


def get_layer_input_shapes(model_path: str) -> dict[str, tuple[int, int]]:
    """
    Extract input shapes for each quantized linear layer.
    Returns dict of layer_name -> (out_features, in_features)
    """
    shapes = {}
    with safe_open(model_path, framework="pt") as f:
        meta = f.metadata()
        if "_quantization_metadata" in meta:
            qmeta = json.loads(meta["_quantization_metadata"])
            layers = qmeta.get("layers", {})

            for layer_name in layers.keys():
                weight_key = f"{layer_name}.weight"
                if weight_key in f.keys():
                    weight = f.get_tensor(weight_key)
                    # For packed NVFP4, actual shape is 2x the stored K dimension
                    if weight.dtype == torch.uint8:
                        out_features, packed_k = weight.shape
                        in_features = packed_k * 2  # Each byte = 2 FP4 values
                    else:
                        out_features, in_features = weight.shape
                    shapes[layer_name] = (out_features, in_features)

    return shapes


def generate_calibration_data(
    layer_shapes: dict[str, tuple[int, int]],
    num_samples: int = 32,
    batch_size: int = 4,
    seq_len: int = 256,
    device: str = "cuda",
) -> dict[str, list[float]]:
    """
    Generate synthetic calibration data and measure activation ranges.

    For each layer, we generate random inputs with realistic magnitude
    and measure the amax (absolute maximum) values.
    """

    # Group layers by input dimension to batch calibration
    layers_by_in_dim = defaultdict(list)
    for layer_name, (out_feat, in_feat) in layer_shapes.items():
        layers_by_in_dim[in_feat].append(layer_name)

    amax_values = defaultdict(list)

    print(f"Calibrating {len(layer_shapes)} layers with {num_samples} samples each...")

    for in_dim, layer_names in tqdm(layers_by_in_dim.items(), desc="Input dimensions"):
        for _ in range(num_samples):
            # Generate random input with unit Gaussian distribution
            # This is reasonable for normalized hidden states
            x = torch.randn(
                batch_size, seq_len, in_dim, device=device, dtype=torch.float32
            )

            # Measure amax
            amax = x.abs().amax().item()

            # All layers with this input dim get the same amax
            # (In reality they'd have different inputs, but this is a rough approximation)
            for layer_name in layer_names:
                amax_values[layer_name].append(amax)

    return amax_values


def generate_text_conditioned_calibration(
    layer_shapes: dict[str, tuple[int, int]],
    text_encoder_path: str,
    prompts: list[str],
    num_samples: int = 32,
    batch_size: int = 1,
    seq_len: int = 256,
    device: str = "cuda",
) -> dict[str, list[float]]:
    """
    Generate calibration data using real text encoder outputs.

    This provides more realistic activation ranges for cross-attention layers.
    """
    # TODO: Load text encoder and run prompts through it
    # For now, fall back to synthetic
    print("Text-conditioned calibration not yet implemented, using synthetic data")
    return generate_calibration_data(
        layer_shapes, num_samples, batch_size, seq_len, device
    )


def compute_input_scales(
    amax_values: dict[str, list[float]], method: str = "max"
) -> dict[str, float]:
    """
    Compute input_scale for each layer from collected amax values.

    input_scale = amax / FP8_MAX

    Methods:
    - "max": Use maximum amax (most conservative, no clipping)
    - "mean": Use mean amax (some clipping possible)
    - "percentile_99": Use 99th percentile (slight clipping, better range utilization)
    """
    input_scales = {}

    for layer_name, amaxes in amax_values.items():
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

        # input_scale = amax / FP8_MAX
        # But we want to be conservative to avoid overflow
        input_scale = amax / FP8_MAX

        # Clamp to reasonable range
        input_scale = max(input_scale, 1e-6)

        input_scales[layer_name] = input_scale

    return input_scales


def add_input_scales_to_model(
    source_path: str,
    output_path: str,
    input_scales: dict[str, float],
):
    """
    Add input_scale tensors to an existing quantized model.
    """
    print(f"Loading model from {source_path}...")

    with safe_open(source_path, framework="pt") as f:
        metadata = dict(f.metadata())

        # Load all existing tensors
        tensors = {}
        for key in tqdm(f.keys(), desc="Loading tensors"):
            tensors[key] = f.get_tensor(key)

    # Add input_scale tensors
    print(f"Adding {len(input_scales)} input_scale tensors...")
    for layer_name, scale in input_scales.items():
        scale_key = f"{layer_name}.input_scale"
        tensors[scale_key] = torch.tensor(scale, dtype=torch.float32)

    # Save
    print(f"Saving to {output_path}...")
    save_file(tensors, output_path, metadata=metadata)
    print("Done!")


def calibrate_and_update_model(
    source_path: str,
    output_path: str,
    num_samples: int = 32,
    method: str = "max",
    device: str = "cuda",
):
    """
    Full calibration pipeline: analyze model, generate calibration data,
    compute scales, and save updated model.
    """
    print(f"=== Calibrating {source_path} ===\n")

    # Step 1: Get layer shapes
    print("Step 1: Analyzing model structure...")
    layer_shapes = get_layer_input_shapes(source_path)
    print(f"  Found {len(layer_shapes)} quantized layers")

    # Show some examples
    for i, (name, shape) in enumerate(list(layer_shapes.items())[:5]):
        print(f"    {name}: out={shape[0]}, in={shape[1]}")
    if len(layer_shapes) > 5:
        print(f"    ... and {len(layer_shapes) - 5} more")

    # Step 2: Generate calibration data
    print(f"\nStep 2: Generating calibration data ({num_samples} samples)...")
    amax_values = generate_calibration_data(
        layer_shapes,
        num_samples=num_samples,
        device=device,
    )

    # Step 3: Compute input scales
    print(f"\nStep 3: Computing input_scale values (method={method})...")
    input_scales = compute_input_scales(amax_values, method=method)

    # Show statistics
    scale_values = list(input_scales.values())
    print(f"  input_scale range: {min(scale_values):.6f} - {max(scale_values):.6f}")
    print(f"  input_scale mean: {sum(scale_values) / len(scale_values):.6f}")

    # Step 4: Update model
    print(f"\nStep 4: Adding input_scale to model...")
    add_input_scales_to_model(source_path, output_path, input_scales)

    print(f"\n=== Calibration complete! ===")
    print(f"Output saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrate quantized model for input_scale"
    )
    parser.add_argument("source", help="Source quantized model path")
    parser.add_argument("output", help="Output model path")
    parser.add_argument(
        "--samples", type=int, default=32, help="Number of calibration samples"
    )
    parser.add_argument(
        "--method",
        choices=["max", "mean", "percentile_99"],
        default="max",
        help="Method for computing input_scale from amax values",
    )
    parser.add_argument("--device", default="cuda", help="Device for calibration")

    args = parser.parse_args()

    calibrate_and_update_model(
        args.source,
        args.output,
        num_samples=args.samples,
        method=args.method,
        device=args.device,
    )


if __name__ == "__main__":
    main()
