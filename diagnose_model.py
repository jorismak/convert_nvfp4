#!/usr/bin/env python3
"""
Diagnostic tool to compare model structure and metadata between our converted models
and working reference models to identify structural differences.
"""

import safetensors
import sys
from pathlib import Path
import numpy as np


def analyze_model_structure(model_path):
    """Analyze the structure of a safetensors model."""
    print(f"\n{'=' * 80}")
    print(f"Analyzing: {Path(model_path).name}")
    print(f"{'=' * 80}")

    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        keys = f.keys()

        print(f"\n📊 Basic Stats:")
        print(f"   Total tensors: {len(keys)}")
        print(f"   File size: {Path(model_path).stat().st_size / (1024**3):.2f} GB")

        print(f"\n📋 Metadata:")
        if metadata:
            for k, v in sorted(metadata.items()):
                print(f"   {k}: {v}")
        else:
            print("   No metadata found")

        # Categorize tensors
        weight_tensors = []
        scale_tensors = []
        scale2_tensors = []
        input_scale_tensors = []
        other_tensors = []

        for key in sorted(keys):
            if key.endswith(".weight_scale_2"):
                scale2_tensors.append(key)
            elif key.endswith(".weight_scale"):
                scale_tensors.append(key)
            elif key.endswith(".input_scale"):
                input_scale_tensors.append(key)
            elif key.endswith(".weight") or key.endswith(".bias"):
                weight_tensors.append(key)
            else:
                other_tensors.append(key)

        print(f"\n🔢 Tensor Categories:")
        print(f"   Weight/bias tensors: {len(weight_tensors)}")
        print(f"   Block scales (.weight_scale): {len(scale_tensors)}")
        print(f"   Tensor scales (.weight_scale_2): {len(scale2_tensors)}")
        print(f"   Input scales (.input_scale): {len(input_scale_tensors)}")
        print(f"   Other tensors: {len(other_tensors)}")

        # Sample tensor info
        print(f"\n📦 Sample Tensor Details (first 3 weight tensors):")
        for key in weight_tensors[:3]:
            tensor = f.get_tensor(key)
            print(f"\n   {key}:")
            print(f"      dtype: {tensor.dtype}")
            print(f"      shape: {tensor.shape}")
            print(f"      size: {tensor.numel()} elements")

            # Check for corresponding scales
            base_key = key.rsplit(".", 1)[0]
            has_scale = f"{base_key}.weight_scale" in keys
            has_scale2 = f"{base_key}.weight_scale_2" in keys
            has_input_scale = f"{base_key}.input_scale" in keys

            print(f"      Has weight_scale: {has_scale}")
            print(f"      Has weight_scale_2: {has_scale2}")
            print(f"      Has input_scale: {has_input_scale}")

            if has_scale:
                scale = f.get_tensor(f"{base_key}.weight_scale")
                print(f"      weight_scale shape: {scale.shape}, dtype: {scale.dtype}")

            if has_scale2:
                scale2 = f.get_tensor(f"{base_key}.weight_scale_2")
                print(
                    f"      weight_scale_2 shape: {scale2.shape}, dtype: {scale2.dtype}"
                )
                print(f"      weight_scale_2 value: {scale2.item():.6e}")

            if has_input_scale:
                input_scale = f.get_tensor(f"{base_key}.input_scale")
                print(
                    f"      input_scale shape: {input_scale.shape}, dtype: {input_scale.dtype}"
                )
                print(f"      input_scale value: {input_scale.item():.6e}")

        # Check input_scale statistics
        if input_scale_tensors:
            print(f"\n📊 Input Scale Statistics:")
            input_scale_values = []
            for key in input_scale_tensors:
                val = f.get_tensor(key).item()
                input_scale_values.append(val)

            input_scale_values = np.array(input_scale_values)
            print(f"   Count: {len(input_scale_values)}")
            print(f"   Min: {input_scale_values.min():.6e}")
            print(f"   Max: {input_scale_values.max():.6e}")
            print(f"   Mean: {input_scale_values.mean():.6e}")
            print(f"   Median: {np.median(input_scale_values):.6e}")
            print(f"   Std: {input_scale_values.std():.6e}")

        return {
            "metadata": metadata,
            "keys": keys,
            "weight_tensors": weight_tensors,
            "scale_tensors": scale_tensors,
            "scale2_tensors": scale2_tensors,
            "input_scale_tensors": input_scale_tensors,
            "other_tensors": other_tensors,
        }


def compare_models(model1_path, model2_path):
    """Compare two models structurally."""
    print(f"\n{'=' * 80}")
    print(f"COMPARING MODELS")
    print(f"{'=' * 80}")
    print(f"Model 1: {Path(model1_path).name}")
    print(f"Model 2: {Path(model2_path).name}")

    info1 = analyze_model_structure(model1_path)
    info2 = analyze_model_structure(model2_path)

    print(f"\n{'=' * 80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'=' * 80}")

    # Compare tensor sets
    keys1 = set(info1["keys"])
    keys2 = set(info2["keys"])

    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common = keys1 & keys2

    print(f"\n🔍 Tensor Key Comparison:")
    print(f"   Tensors in both models: {len(common)}")
    print(f"   Only in model 1: {len(only_in_1)}")
    print(f"   Only in model 2: {len(only_in_2)}")

    if only_in_1:
        print(f"\n   Tensors only in model 1 (first 10):")
        for key in sorted(only_in_1)[:10]:
            print(f"      - {key}")

    if only_in_2:
        print(f"\n   Tensors only in model 2 (first 10):")
        for key in sorted(only_in_2)[:10]:
            print(f"      - {key}")

    # Compare metadata
    print(f"\n📋 Metadata Comparison:")
    meta1 = info1["metadata"] or {}
    meta2 = info2["metadata"] or {}

    all_meta_keys = set(meta1.keys()) | set(meta2.keys())

    for key in sorted(all_meta_keys):
        val1 = meta1.get(key, "MISSING")
        val2 = meta2.get(key, "MISSING")
        if val1 != val2:
            print(f"   {key}:")
            print(f"      Model 1: {val1}")
            print(f"      Model 2: {val2}")
        else:
            print(f"   {key}: {val1} (same)")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Single model analysis
        analyze_model_structure(sys.argv[1])
    elif len(sys.argv) == 3:
        # Compare two models
        compare_models(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("  Analyze single model: python diagnose_model.py <model_path>")
        print(
            "  Compare two models:   python diagnose_model.py <model1_path> <model2_path>"
        )
        sys.exit(1)
