import safetensors.torch
import os
from collections import defaultdict, Counter
import json


def analyze_safetensors(file_path):
    """Analyze a safetensors file and return detailed structure information."""
    print(f"\n{'=' * 80}")
    print(f"Analyzing: {os.path.basename(file_path)}")
    print(f"{'=' * 80}")

    # Load the safetensors file
    with safetensors.torch.safe_open(file_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        tensor_names = f.keys()

        # Statistics
        total_tensors = len(tensor_names)
        dtype_counts = Counter()
        shape_info = {}

        # Categorize tensors
        quantized_tensors = []
        non_quantized_tensors = []
        scale_tensors = []

        # Layer type categorization
        layer_types = defaultdict(list)

        for name in tensor_names:
            tensor = f.get_tensor(name)
            dtype = str(tensor.dtype)
            shape = tuple(tensor.shape)

            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            shape_info[name] = {"dtype": dtype, "shape": shape, "numel": tensor.numel()}

            # Categorize by quantization
            if "uint8" in dtype or "float8" in dtype:
                quantized_tensors.append(name)
            elif "scale" in name.lower():
                scale_tensors.append(name)
            else:
                non_quantized_tensors.append(name)

            # Categorize by layer type
            if "double_blocks" in name:
                layer_types["double_blocks"].append(name)
            elif "single_blocks" in name:
                layer_types["single_blocks"].append(name)
            elif (
                "img_in" in name
                or "txt_in" in name
                or "time_in" in name
                or "vector_in" in name
            ):
                layer_types["input_layers"].append(name)
            elif "final_layer" in name:
                layer_types["final_layer"].append(name)
            elif "norm" in name:
                layer_types["norm_layers"].append(name)
            elif "guidance_in" in name:
                layer_types["guidance"].append(name)
            else:
                layer_types["other"].append(name)

    # Print summary
    print(f"\nTotal tensors: {total_tensors}")
    print(f"Quantized tensors (uint8/float8): {len(quantized_tensors)}")
    print(f"Scale tensors: {len(scale_tensors)}")
    print(f"Non-quantized tensors: {len(non_quantized_tensors)}")

    print(f"\nDtype distribution:")
    for dtype, count in sorted(dtype_counts.items()):
        print(f"  {dtype}: {count}")

    print(f"\nMetadata:")
    if metadata:
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        print("  No metadata found")

    # Print layer type distribution
    print(f"\nLayer type distribution:")
    for layer_type, tensors in sorted(layer_types.items()):
        print(f"  {layer_type}: {len(tensors)} tensors")

    # Analyze quantization patterns by layer type
    print(f"\n{'=' * 80}")
    print("QUANTIZATION ANALYSIS BY LAYER TYPE")
    print(f"{'=' * 80}")

    for layer_type, tensors in sorted(layer_types.items()):
        print(f"\n{layer_type.upper()} ({len(tensors)} tensors):")

        # Count dtypes in this layer type
        type_dtypes = Counter()
        for name in tensors:
            type_dtypes[shape_info[name]["dtype"]] += 1

        print(f"  Dtype distribution:")
        for dtype, count in sorted(type_dtypes.items()):
            print(f"    {dtype}: {count}")

        # Show sample tensor names (first 5 and last 5)
        print(f"  Sample tensors:")
        sample_tensors = tensors[:5] + (tensors[-5:] if len(tensors) > 10 else [])
        for name in sample_tensors[:10]:
            info = shape_info[name]
            print(f"    {name}")
            print(f"      dtype={info['dtype']}, shape={info['shape']}")

    # Analyze quantization naming patterns
    print(f"\n{'=' * 80}")
    print("QUANTIZATION NAMING PATTERNS")
    print(f"{'=' * 80}")

    # Find all base names (without .weight_scale suffixes)
    base_names = set()
    scale_patterns = defaultdict(list)

    for name in tensor_names:
        if ".weight_scale" in name:
            # Extract base name and scale suffix
            if ".weight_scale_2" in name:
                base = name.replace(".weight_scale_2", "")
                scale_patterns[base].append("weight_scale_2")
            else:
                base = name.replace(".weight_scale", "")
                scale_patterns[base].append("weight_scale")
        elif ".input_scale" in name:
            base = name.replace(".input_scale", "")
            scale_patterns[base].append("input_scale")
        elif ".weight" in name and name not in [
            n for n in tensor_names if "scale" in n
        ]:
            base = name.replace(".weight", "")
            base_names.add(base)

    # Print quantization patterns
    if scale_patterns:
        print("\nQuantized layers (with scales):")
        sample_count = 0
        for base, scales in sorted(scale_patterns.items()):
            if sample_count < 10:
                weight_name = base + ".weight"
                if weight_name in tensor_names:
                    weight_info = shape_info[weight_name]
                    print(f"\n  Base: {base}")
                    print(
                        f"    {weight_name}: {weight_info['dtype']}, {weight_info['shape']}"
                    )
                    for scale in scales:
                        scale_name = base + "." + scale
                        if scale_name in tensor_names:
                            scale_info = shape_info[scale_name]
                            print(
                                f"    {scale_name}: {scale_info['dtype']}, {scale_info['shape']}"
                            )
                sample_count += 1

    # Analyze first and last blocks
    print(f"\n{'=' * 80}")
    print("FIRST/LAST BLOCK ANALYSIS")
    print(f"{'=' * 80}")

    if "double_blocks" in layer_types:
        double_block_nums = set()
        for name in layer_types["double_blocks"]:
            # Extract block number
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "double_blocks" and i + 1 < len(parts):
                    double_block_nums.add(int(parts[i + 1]))
                    break

        if double_block_nums:
            min_block = min(double_block_nums)
            max_block = max(double_block_nums)

            print(f"\nDouble blocks range: {min_block} to {max_block}")
            print(f"\nFirst double block (block {min_block}):")
            first_block_tensors = [
                n
                for n in layer_types["double_blocks"]
                if f".double_blocks.{min_block}." in n
            ]
            for name in sorted(first_block_tensors)[:10]:
                info = shape_info[name]
                print(f"  {name}: {info['dtype']}")

            print(f"\nLast double block (block {max_block}):")
            last_block_tensors = [
                n
                for n in layer_types["double_blocks"]
                if f".double_blocks.{max_block}." in n
            ]
            for name in sorted(last_block_tensors)[:10]:
                info = shape_info[name]
                print(f"  {name}: {info['dtype']}")

    if "single_blocks" in layer_types:
        single_block_nums = set()
        for name in layer_types["single_blocks"]:
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "single_blocks" and i + 1 < len(parts):
                    single_block_nums.add(int(parts[i + 1]))
                    break

        if single_block_nums:
            min_block = min(single_block_nums)
            max_block = max(single_block_nums)

            print(f"\nSingle blocks range: {min_block} to {max_block}")
            print(f"\nFirst single block (block {min_block}):")
            first_block_tensors = [
                n
                for n in layer_types["single_blocks"]
                if f".single_blocks.{min_block}." in n
            ]
            for name in sorted(first_block_tensors)[:10]:
                info = shape_info[name]
                print(f"  {name}: {info['dtype']}")

            print(f"\nLast single block (block {max_block}):")
            last_block_tensors = [
                n
                for n in layer_types["single_blocks"]
                if f".single_blocks.{max_block}." in n
            ]
            for name in sorted(last_block_tensors)[:10]:
                info = shape_info[name]
                print(f"  {name}: {info['dtype']}")

    return {
        "total_tensors": total_tensors,
        "quantized_count": len(quantized_tensors),
        "scale_count": len(scale_tensors),
        "non_quantized_count": len(non_quantized_tensors),
        "dtype_counts": dict(dtype_counts),
        "layer_types": {k: len(v) for k, v in layer_types.items()},
        "shape_info": shape_info,
        "tensor_names": list(tensor_names),
    }


def compare_models(results):
    """Compare the analysis results of multiple models."""
    print(f"\n\n{'=' * 80}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'=' * 80}")

    model_names = list(results.keys())

    # Compare tensor counts
    print("\nTensor count comparison:")
    for name in model_names:
        print(f"  {name}: {results[name]['total_tensors']} tensors")

    # Compare dtypes
    print("\nDtype comparison:")
    all_dtypes = set()
    for result in results.values():
        all_dtypes.update(result["dtype_counts"].keys())

    for dtype in sorted(all_dtypes):
        print(f"  {dtype}:")
        for name in model_names:
            count = results[name]["dtype_counts"].get(dtype, 0)
            print(f"    {name}: {count}")

    # Find missing tensors
    print("\nTensor name differences:")
    tensor_sets = {name: set(results[name]["tensor_names"]) for name in model_names}

    for i, name1 in enumerate(model_names):
        for name2 in model_names[i + 1 :]:
            only_in_1 = tensor_sets[name1] - tensor_sets[name2]
            only_in_2 = tensor_sets[name2] - tensor_sets[name1]

            if only_in_1:
                print(f"\n  Only in {name1} ({len(only_in_1)} tensors):")
                for tensor in sorted(only_in_1)[:10]:
                    print(f"    {tensor}")
                if len(only_in_1) > 10:
                    print(f"    ... and {len(only_in_1) - 10} more")

            if only_in_2:
                print(f"\n  Only in {name2} ({len(only_in_2)} tensors):")
                for tensor in sorted(only_in_2)[:10]:
                    print(f"    {tensor}")
                if len(only_in_2) > 10:
                    print(f"    ... and {len(only_in_2) - 10} more")

    # Find dtype differences for same tensors
    print("\n\nDtype differences for common tensors:")
    common_tensors = set.intersection(
        *[set(results[name]["tensor_names"]) for name in model_names]
    )

    dtype_diffs = []
    for tensor in sorted(common_tensors):
        dtypes = {
            name: results[name]["shape_info"][tensor]["dtype"] for name in model_names
        }
        if len(set(dtypes.values())) > 1:
            dtype_diffs.append((tensor, dtypes))

    if dtype_diffs:
        print(f"\n  Found {len(dtype_diffs)} tensors with different dtypes:")
        for tensor, dtypes in dtype_diffs[:20]:
            print(f"\n  {tensor}:")
            for name, dtype in dtypes.items():
                print(f"    {name}: {dtype}")
        if len(dtype_diffs) > 20:
            print(f"\n  ... and {len(dtype_diffs) - 20} more differences")
    else:
        print("  No dtype differences found for common tensors")


if __name__ == "__main__":
    models = {
        "flux1-dev-nvfp4 (WORKING)": "D:/ComfyUI/ComfyUI/models/diffusion_models/flux1-dev-nvfp4.safetensors",
        "Wan2_2-TI2V-5B_fp8 (WORKING)": "D:/ComfyUI/ComfyUI/models/diffusion_models/Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors",
        "wan2.2-ti2v-5b-nvfp4 (OURS-BROKEN)": "D:/ComfyUI/ComfyUI/models/diffusion_models/wan2.2-ti2v-5b-nvfp4-ck.safetensors",
    }

    results = {}

    for name, path in models.items():
        if os.path.exists(path):
            results[name] = analyze_safetensors(path)
        else:
            print(f"\nWARNING: File not found: {path}")

    if len(results) > 1:
        compare_models(results)

    print(f"\n\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
