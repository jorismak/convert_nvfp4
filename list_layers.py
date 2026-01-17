#!/usr/bin/env python3
"""
List layer/block names from safetensors (single file or sharded index/directory).

Usage:
  python list_layers.py path/to/model.safetensors
  python list_layers.py path/to/model.safetensors.index.json
  python list_layers.py path/to/sharded_dir/

Options:
  --mode layer|tensor   Output base layer names (default: layer) or raw tensor names
  --filter REGEX       Only include names matching regex
    --only-blocks        Convenience filter for r"(^|\\.)blocks\\.\\d+\\."
    --show-dtype         Append dtype/format info to each line (may be slow)
  --sort               Sort output (default: True)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set

from safetensors import safe_open

KNOWN_SUFFIXES = {
    ".weight",
    ".bias",
    ".input_scale",
    ".weight_scale",
    ".weight_scale_2",
    ".scale_weight",
    ".scale_input",
    ".scale",
}


def _detect_input(path: Path) -> List[str]:
    if path.is_dir():
        index_files = list(path.glob("*.safetensors.index.json"))
        if index_files:
            return _load_index(index_files[0])
        safes = list(path.glob("*.safetensors"))
        if len(safes) == 1:
            return _load_safetensors(safes[0])
        raise FileNotFoundError("No index.json or single safetensors found in directory")

    if path.suffix == ".json" or str(path).endswith(".index.json"):
        return _load_index(path)

    if path.suffix == ".safetensors":
        return _load_safetensors(path)

    raise ValueError(f"Unsupported input: {path}")


def _load_index(index_path: Path) -> List[str]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map", {})
    return list(weight_map.keys())


def _load_safetensors(file_path: Path) -> List[str]:
    with safe_open(file_path, framework="pt", device="cpu") as f:
        return list(f.keys())


def _load_index_map(index_path: Path) -> Dict[str, Path]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map", {})
    base = index_path.parent
    return {name: base / shard for name, shard in weight_map.items()}


def _dtype_label(dtype) -> str:
    s = str(dtype)
    if "float8_e4m3fn" in s:
        return "fp8_e4m3fn"
    if "float8_e5m2" in s:
        return "fp8_e5m2"
    if "bfloat16" in s:
        return "bf16"
    if "float16" in s:
        return "fp16"
    if "float32" in s:
        return "fp32"
    if "uint8" in s:
        return "uint8"
    if "int8" in s:
        return "int8"
    return s


def _load_dtypes(input_path: Path, tensor_names: List[str]) -> Dict[str, str]:
    dtypes: Dict[str, str] = {}

    if input_path.is_dir():
        index_files = list(input_path.glob("*.safetensors.index.json"))
        if index_files:
            shard_map = _load_index_map(index_files[0])
            shard_to_names: Dict[Path, List[str]] = {}
            for name in tensor_names:
                shard = shard_map.get(name)
                if shard is None:
                    continue
                shard_to_names.setdefault(shard, []).append(name)
            for shard_path, names in shard_to_names.items():
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for name in names:
                        try:
                            dtypes[name] = _dtype_label(f.get_tensor(name).dtype)
                        except Exception:
                            continue
            return dtypes

        safes = list(input_path.glob("*.safetensors"))
        if len(safes) == 1:
            input_path = safes[0]

    if input_path.suffix == ".json" or str(input_path).endswith(".index.json"):
        shard_map = _load_index_map(input_path)
        shard_to_names: Dict[Path, List[str]] = {}
        for name in tensor_names:
            shard = shard_map.get(name)
            if shard is None:
                continue
            shard_to_names.setdefault(shard, []).append(name)
        for shard_path, names in shard_to_names.items():
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for name in names:
                    try:
                        dtypes[name] = _dtype_label(f.get_tensor(name).dtype)
                    except Exception:
                        continue
        return dtypes

    if input_path.suffix == ".safetensors":
        with safe_open(input_path, framework="pt", device="cpu") as f:
            for name in tensor_names:
                try:
                    dtypes[name] = _dtype_label(f.get_tensor(name).dtype)
                except Exception:
                    continue
        return dtypes

    return dtypes


def _base_layer_name(name: str) -> str:
    for suffix in KNOWN_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    # Fallback: strip last segment after dot
    if "." in name:
        return name.rsplit(".", 1)[0]
    return name


def _filter(names: Iterable[str], regex: str | None) -> List[str]:
    if not regex:
        return list(names)
    pattern = re.compile(regex)
    return [n for n in names if pattern.search(n)]


def _summarize_layer_format(layer: str, dtype_map: Dict[str, str]) -> str:
    weight = f"{layer}.weight"
    w_dtype = dtype_map.get(weight)
    if not w_dtype:
        return "unknown"

    block_scale = dtype_map.get(f"{layer}.weight_scale")
    tensor_scale = dtype_map.get(f"{layer}.weight_scale_2")
    input_scale = dtype_map.get(f"{layer}.input_scale")

    if w_dtype == "uint8" and block_scale and tensor_scale:
        fmt = f"nvfp4(weight:{w_dtype}, block_scale:{block_scale}, tensor_scale:{tensor_scale})"
    elif w_dtype.startswith("fp8"):
        if block_scale:
            fmt = f"fp8(weight:{w_dtype}, scale:{block_scale})"
        else:
            fmt = f"fp8(weight:{w_dtype})"
    else:
        fmt = w_dtype

    if input_scale:
        fmt += f", input_scale:{input_scale}"
    return fmt


def main() -> None:
    parser = argparse.ArgumentParser(description="List layer/block names from safetensors")
    parser.add_argument("input", help="Path to safetensors, index.json, or directory")
    parser.add_argument("--mode", choices=["layer", "tensor"], default="layer")
    parser.add_argument("--filter", default=None, help="Regex filter")
    parser.add_argument("--only-blocks", action="store_true", help="Filter to blocks.N.*")
    parser.add_argument("--show-dtype", action="store_true", help="Append dtype/format info")
    parser.add_argument("--sort", action="store_true", default=True)
    args = parser.parse_args()

    names = _detect_input(Path(args.input))

    if args.mode == "layer":
        names = [_base_layer_name(n) for n in names]
        # de-duplicate
        names = list(dict.fromkeys(names))

    if args.only_blocks:
        args.filter = r"(^|\.)blocks\.\d+\."

    names = _filter(names, args.filter)

    if args.sort:
        names = sorted(names)

    dtype_map: Dict[str, str] = {}
    if args.show_dtype:
        dtype_map = _load_dtypes(Path(args.input), _detect_input(Path(args.input)))

    for name in names:
        if args.show_dtype:
            if args.mode == "layer":
                fmt = _summarize_layer_format(name, dtype_map)
                print(f"{name}\t{fmt}")
            else:
                dtype = dtype_map.get(name, "unknown")
                print(f"{name}\t{dtype}")
        else:
            print(name)


if __name__ == "__main__":
    main()
