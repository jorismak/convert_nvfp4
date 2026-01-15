#!/usr/bin/env python3
"""
NVFP4 Quantization Converter for ComfyUI

Converts safetensors diffusion models to NVFP4 (4-bit floating point) quantization
format compatible with ComfyUI's mixed precision system.

Supports both single safetensors files and sharded models (multiple .safetensors
files with an index.json).

Supported models:
- Wan2.1 (t2v, i2v, vace, camera, etc.)
- Wan2.2 (t2v, s2v, animate, etc.)
- Qwen Image / Qwen Image Edit
- Any model with Linear layers

Usage:
  python convert_nvfp4.py input.safetensors output.safetensors [options]
  python convert_nvfp4.py model_index.json output.safetensors [options]
  python convert_nvfp4.py /path/to/sharded_model/ output.safetensors [options]

Examples:
  # Convert single file
  python convert_nvfp4.py model.safetensors model_nvfp4.safetensors

  # Convert sharded model (via index.json)
  python convert_nvfp4.py model.safetensors.index.json model_nvfp4.safetensors

  # Convert sharded model (via directory)
  python convert_nvfp4.py ./my_model/ model_nvfp4.safetensors

  # Safe mode - skip sensitive layers
  python convert_nvfp4.py model.safetensors model_nvfp4.safetensors --mode safe

  # Preview what would be quantized
  python convert_nvfp4.py model.safetensors output.safetensors --dry-run

  # With basic calibration
  python convert_nvfp4.py model.safetensors model_nvfp4.safetensors --calibrate
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# =============================================================================
# Constants
# =============================================================================

F4_E2M1_MAX = 6.0  # Maximum value representable in FP4 E2M1
F8_E4M3_MAX = 448.0  # Maximum value representable in FP8 E4M3
NVFP4_BLOCK_SIZE = 16  # Number of values per quantization block

# Patterns for "safe" mode - layers to skip
SAFE_MODE_SKIP_PATTERNS = [
    r".*\.head\..*",  # Output projection heads
    r".*modulation.*",  # Adaptive layer norm params
    r".*time_embedding.*",  # Time conditioning
    r".*time_projection.*",  # Time projection
    r".*text_embedding.*",  # Text conditioning
    r".*img_emb.*",  # Image embeddings
    r".*img_in\..*",  # Image input projections (but not img_mlp)
    r".*txt_in\..*",  # Text input projections
    r".*patch_embedding.*",  # Patch embeddings (Conv3d anyway)
    r".*pe_embedder.*",  # Positional embeddings
    r".*\.norm.*",  # Normalization layers
]

# Minimum dimensions for quantization (skip small matrices)
MIN_DIM_FOR_QUANTIZATION = 256

# =============================================================================
# Presets for specific model architectures
# =============================================================================

# Wan 2.1/2.2 preset configuration
# These models have 32 transformer blocks by default (5B), or 40 blocks (14B)
# Conservative preset: skip first 2 and last 2 blocks
# Works for all Wan variants: T2V, I2V, Camera, Vace, S2V, Animate, HuMo, etc.
WAN_PRESET_CONFIG = {
    "name": "wan",
    "description": "Wan 2.1/2.2 video models (conservative - skips first/last 2 blocks)",
    "block_prefix": "blocks",  # Wan uses "blocks.N.xxx"
    "skip_first_n_blocks": 2,  # blocks.0, blocks.1
    "skip_last_n_blocks": 2,  # blocks.30, blocks.31 (for 32-layer model)
    # Additional patterns to skip (on top of safe mode patterns)
    "extra_skip_patterns": [
        r".*\.head\..*",  # Output projection head
        r".*modulation.*",  # All modulation parameters
        r".*time_embedding.*",  # Time conditioning
        r".*time_projection.*",  # Time projection
        r".*text_embedding.*",  # Text conditioning
        r".*img_emb.*",  # Image embeddings (for i2v models)
        r".*patch_embedding.*",  # Patch embeddings
        r".*\.norm.*",  # Normalization layers
        r".*vace_patch_embedding.*",  # Vace patch embedding
        r".*vace_blocks\..*",  # Vace blocks (keep at full precision)
        r".*control_adapter.*",  # Camera control adapter
        r".*ref_conv.*",  # Reference convolution
    ],
}

# Qwen Image preset configuration (for qwen-image-2512 - Dec 2025)
# These models have 60 transformer blocks by default
# Conservative preset: skip first 3 and last 3 blocks
QWEN_IMAGE_PRESET_CONFIG = {
    "name": "qwen-image",
    "description": "Qwen Image 2512 models (conservative - skips first/last 3 blocks)",
    "block_prefix": "transformer_blocks",  # Qwen uses "transformer_blocks.N.xxx"
    "skip_first_n_blocks": 3,  # transformer_blocks.0, 1, 2
    "skip_last_n_blocks": 3,  # transformer_blocks.57, 58, 59 (for 60-layer model)
    # Additional patterns to skip
    "extra_skip_patterns": [
        r".*\.img_in\..*",  # Image input projection
        r".*\.txt_in\..*",  # Text input projection
        r".*\.txt_norm\..*",  # Text normalization
        r".*\.norm_out\..*",  # Output normalization
        r".*\.proj_out\..*",  # Final output projection
        r".*time_text_embed.*",  # Timestep embeddings
        r".*pe_embedder.*",  # Positional embeddings
        r".*\.norm.*",  # All normalization layers
        r".*_mod\.",  # Modulation layers (img_mod, txt_mod)
    ],
}

# Qwen Image Edit preset configuration (for qwen-image-edit-2511 - Nov 2025)
# Same architecture as Qwen Image but may have different sensitivity
QWEN_IMAGE_EDIT_PRESET_CONFIG = {
    "name": "qwen-image-edit",
    "description": "Qwen Image Edit 2511 models (conservative - skips first/last 3 blocks)",
    "block_prefix": "transformer_blocks",
    "skip_first_n_blocks": 3,
    "skip_last_n_blocks": 3,
    "extra_skip_patterns": [
        r".*\.img_in\..*",  # Image input projection
        r".*\.txt_in\..*",  # Text input projection
        r".*\.txt_norm\..*",  # Text normalization
        r".*\.norm_out\..*",  # Output normalization
        r".*\.proj_out\..*",  # Final output projection
        r".*time_text_embed.*",  # Timestep embeddings
        r".*pe_embedder.*",  # Positional embeddings
        r".*\.norm.*",  # All normalization layers
        r".*_mod\.",  # Modulation layers
        r".*__index_timestep_zero__.*",  # Special index tensor (edit model marker)
    ],
}

# =============================================================================
# Unsloth-derived presets (based on GGUF quantization analysis)
# =============================================================================

# Unsloth Qwen Image Edit 2511 preset - derived from Q4_K_M GGUF analysis
# Skip patterns based on layers that Unsloth kept at Q6_K or higher precision
# Layers at Q4_K/Q5_K will be quantized to NVFP4, Q6_K+ kept at FP16 (or FP8 with --use-fp8)
#
# GGUF analysis summary:
#   Q4_K (560 tensors): Most Q/K/out projections, mlp.net.0, mods for blocks 2-57
#   Q5_K (20 tensors): Block 1 and 58 (transition blocks)
#   Q6_K (258 tensors): All V projections, mlp.net.2, entire blocks 0 and 59
#   Q8_0 (2 tensors): Block 0 and 59 img_mod
#   BF16 (6 tensors): img_in, txt_in, proj_out, norm_out, timestep_embed
#
# With --use-fp8: Q6_K and Q8_0 layers become FP8 instead of FP16

# Patterns for layers that MUST stay at FP16 (BF16 in GGUF - I/O layers)
UNSLOTH_QWEN_IMAGE_EDIT_2511_SKIP_PATTERNS = [
    r"^img_in\b",  # Image input projection
    r"^txt_in\b",  # Text input projection
    r"^proj_out\b",  # Final output projection
    r"^norm_out\b",  # Output normalization
    r"^time_text_embed\.",  # Timestep embeddings
]

# Patterns for layers that can be FP8 (Q6_K/Q8_0 in GGUF)
# Without --use-fp8, these are also skipped (kept at FP16)
UNSLOTH_QWEN_IMAGE_EDIT_2511_FP8_PATTERNS = [
    # === First block (block 0) - all weights were Q6_K/Q8_0 ===
    r"^transformer_blocks\.0\.",
    # === Last block (block 59) - all weights were Q6_K/Q8_0 ===
    r"^transformer_blocks\.59\.",
    # === V projections - ALL blocks kept at Q6_K ===
    r"\.attn\.to_v\b",  # Image V projection
    r"\.attn\.add_v_proj\b",  # Text V projection
    # === MLP down projections - ALL blocks kept at Q6_K ===
    r"\.img_mlp\.net\.2\b",  # Image MLP down projection
    r"\.txt_mlp\.net\.2\b",  # Text MLP down projection
]

UNSLOTH_QWEN_IMAGE_EDIT_2511_CONFIG = {
    "name": "unsloth-qwen-image-edit-2511",
    "type": "static",  # Static preset - hardcoded patterns, not dynamic
    "description": (
        "Derived from Unsloth's Q4_K_M quantization. "
        "Skips V projections, MLP down projections, first/last blocks, and I/O layers."
    ),
    "skip_patterns": UNSLOTH_QWEN_IMAGE_EDIT_2511_SKIP_PATTERNS,
    "fp8_patterns": UNSLOTH_QWEN_IMAGE_EDIT_2511_FP8_PATTERNS,
}

# Unsloth Qwen Image 2512 preset - derived from Q5_K_S GGUF analysis
# Skip patterns based on layers that Unsloth kept at Q6_K or higher precision
# Layers at Q5_K will be quantized to NVFP4, Q6_K+ kept at FP16 (or FP8 with --use-fp8)
#
# GGUF analysis summary:
#   Q5_K (784 tensors): All layers in blocks 2-57
#   Q6_K (56 tensors): Entire blocks 0, 1, 58, 59 (first 2 + last 2)
#   BF16 (6 tensors): img_in, txt_in, proj_out, norm_out, timestep_embed
#
# With --use-fp8: Q6_K layers (blocks 0,1,58,59) become FP8 instead of FP16

# Patterns for layers that MUST stay at FP16 (BF16 in GGUF - I/O layers)
UNSLOTH_QWEN_IMAGE_2512_SKIP_PATTERNS = [
    r"^img_in\b",  # Image input projection
    r"^txt_in\b",  # Text input projection
    r"^proj_out\b",  # Final output projection
    r"^norm_out\b",  # Output normalization
    r"^time_text_embed\.",  # Timestep embeddings
]

# Patterns for layers that can be FP8 (Q6_K in GGUF)
# Without --use-fp8, these are also skipped (kept at FP16)
UNSLOTH_QWEN_IMAGE_2512_FP8_PATTERNS = [
    # === First 2 blocks (blocks 0, 1) - all weights were Q6_K ===
    r"^transformer_blocks\.0\.",
    r"^transformer_blocks\.1\.",
    # === Last 2 blocks (blocks 58, 59) - all weights were Q6_K ===
    r"^transformer_blocks\.58\.",
    r"^transformer_blocks\.59\.",
]

UNSLOTH_QWEN_IMAGE_2512_CONFIG = {
    "name": "unsloth-qwen-image-2512",
    "type": "static",  # Static preset - hardcoded patterns, not dynamic
    "description": (
        "Derived from Unsloth's Q5_K_S quantization. "
        "Skips first/last 2 blocks and I/O layers."
    ),
    "skip_patterns": UNSLOTH_QWEN_IMAGE_2512_SKIP_PATTERNS,
    "fp8_patterns": UNSLOTH_QWEN_IMAGE_2512_FP8_PATTERNS,
}

# Unsloth Wan preset - derived from Q5_K_M GGUF analysis
# Skip patterns based on layers that Unsloth kept at Q6_K or higher precision
# Layers at Q5_K will be quantized to NVFP4, Q6_K+ kept at FP16 (or FP8 with --use-fp8)
#
# GGUF analysis summary (identical pattern across all Wan models):
#   14B (40 blocks): Q5_K=280, Q6_K=120
#   5B (30 blocks):  Q5_K=210, Q6_K=90
#   Pattern: 7 tensors/block quantized, 3 tensors/block skipped
#
#   Q5_K: Q, K, O projections + ffn.0 (up proj) for all blocks
#   Q6_K: V projections (self_attn.v, cross_attn.v) + ffn.2 (down proj) for all blocks
#   F16: All biases, norms, modulation (1D tensors)
#   F32: patch_embedding.weight (Conv3d)
#
# With --use-fp8: Q6_K layers (V proj, ffn.2) become FP8 instead of FP16
# Works for ALL Wan models: 2.1, 2.2, 5B, 14B, T2V, I2V, TI2V, MAGREF, etc.

# Patterns for layers that MUST stay at FP16 (F16/F32 in GGUF)
# For Wan, all the F16 layers are biases/norms (1D) which we don't quantize anyway
# So no explicit skip patterns needed - they're handled by is_linear_weight check
UNSLOTH_WAN_SKIP_PATTERNS = []

# Patterns for layers that can be FP8 (Q6_K in GGUF)
# Without --use-fp8, these are also skipped (kept at FP16)
UNSLOTH_WAN_FP8_PATTERNS = [
    # === V projections - ALL blocks kept at Q6_K ===
    r"\.self_attn\.v\b",  # Self-attention V projection
    r"\.cross_attn\.v\b",  # Cross-attention V projection
    # === FFN down projections - ALL blocks kept at Q6_K ===
    r"\.ffn\.2\b",  # FFN down projection (ffn.2.weight)
]

UNSLOTH_WAN_CONFIG = {
    "name": "unsloth-wan",
    "type": "static",  # Static preset - hardcoded patterns, not dynamic
    "description": (
        "Derived from Unsloth's Q5_K_M quantization for Wan models. "
        "Skips V projections and FFN down projections. Works for all Wan variants."
    ),
    "skip_patterns": UNSLOTH_WAN_SKIP_PATTERNS,
    "fp8_patterns": UNSLOTH_WAN_FP8_PATTERNS,
}

# =============================================================================
# Smart Preset - Auto-detects model architecture and applies Unsloth-like rules
# =============================================================================

# The "smart" preset learns from Unsloth's quantization patterns:
# 1. V projections (value in attention) → FP8 (they aggregate info, sensitive to precision)
# 2. Down projections (gate before residual) → FP8 (final bottleneck, quality-critical)
# 3. I/O layers (embeddings, heads, norms) → FP16 (always preserved)
# 4. Everything else → NVFP4 (Q, K, O projections, up projections, etc.)
#
# This is a "smart" preset that auto-detects common layer naming patterns.

# Common patterns for V projections across architectures
SMART_V_PROJECTION_PATTERNS = [
    # Wan-style: blocks.N.self_attn.v, blocks.N.cross_attn.v
    r"\.self_attn\.v\b",
    r"\.cross_attn\.v\b",
    # Qwen-style: transformer_blocks.N.attn.to_v, transformer_blocks.N.attn.add_v_proj
    r"\.attn\.to_v\b",
    r"\.attn\.add_v_proj\b",
    # Generic patterns for other architectures
    r"\.v_proj\b",
    r"\.to_v\b",
    r"\.value\b",
    r"\.wv\b",  # Some models use wv for value projection
]

# Common patterns for down/gate projections across architectures
SMART_DOWN_PROJECTION_PATTERNS = [
    # Wan-style: blocks.N.ffn.2
    r"\.ffn\.2\b",
    # Qwen-style: transformer_blocks.N.img_mlp.net.2, transformer_blocks.N.txt_mlp.net.2
    r"\.mlp\.net\.2\b",
    r"\.img_mlp\.net\.2\b",
    r"\.txt_mlp\.net\.2\b",
    # Generic FFN down projection patterns
    r"\.mlp\.down_proj\b",
    r"\.mlp\.w2\b",
    r"\.mlp\.c_proj\b",
    r"\.feed_forward\.w2\b",
    r"\.ff\.net\.2\b",  # Some use ff instead of ffn
    r"\.ffn\.down\b",
    r"\.fc2\b",  # Simple feedforward naming
]

# Common patterns for I/O layers that should always stay FP16
SMART_IO_SKIP_PATTERNS = [
    # Input projections
    r"^img_in\b",
    r"^txt_in\b",
    r"^patch_embed",
    r"^embed",
    r"^wte\b",  # Token embeddings (GPT-style)
    r"^wpe\b",  # Position embeddings (GPT-style)
    r"\.patch_embedding\b",
    r"\.pos_embed",
    r"\.token_embed",
    # Output projections
    r"^proj_out\b",
    r"^head\b",
    r"^lm_head\b",
    r"\.head\b",
    r"^final_layer\b",
    r"^output\b",
    # Normalization (usually 1D but catch any 2D variants)
    r"^norm_out\b",
    r"\.norm_out\b",
    r"^final_norm\b",
    r"\.final_layer_norm\b",
    # Time/text embeddings
    r"^time_embed",
    r"^time_text_embed",
    r"^timestep_embed",
    r"\.time_embedding\b",
    r"\.text_embedding\b",
    # Modulation layers (usually 1D but catch any)
    r"modulation",
]

SMART_PRESET_CONFIG = {
    "name": "smart",
    "type": "smart",  # Special type - dynamically generates patterns
    "description": (
        "Auto-detects model architecture and applies Unsloth-like quantization. "
        "V projections and down projections → FP8, I/O layers → FP16, rest → NVFP4."
    ),
    "skip_patterns": SMART_IO_SKIP_PATTERNS,
    "fp8_patterns": SMART_V_PROJECTION_PATTERNS + SMART_DOWN_PROJECTION_PATTERNS,
}

# Available presets
PRESETS = {
    "smart": SMART_PRESET_CONFIG,
    "wan": WAN_PRESET_CONFIG,
    "qwen-image": QWEN_IMAGE_PRESET_CONFIG,
    "qwen-image-edit": QWEN_IMAGE_EDIT_PRESET_CONFIG,
    "unsloth-qwen-image-edit-2511": UNSLOTH_QWEN_IMAGE_EDIT_2511_CONFIG,
    "unsloth-qwen-image-2512": UNSLOTH_QWEN_IMAGE_2512_CONFIG,
    "unsloth-wan": UNSLOTH_WAN_CONFIG,
}


def get_preset_skip_patterns(
    preset_name: str, num_layers: int
) -> Tuple[List[str], List[str]]:
    """
    Generate skip patterns and FP8 patterns for a preset based on model configuration.

    Args:
        preset_name: Name of the preset (e.g., "wan", "qwen-image")
        num_layers: Number of transformer blocks in the model

    Returns:
        Tuple of (skip_patterns, fp8_patterns):
        - skip_patterns: Patterns for layers that must stay FP16
        - fp8_patterns: Patterns for layers that can be FP8 (or FP16 if --use-fp8 not set)
    """
    if preset_name not in PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}"
        )

    config = PRESETS[preset_name]

    # Handle static and smart presets (hardcoded patterns, no dynamic generation)
    if config.get("type") in ("static", "smart"):
        skip_patterns = list(config.get("skip_patterns", []))
        fp8_patterns = list(config.get("fp8_patterns", []))
        return skip_patterns, fp8_patterns

    # Dynamic presets - generate patterns based on num_layers
    # Dynamic presets don't have FP8 patterns (they were designed before this feature)
    patterns = list(config.get("extra_skip_patterns", []))

    # Get block prefix (different models use different naming)
    block_prefix = config.get("block_prefix", "blocks")

    # Generate block skip patterns based on num_layers
    skip_first = config.get("skip_first_n_blocks", 0)
    skip_last = config.get("skip_last_n_blocks", 0)

    if skip_first > 0:
        # Pattern to match first N blocks
        first_blocks = "|".join(str(i) for i in range(skip_first))
        patterns.append(rf".*{block_prefix}\.({first_blocks})\..*")

    if skip_last > 0:
        # Pattern to match last N blocks
        last_block_indices = [num_layers - 1 - i for i in range(skip_last)]
        last_blocks = "|".join(str(i) for i in sorted(last_block_indices))
        patterns.append(rf".*{block_prefix}\.({last_blocks})\..*")

    return patterns, []  # Dynamic presets have no FP8 patterns


def detect_num_layers(tensor_names: List[str]) -> int:
    """
    Detect the number of transformer blocks from tensor names.
    Supports both Wan-style (blocks.N) and Qwen-style (transformer_blocks.N).

    Args:
        tensor_names: List of tensor names in the model

    Returns:
        Number of blocks detected
    """
    block_indices = set()

    # Try multiple block naming patterns
    block_patterns = [
        re.compile(r"blocks\.(\d+)\."),  # Wan models
        re.compile(r"transformer_blocks\.(\d+)\."),  # Qwen Image models
    ]

    for name in tensor_names:
        for pattern in block_patterns:
            match = pattern.search(name)
            if match:
                block_indices.add(int(match.group(1)))
                break

    if not block_indices:
        return 0

    return max(block_indices) + 1


def detect_model_type(tensor_names: List[str]) -> Optional[str]:
    """
    Detect the model type from tensor names.

    Args:
        tensor_names: List of tensor names in the model

    Returns:
        Detected model type or None
    """
    tensor_set = set(tensor_names)

    # Wan model detection (same logic as ComfyUI's model_detection.py)
    if any("head.modulation" in n for n in tensor_names):
        # It's a Wan model
        if any("vace_patch_embedding" in n for n in tensor_names):
            return "wan_vace"
        elif any("control_adapter.conv" in n for n in tensor_names):
            if any("img_emb.proj.0" in n for n in tensor_names):
                return "wan_camera"
            else:
                return "wan_camera_2.2"
        elif any("casual_audio_encoder" in n for n in tensor_names):
            return "wan_s2v"
        elif any("audio_proj.audio_proj_glob" in n for n in tensor_names):
            return "wan_humo"
        elif any("face_adapter.fuser_blocks" in n for n in tensor_names):
            return "wan_animate"
        elif any("img_emb.proj.0" in n for n in tensor_names):
            return "wan_i2v"
        else:
            return "wan_t2v"

    # Qwen Image model detection (same logic as ComfyUI's model_detection.py)
    if any("txt_norm.weight" in n for n in tensor_names):
        # It's a Qwen Image model
        if any("__index_timestep_zero__" in n for n in tensor_names):
            # Edit model (2511 - Nov 2025)
            return "qwen_image_edit_2511"
        elif any(
            "time_text_embed.addition_t_embedding.weight" in n for n in tensor_names
        ):
            # Layered model variant
            return "qwen_image_layered"
        else:
            # Standard Qwen Image (2512 - Dec 2025)
            return "qwen_image_2512"

    return None


# =============================================================================
# Sharded Model Loading
# =============================================================================


def detect_input_type(input_path: Path) -> Tuple[str, Path, Optional[Dict]]:
    """
    Detect whether input is a single file, index.json, or directory with shards.

    Args:
        input_path: Path to input file or directory

    Returns:
        Tuple of (input_type, base_path, index_data)
        - input_type: "single", "sharded_index", or "sharded_dir"
        - base_path: Directory containing the shard files (or the single file)
        - index_data: Parsed index.json data (or None for single file)
    """
    if input_path.is_dir():
        # Look for index.json in directory
        index_patterns = [
            "*.safetensors.index.json",
            "*model.safetensors.index.json",
            "model.safetensors.index.json",
        ]
        index_file = None
        for pattern in index_patterns:
            matches = list(input_path.glob(pattern))
            if matches:
                index_file = matches[0]
                break

        if index_file is None:
            # Check if there are multiple safetensors files
            st_files = list(input_path.glob("*.safetensors"))
            if len(st_files) > 1:
                raise ValueError(
                    f"Directory contains {len(st_files)} safetensors files but no index.json. "
                    "Please specify the index.json file directly."
                )
            elif len(st_files) == 1:
                return "single", st_files[0], None
            else:
                raise ValueError(f"No safetensors files found in {input_path}")

        with open(index_file, "r") as f:
            index_data = json.load(f)
        return "sharded_index", input_path, index_data

    elif input_path.suffix == ".json" or str(input_path).endswith(".index.json"):
        # Index JSON file
        with open(input_path, "r") as f:
            index_data = json.load(f)
        return "sharded_index", input_path.parent, index_data

    elif input_path.suffix == ".safetensors":
        return "single", input_path, None

    else:
        raise ValueError(f"Unsupported input type: {input_path}")


def get_shard_files(base_path: Path, index_data: Dict) -> Dict[str, Path]:
    """
    Get mapping of tensor names to their shard file paths.

    Args:
        base_path: Directory containing shard files
        index_data: Parsed index.json

    Returns:
        Dict mapping tensor names to file paths
    """
    weight_map = index_data.get("weight_map", {})
    tensor_to_file = {}

    for tensor_name, shard_filename in weight_map.items():
        shard_path = base_path / shard_filename
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")
        tensor_to_file[tensor_name] = shard_path

    return tensor_to_file


def load_sharded_tensors(
    base_path: Path,
    index_data: Dict,
    tensor_names: Optional[List[str]] = None,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str], List[str]]:
    """
    Load tensors from a sharded model.

    Args:
        base_path: Directory containing shard files
        index_data: Parsed index.json
        tensor_names: Optional list of specific tensors to load (None = all)
        device: Device to load tensors to
        verbose: Show progress

    Returns:
        Tuple of (tensors_dict, metadata, all_tensor_names)
    """
    weight_map = index_data.get("weight_map", {})
    metadata = index_data.get("metadata", {})

    # Get all tensor names if not specified
    all_tensor_names = list(weight_map.keys())
    if tensor_names is None:
        tensor_names = all_tensor_names

    # Group tensors by shard file for efficient loading
    shard_to_tensors: Dict[str, List[str]] = {}
    for name in tensor_names:
        if name not in weight_map:
            continue
        shard_file = weight_map[name]
        if shard_file not in shard_to_tensors:
            shard_to_tensors[shard_file] = []
        shard_to_tensors[shard_file].append(name)

    # Load tensors from each shard
    tensors = {}
    shard_files = list(shard_to_tensors.keys())

    iterator = tqdm(shard_files, desc="Loading shards", disable=not verbose)
    for shard_file in iterator:
        shard_path = base_path / shard_file
        iterator.set_postfix({"file": shard_file})

        with safe_open(shard_path, framework="pt", device=device) as f:
            for name in shard_to_tensors[shard_file]:
                tensors[name] = f.get_tensor(name)

    return tensors, metadata, all_tensor_names


def load_tensor_from_shards(
    tensor_name: str, base_path: Path, weight_map: Dict[str, str], device: str = "cpu"
) -> torch.Tensor:
    """
    Load a single tensor from sharded files.

    Args:
        tensor_name: Name of tensor to load
        base_path: Directory containing shard files
        weight_map: Mapping of tensor names to shard filenames
        device: Device to load to

    Returns:
        Loaded tensor
    """
    if tensor_name not in weight_map:
        raise KeyError(f"Tensor {tensor_name} not found in weight map")

    shard_file = weight_map[tensor_name]
    shard_path = base_path / shard_file

    with safe_open(shard_path, framework="pt", device=device) as f:
        return f.get_tensor(tensor_name)


# =============================================================================
# NVFP4 Quantization Functions
# =============================================================================


def roundup(x: int, multiple: int) -> int:
    """Round up x to the nearest multiple."""
    return ((x + multiple - 1) // multiple) * multiple


def ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def to_blocked(input_matrix: torch.Tensor, flatten: bool = True) -> torch.Tensor:
    """
    Rearrange a matrix into blocked layout for cuBLAS FP4 operations.

    See: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)
        flatten: If True, return flattened tensor

    Returns:
        Rearranged tensor in blocked layout
    """
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    if flatten:
        return rearranged.flatten()
    return rearranged.reshape(padded_rows, padded_cols)


def stochastic_float_to_fp4_e2m1(
    x: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """
    Convert float tensor to packed FP4 E2M1 format with stochastic rounding.

    FP4 E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
    Range: [-6.0, 6.0]

    Args:
        x: Input float tensor (values should be pre-scaled to FP4 range)
        generator: Random generator for stochastic rounding

    Returns:
        Packed uint8 tensor (2 FP4 values per byte)
    """
    orig_shape = x.shape
    sign = torch.signbit(x).to(torch.uint8)

    # Add stochastic noise before rounding
    exp = torch.floor(torch.log2(x.abs()) + 1.0).clamp(0, 3)
    x = (
        x
        + (
            torch.rand(
                x.size(),
                dtype=x.dtype,
                layout=x.layout,
                device=x.device,
                generator=generator,
            )
            - 0.5
        )
        * (2 ** (exp - 2.0))
        * 1.25
    )

    x = x.abs()
    exp = torch.floor(torch.log2(x) + 1.1925).clamp(0, 3)

    mantissa = (
        torch.where(exp > 0, (x / (2.0 ** (exp - 1)) - 1.0) * 2.0, (x * 2.0), out=x)
        .round()
        .to(torch.uint8)
    )
    del x

    exp = exp.to(torch.uint8)

    # Pack: sign(1) | exp(2) | mantissa(1) = 4 bits
    fp4 = (sign << 3) | (exp << 1) | mantissa
    del sign, exp, mantissa

    # Pack two FP4 values into one uint8
    fp4_flat = fp4.view(-1)
    packed = (fp4_flat[0::2] << 4) | fp4_flat[1::2]
    return packed.reshape(list(orig_shape)[:-1] + [-1])


def quantize_nvfp4_block(
    x: torch.Tensor, per_tensor_scale: torch.Tensor, generator: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor block to NVFP4 format.

    Args:
        x: Input tensor of shape (rows, cols)
        per_tensor_scale: Global scale factor
        generator: Random generator for stochastic rounding

    Returns:
        Tuple of (quantized_data, block_scales)
    """
    orig_shape = x.shape
    block_size = NVFP4_BLOCK_SIZE

    # Reshape to blocks
    x = x.reshape(orig_shape[0], -1, block_size)

    # Compute per-block scales
    block_amax = torch.amax(torch.abs(x), dim=-1)
    scaled_block_scales_fp8 = torch.clamp(
        (block_amax / F4_E2M1_MAX) / per_tensor_scale.to(x.dtype), max=F8_E4M3_MAX
    ).to(torch.float8_e4m3fn)

    # Scale the data
    x = x / (
        per_tensor_scale.to(x.dtype) * scaled_block_scales_fp8.to(x.dtype)
    ).unsqueeze(-1)
    x = x.view(orig_shape).nan_to_num()

    # Quantize to FP4
    data_fp4 = stochastic_float_to_fp4_e2m1(x, generator=generator)

    return data_fp4, scaled_block_scales_fp8


def quantize_nvfp4(
    weight: torch.Tensor, seed: int = 0, block_size: int = 4096 * 4096
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor to NVFP4 format.

    Args:
        weight: Input weight tensor of shape (out_features, in_features)
        seed: Seed for stochastic rounding (0 = deterministic)
        block_size: Processing block size for memory efficiency

    Returns:
        Tuple of (quantized_weight, block_scale, tensor_scale)
    """
    if weight.dim() != 2:
        raise ValueError(f"NVFP4 requires 2D tensor, got {weight.dim()}D")

    # Ensure float32 for accurate quantization
    x = weight.float()

    # Compute per-tensor scale
    tensor_scale = torch.amax(x.abs()) / (F8_E4M3_MAX * F4_E2M1_MAX)
    tensor_scale = tensor_scale.to(torch.float32)

    # Pad to multiple of 16
    orig_shape = x.shape
    rows, cols = orig_shape
    padded_rows = roundup(rows, 16)
    padded_cols = roundup(cols, 16)

    if padded_rows != rows or padded_cols != cols:
        x = torch.nn.functional.pad(x, (0, padded_cols - cols, 0, padded_rows - rows))

    padded_shape = x.shape

    # Prepare output tensors
    output_fp4 = torch.empty(
        padded_shape[0], padded_shape[1] // 2, dtype=torch.uint8, device=x.device
    )
    output_block = torch.empty(
        padded_shape[0],
        padded_shape[1] // NVFP4_BLOCK_SIZE,
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )

    # Initialize generator
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    # Process in blocks for memory efficiency
    num_slices = max(1, x.numel() // block_size)
    slice_size = max(1, round(x.shape[0] / num_slices))

    for i in range(0, x.shape[0], slice_size):
        fp4, block = quantize_nvfp4_block(
            x[i : i + slice_size], tensor_scale, generator=generator
        )
        output_fp4[i : i + slice_size].copy_(fp4)
        output_block[i : i + slice_size].copy_(block)

    # Convert block scales to blocked layout
    blocked_scales = to_blocked(output_block, flatten=False)

    return output_fp4, blocked_scales, tensor_scale


def quantize_fp8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor to FP8 E4M3 format with per-tensor scaling.

    Args:
        weight: Input weight tensor of shape (out_features, in_features)

    Returns:
        Tuple of (quantized_weight_fp8, scale)
    """
    # Compute per-tensor scale
    amax = torch.amax(weight.abs())
    scale = amax / F8_E4M3_MAX
    scale = torch.clamp(scale, min=1e-12).to(torch.float32)

    # Scale and convert to FP8
    scaled = weight.float() / scale
    fp8_weight = scaled.to(torch.float8_e4m3fn)

    return fp8_weight, scale


def try_use_comfy_kitchen() -> bool:
    """Try to import and use comfy_kitchen for optimized quantization."""
    try:
        import comfy_kitchen as ck

        return True
    except ImportError:
        return False


# =============================================================================
# Layer Classification
# =============================================================================


def is_linear_weight(name: str, tensor: torch.Tensor) -> bool:
    """
    Check if a tensor is a Linear layer weight.

    Args:
        name: Tensor name (should end with .weight)
        tensor: The tensor

    Returns:
        True if this is a Linear layer weight
    """
    if not name.endswith(".weight"):
        return False
    if tensor.dim() != 2:
        return False
    return True


def should_skip_layer_safe_mode(layer_name: str, weight_shape: Tuple[int, ...]) -> bool:
    """
    Check if a layer should be skipped in safe mode.

    Args:
        layer_name: Full layer name
        weight_shape: Shape of the weight tensor

    Returns:
        True if layer should be skipped
    """
    # Check against skip patterns
    for pattern in SAFE_MODE_SKIP_PATTERNS:
        if re.match(pattern, layer_name, re.IGNORECASE):
            return True

    # Skip small matrices
    if (
        weight_shape[0] < MIN_DIM_FOR_QUANTIZATION
        or weight_shape[1] < MIN_DIM_FOR_QUANTIZATION
    ):
        return True

    return False


def classify_layers(
    state_dict_keys: List[str],
    tensors: Dict[str, torch.Tensor],
    mode: str = "all",
    preset: Optional[str] = None,
    exclude_patterns: Optional[List[str]] = None,
    include_patterns: Optional[List[str]] = None,
    use_fp8: bool = False,
) -> Tuple[Set[str], Set[str], Set[str], Dict]:
    """
    Classify layers into quantize, fp8, and skip sets.

    Args:
        state_dict_keys: List of all tensor names
        tensors: Dict mapping names to tensors
        mode: "all" or "safe"
        preset: Optional preset name (e.g., "wan") for model-specific settings
        exclude_patterns: Additional patterns to exclude
        include_patterns: Patterns to force include
        use_fp8: If True, FP8-eligible layers go to fp8_layers; otherwise to skip_layers

    Returns:
        Tuple of (layers_to_quantize, layers_to_fp8, layers_to_skip, info_dict)
    """
    exclude_patterns = list(exclude_patterns or [])
    include_patterns = list(include_patterns or [])

    info = {
        "preset": preset,
        "preset_skip_patterns": [],
        "preset_fp8_patterns": [],
        "num_layers": 0,
        "model_type": None,
    }

    fp8_patterns: List[str] = []

    # If preset is specified, detect model info and add preset patterns
    if preset:
        num_layers = detect_num_layers(state_dict_keys)
        model_type = detect_model_type(state_dict_keys)
        info["num_layers"] = num_layers
        info["model_type"] = model_type

        if num_layers > 0:
            preset_skip_patterns, preset_fp8_patterns = get_preset_skip_patterns(
                preset, num_layers
            )
            info["preset_skip_patterns"] = preset_skip_patterns
            info["preset_fp8_patterns"] = preset_fp8_patterns
            # Prepend preset skip patterns to exclude patterns
            exclude_patterns = preset_skip_patterns + exclude_patterns
            # Store FP8 patterns separately
            fp8_patterns = preset_fp8_patterns
        else:
            print(f"Warning: Could not detect number of layers for preset '{preset}'")

    # Compile regex patterns
    exclude_re = [re.compile(p, re.IGNORECASE) for p in exclude_patterns]
    include_re = [re.compile(p, re.IGNORECASE) for p in include_patterns]
    fp8_re = [re.compile(p, re.IGNORECASE) for p in fp8_patterns]

    quantize_layers = set()
    fp8_layers = set()
    skip_layers = set()

    for name in state_dict_keys:
        if not name.endswith(".weight"):
            continue

        tensor = tensors.get(name)
        if tensor is None:
            continue

        layer_name = name[:-7]  # Remove ".weight"

        # Check if it's a Linear layer
        if not is_linear_weight(name, tensor):
            skip_layers.add(layer_name)
            continue

        # Check force include patterns first
        force_include = any(p.search(layer_name) for p in include_re)

        if force_include:
            quantize_layers.add(layer_name)
            continue

        # Check exclude patterns (must stay FP16)
        if any(p.search(layer_name) for p in exclude_re):
            skip_layers.add(layer_name)
            continue

        # Check FP8 patterns (can be FP8 or FP16 depending on --use-fp8)
        if any(p.search(layer_name) for p in fp8_re):
            if use_fp8:
                fp8_layers.add(layer_name)
            else:
                skip_layers.add(layer_name)
            continue

        # Safe mode checks
        if mode == "safe":
            if should_skip_layer_safe_mode(layer_name, tensor.shape):
                skip_layers.add(layer_name)
                continue

        quantize_layers.add(layer_name)

    return quantize_layers, fp8_layers, skip_layers, info


# =============================================================================
# Calibration
# =============================================================================


def calibrate_layer(
    weight: torch.Tensor,
    num_steps: int = 8,
    batch_size: int = 4,
    seq_len: int = 1024,
) -> torch.Tensor:
    """
    Basic calibration to estimate input_scale for a Linear layer.

    Uses random inputs with normal distribution to estimate
    the typical range of activations.

    Args:
        weight: Weight tensor of shape (out_features, in_features)
        num_steps: Number of calibration steps
        batch_size: Batch size for calibration
        seq_len: Sequence length for calibration

    Returns:
        input_scale tensor (scalar)
    """
    in_features = weight.shape[1]
    device = weight.device
    dtype = weight.dtype

    # Track maximum activation value
    amax = torch.tensor(0.0, device=device, dtype=torch.float32)

    for _ in range(num_steps):
        # Generate random input (normal distribution, similar to real activations)
        x = torch.randn(batch_size, seq_len, in_features, device=device, dtype=dtype)

        # Track amax
        amax = torch.maximum(amax, torch.amax(x.abs()))

    # Compute input scale
    input_scale = amax / F8_E4M3_MAX

    # Clamp to reasonable range
    input_scale = torch.clamp(input_scale, min=1e-12)

    return input_scale


# =============================================================================
# Main Conversion Logic
# =============================================================================


def convert_to_nvfp4(
    input_path: str,
    output_path: str,
    mode: str = "all",
    preset: Optional[str] = None,
    exclude_patterns: Optional[List[str]] = None,
    include_patterns: Optional[List[str]] = None,
    calibrate: bool = False,
    calibrate_steps: int = 8,
    seed: int = 0,
    device: str = "cuda",
    dtype: str = "bfloat16",
    dry_run: bool = False,
    verbose: bool = False,
    use_fp8: bool = False,
) -> Dict:
    """
    Convert a safetensors model to NVFP4 quantization.

    Supports both single safetensors files and sharded models.

    Args:
        input_path: Path to input safetensors file, index.json, or directory
        output_path: Path to output safetensors file
        mode: "all" or "safe"
        preset: Optional preset name for model-specific settings (e.g., "wan")
        exclude_patterns: Patterns for layers to skip
        include_patterns: Patterns to force include
        calibrate: Whether to run calibration
        calibrate_steps: Number of calibration steps
        seed: Random seed for stochastic rounding
        device: Device to use (cuda/cpu)
        dtype: Compute dtype (bfloat16/float16)
        dry_run: If True, only print what would be done
        verbose: Print detailed progress
        use_fp8: If True, use FP8 for intermediate precision layers (Q6_K equivalent)

    Returns:
        Dict with conversion statistics
    """
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Setup device and dtype
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    compute_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    # Detect input type (single file vs sharded)
    input_type, base_path, index_data = detect_input_type(input_path_obj)

    is_sharded = input_type in ("sharded_index", "sharded_dir")
    num_shards = 0

    if is_sharded and index_data is not None:
        print(f"Detected sharded model in: {base_path}")
        weight_map = index_data.get("weight_map", {})
        tensor_names = list(weight_map.keys())
        metadata = index_data.get("metadata", {})
        num_shards = len(set(weight_map.values()))
        print(f"Found {len(tensor_names)} tensors across {num_shards} shard files")
    else:
        print(f"Loading model from: {base_path}")
        # Single file - load metadata
        with safe_open(base_path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            tensor_names = list(f.keys())
        weight_map = None

    print(f"Total tensors: {len(tensor_names)}")
    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"Compute dtype: {dtype}")

    # First pass: load tensors to classify (we need shapes)
    # For sharded models, load only weight tensors for classification
    tensors = {}
    weight_tensor_names = [n for n in tensor_names if n.endswith(".weight")]

    print("Loading tensors for classification...")

    if is_sharded:
        # Load only weight tensors initially for classification
        tensors, _, _ = load_sharded_tensors(
            base_path,
            index_data,
            tensor_names=weight_tensor_names,
            device="cpu",
            verbose=verbose,
        )
    else:
        with safe_open(base_path, framework="pt", device="cpu") as f:
            for name in tqdm(tensor_names, desc="Loading", disable=not verbose):
                tensors[name] = f.get_tensor(name)

    # Classify layers
    quantize_layers, fp8_layers, skip_layers, classify_info = classify_layers(
        tensor_names,
        tensors,
        mode,
        preset,
        exclude_patterns or [],
        include_patterns or [],
        use_fp8,
    )

    # Print preset info if used
    if preset and classify_info.get("num_layers"):
        print(f"\n--- Preset: {preset} ---")
        print(f"Detected model type: {classify_info.get('model_type', 'unknown')}")
        print(f"Detected {classify_info['num_layers']} transformer blocks")
        preset_config = PRESETS.get(preset, {})
        skip_first = preset_config.get("skip_first_n_blocks", 0)
        skip_last = preset_config.get("skip_last_n_blocks", 0)
        if skip_first or skip_last:
            print(f"Skipping: first {skip_first} blocks, last {skip_last} blocks")
        if use_fp8 and classify_info.get("preset_fp8_patterns"):
            print(f"FP8 mode enabled: {len(fp8_layers)} layers will use FP8")

    print(f"\nLayers to quantize (NVFP4): {len(quantize_layers)}")
    if use_fp8:
        print(f"Layers to quantize (FP8): {len(fp8_layers)}")
    print(f"Layers to skip (FP16): {len(skip_layers)}")

    if verbose or dry_run:
        print("\n--- Layers to QUANTIZE (NVFP4) ---")
        for layer in sorted(quantize_layers):
            weight = tensors.get(f"{layer}.weight")
            if weight is not None:
                print(f"  {layer}: {tuple(weight.shape)}")

        if use_fp8 and fp8_layers:
            print("\n--- Layers to QUANTIZE (FP8) ---")
            for layer in sorted(fp8_layers):
                weight = tensors.get(f"{layer}.weight")
                if weight is not None:
                    print(f"  {layer}: {tuple(weight.shape)}")

        print("\n--- Layers to SKIP (FP16) ---")
        for layer in sorted(skip_layers):
            weight = tensors.get(f"{layer}.weight")
            if weight is not None:
                print(f"  {layer}: {tuple(weight.shape)} (Linear, skipped)")

    if dry_run:
        # Calculate estimated sizes - need to load all tensors for accurate size
        if is_sharded:
            # Load remaining tensors for size calculation
            remaining = [n for n in tensor_names if n not in tensors]
            if remaining:
                extra_tensors, _, _ = load_sharded_tensors(
                    base_path,
                    index_data,
                    tensor_names=remaining,
                    device="cpu",
                    verbose=verbose,
                )
                tensors.update(extra_tensors)

        original_size = sum(t.numel() * t.element_size() for t in tensors.values())
        quantized_size = 0
        for name, tensor in tensors.items():
            if name.endswith(".weight"):
                layer = name[:-7]
                if layer in quantize_layers:
                    # FP4: 0.5 bytes per element + block scales overhead (~6.25%)
                    quantized_size += tensor.numel() * 0.5 * 1.0625
                    # Plus tensor scale (4 bytes) and comfy_quant (~20 bytes)
                    quantized_size += 24
                elif layer in fp8_layers:
                    # FP8: 1 byte per element
                    quantized_size += tensor.numel() * 1
                else:
                    quantized_size += tensor.numel() * tensor.element_size()
            else:
                quantized_size += tensor.numel() * tensor.element_size()

        print(f"\n--- Estimated sizes ---")
        print(f"Original size: {original_size / 1e9:.2f} GB")
        print(f"Quantized size: {quantized_size / 1e9:.2f} GB")
        print(f"Compression ratio: {original_size / quantized_size:.2f}x")

        return {
            "quantized_layers": len(quantize_layers),
            "fp8_layers": len(fp8_layers),
            "skipped_layers": len(skip_layers),
            "original_size": original_size,
            "estimated_size": quantized_size,
        }

    # For sharded models in non-dry-run, load all tensors now
    if is_sharded:
        print("Loading all tensors from shards...")
        tensors, _, _ = load_sharded_tensors(
            base_path, index_data, device="cpu", verbose=verbose
        )

    # Second pass: quantize and build output
    output_tensors = {}
    stats = {
        "quantized_layers": 0,
        "fp8_layers": 0,
        "skipped_layers": 0,
        "calibrated_layers": 0,
        "original_bytes": 0,
        "quantized_bytes": 0,
    }

    # Track which tensors belong to quantized/fp8 layers
    processed_layer_tensors = set()
    for layer in quantize_layers:
        processed_layer_tensors.add(f"{layer}.weight")
        processed_layer_tensors.add(f"{layer}.bias")
    for layer in fp8_layers:
        processed_layer_tensors.add(f"{layer}.weight")
        processed_layer_tensors.add(f"{layer}.bias")

    print("\nQuantizing layers...")

    # Process NVFP4 quantized layers
    for layer in tqdm(sorted(quantize_layers), desc="Quantizing (NVFP4)"):
        weight_name = f"{layer}.weight"
        bias_name = f"{layer}.bias"

        weight = tensors[weight_name]
        stats["original_bytes"] += weight.numel() * weight.element_size()

        # Move to device for quantization
        weight_device = weight.to(device=device, dtype=torch.float32)

        # Quantize
        try:
            quantized_weight, block_scale, tensor_scale = quantize_nvfp4(
                weight_device, seed=seed
            )
        except Exception as e:
            print(f"\nWarning: Failed to quantize {layer}: {e}")
            print(f"  Keeping original weight")
            output_tensors[weight_name] = weight
            if bias_name in tensors:
                output_tensors[bias_name] = tensors[bias_name]
            stats["skipped_layers"] += 1
            continue

        # Store quantized tensors
        output_tensors[weight_name] = quantized_weight.cpu()
        output_tensors[f"{layer}.weight_scale"] = block_scale.view(torch.uint8).cpu()
        output_tensors[f"{layer}.weight_scale_2"] = tensor_scale.cpu()

        # Comfy quant config
        quant_config = {"format": "nvfp4"}
        config_bytes = json.dumps(quant_config).encode("utf-8")
        output_tensors[f"{layer}.comfy_quant"] = torch.tensor(
            list(config_bytes), dtype=torch.uint8
        )

        # Calibration
        if calibrate:
            input_scale = calibrate_layer(
                weight_device,
                num_steps=calibrate_steps,
            )
            output_tensors[f"{layer}.input_scale"] = input_scale.cpu()
            stats["calibrated_layers"] += 1

        # Copy bias if exists
        if bias_name in tensors:
            output_tensors[bias_name] = tensors[bias_name]

        stats["quantized_layers"] += 1
        stats["quantized_bytes"] += (
            quantized_weight.numel() * quantized_weight.element_size()
            + block_scale.numel() * block_scale.element_size()
            + tensor_scale.numel() * tensor_scale.element_size()
        )

        # Free GPU memory
        del weight_device
        if device == "cuda":
            torch.cuda.empty_cache()

    # Process FP8 layers
    if fp8_layers:
        for layer in tqdm(sorted(fp8_layers), desc="Quantizing (FP8)"):
            weight_name = f"{layer}.weight"
            bias_name = f"{layer}.bias"

            weight = tensors[weight_name]
            stats["original_bytes"] += weight.numel() * weight.element_size()

            # Move to device for quantization
            weight_device = weight.to(device=device, dtype=torch.float32)

            # Quantize to FP8
            try:
                fp8_weight, scale = quantize_fp8(weight_device)
            except Exception as e:
                print(f"\nWarning: Failed to FP8 quantize {layer}: {e}")
                print(f"  Keeping original weight")
                output_tensors[weight_name] = weight
                if bias_name in tensors:
                    output_tensors[bias_name] = tensors[bias_name]
                stats["skipped_layers"] += 1
                continue

            # Store FP8 quantized tensors
            output_tensors[weight_name] = fp8_weight.cpu()
            output_tensors[f"{layer}.weight_scale"] = scale.cpu()

            # Comfy quant config for FP8
            quant_config = {"format": "fp8_e4m3fn"}
            config_bytes = json.dumps(quant_config).encode("utf-8")
            output_tensors[f"{layer}.comfy_quant"] = torch.tensor(
                list(config_bytes), dtype=torch.uint8
            )

            # Copy bias if exists
            if bias_name in tensors:
                output_tensors[bias_name] = tensors[bias_name]

            stats["fp8_layers"] += 1
            stats["quantized_bytes"] += (
                fp8_weight.numel() * fp8_weight.element_size()
                + scale.numel() * scale.element_size()
            )

            # Free GPU memory
            del weight_device
            if device == "cuda":
                torch.cuda.empty_cache()

    # Copy non-quantized tensors
    print("Copying non-quantized tensors...")
    for name in tqdm(tensor_names, desc="Copying"):
        if name not in output_tensors and name not in processed_layer_tensors:
            output_tensors[name] = tensors[name]
            stats["original_bytes"] += (
                tensors[name].numel() * tensors[name].element_size()
            )
            stats["quantized_bytes"] += (
                tensors[name].numel() * tensors[name].element_size()
            )

    # Add skipped layer count
    stats["skipped_layers"] = len(skip_layers)

    # Save output
    print(f"\nSaving to: {output_path_obj}")

    # Prepare metadata - safetensors requires all values to be strings
    output_metadata = {}
    if isinstance(metadata, dict):
        for k, v in metadata.items():
            output_metadata[str(k)] = str(v) if v is not None else ""
    output_metadata["nvfp4_converter"] = "convert_nvfp4.py"
    output_metadata["nvfp4_mode"] = mode
    output_metadata["nvfp4_quantized_layers"] = str(stats["quantized_layers"])
    if preset:
        output_metadata["nvfp4_preset"] = preset
        if classify_info.get("num_layers"):
            output_metadata["nvfp4_num_layers"] = str(classify_info["num_layers"])
        if classify_info.get("model_type"):
            output_metadata["nvfp4_model_type"] = str(classify_info["model_type"])
    if is_sharded:
        output_metadata["nvfp4_source_shards"] = str(num_shards)
    if use_fp8 and stats["fp8_layers"] > 0:
        output_metadata["nvfp4_fp8_layers"] = str(stats["fp8_layers"])

    save_file(output_tensors, output_path_obj, metadata=output_metadata)

    # Print summary
    print("\n" + "=" * 50)
    print("Conversion complete!")
    print("=" * 50)
    if is_sharded:
        print(f"Consolidated {num_shards} shards into single file")
    print(f"Quantized layers (NVFP4): {stats['quantized_layers']}")
    if stats["fp8_layers"] > 0:
        print(f"Quantized layers (FP8): {stats['fp8_layers']}")
    print(f"Skipped layers (FP16): {stats['skipped_layers']}")
    if calibrate:
        print(f"Calibrated layers: {stats['calibrated_layers']}")
    print(f"Original size: {stats['original_bytes'] / 1e9:.2f} GB")
    print(f"Quantized size: {stats['quantized_bytes'] / 1e9:.2f} GB")
    if stats["quantized_bytes"] > 0:
        print(
            f"Compression ratio: {stats['original_bytes'] / stats['quantized_bytes']:.2f}x"
        )
    print(f"Output saved to: {output_path_obj}")

    return stats


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert safetensors models to NVFP4 quantization for ComfyUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single safetensors file
  python convert_nvfp4.py model.safetensors model_nvfp4.safetensors

  # Convert sharded model (via index.json)
  python convert_nvfp4.py model.safetensors.index.json model_nvfp4.safetensors

  # Convert sharded model (via directory containing index.json)
  python convert_nvfp4.py ./my_sharded_model/ model_nvfp4.safetensors

  # Use Wan preset for Wan 2.1/2.2 models (recommended for quality)
  python convert_nvfp4.py wan_model.safetensors wan_nvfp4.safetensors --preset wan

  # Use Qwen Image preset
  python convert_nvfp4.py qwen_image.safetensors qwen_nvfp4.safetensors --preset qwen-image

  # Use Qwen Image Edit preset
  python convert_nvfp4.py qwen_edit.safetensors qwen_edit_nvfp4.safetensors --preset qwen-image-edit

  # Use Unsloth-derived preset for Qwen Image Edit (best quality)
  python convert_nvfp4.py qwen_edit.safetensors qwen_edit_nvfp4.safetensors --preset unsloth-qwen-image-edit-2511

  # Use Unsloth-derived preset for Qwen Image 2512
  python convert_nvfp4.py qwen_image.safetensors qwen_nvfp4.safetensors --preset unsloth-qwen-image-2512

  # Use Unsloth-derived preset for any Wan model (best quality)
  python convert_nvfp4.py wan_model.safetensors wan_nvfp4.safetensors --preset unsloth-wan

  # Mixed precision: NVFP4 + FP8 for intermediate layers (better quality)
  python convert_nvfp4.py wan_model.safetensors wan_mixed.safetensors --preset unsloth-wan --use-fp8

  # Safe mode - skip sensitive layers  
  python convert_nvfp4.py model.safetensors model_nvfp4.safetensors --mode safe

  # Preview what would be quantized
  python convert_nvfp4.py model.safetensors output.safetensors --dry-run

  # Exclude specific patterns
  python convert_nvfp4.py model.safetensors output.safetensors --exclude ".*head.*" --exclude ".*proj_out.*"

  # With basic calibration
  python convert_nvfp4.py model.safetensors model_nvfp4.safetensors --calibrate

  # Smart preset - auto-detect architecture and apply Unsloth-like rules
  python convert_nvfp4.py any_model.safetensors any_nvfp4.safetensors --preset smart

  # Smart preset with FP8 for best quality on unknown models
  python convert_nvfp4.py any_model.safetensors any_mixed.safetensors --preset smart --use-fp8

Presets:
  smart           - (Recommended for unknown models) Auto-detects architecture and
                    applies Unsloth-like quantization rules. V projections and down
                    projections use FP8 (with --use-fp8) or skip, I/O layers stay FP16,
                    everything else becomes NVFP4. Works with any transformer model.

  wan             - Wan 2.1/2.2 video models. Skips first/last 2 transformer blocks
                    plus embeddings, head, and modulation layers for better quality.
                    Works for ALL Wan variants: T2V, I2V, TI2V, Camera, Vace, S2V,
                    Animate, HuMo, MagRef. Supports both 5B (32 blocks) and 14B models.

  qwen-image      - Qwen Image 2512 (Dec 2025). Skips first/last 3 transformer blocks
                    plus input/output projections and modulation layers.

  qwen-image-edit - Qwen Image Edit 2511 (Nov 2025). Same as qwen-image but
                    optimized for the edit model variant.

  unsloth-qwen-image-edit-2511
                  - Derived from Unsloth's Q4_K_M quantization. Skips V projections,
                    MLP down projections, first/last blocks, and I/O layers.
                    Based on analysis of which layers Unsloth kept at Q6_K+ precision.

  unsloth-qwen-image-2512
                  - Derived from Unsloth's Q5_K_S quantization. Skips first/last 2
                    blocks and I/O layers. Simpler pattern - more layers quantized.

  unsloth-wan     - Derived from Unsloth's Q5_K_M quantization for Wan models.
                    Skips V projections and FFN down projections across all blocks.
                    Works for ALL Wan: 2.1, 2.2, 5B, 14B, T2V, I2V, TI2V, MAGREF, etc.

Mixed Precision (--use-fp8):
  When using --use-fp8 with smart or Unsloth-derived presets, layers are quantized as:
    - NVFP4 (4-bit): Q, K, O projections, up projections, most layers
    - FP8 E4M3 (8-bit): V projections, down/gate projections (quality-critical)
    - FP16: I/O layers (embeddings, heads, modulators)
  
  This provides better quality than pure NVFP4 with moderate size increase.
  For Wan 14B, --use-fp8 adds ~5GB but preserves quality in sensitive layers.

Supported inputs:
  - Single .safetensors file
  - model.safetensors.index.json (sharded model index)
  - Directory containing index.json and shard files

Supported models:
  - Wan 2.1 / Wan 2.2 (all variants: T2V, I2V, TI2V, Camera, Vace, etc.)
  - Qwen Image 2512 / Qwen Image Edit 2511
  - Any model with Linear layers
        """,
    )

    parser.add_argument(
        "input", help="Input safetensors file, index.json, or directory with shards"
    )
    parser.add_argument(
        "output", help="Output safetensors file (single consolidated file)"
    )

    parser.add_argument(
        "--mode",
        choices=["all", "safe"],
        default="all",
        help="Quantization mode: 'all' quantizes all Linear layers, 'safe' skips sensitive layers (default: all)",
    )

    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=None,
        metavar="NAME",
        help="Use a model-specific preset (e.g., 'wan' for Wan 2.1/2.2 models). "
        "Presets configure optimal skip patterns for specific architectures.",
    )

    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Regex pattern for layers to skip (can specify multiple)",
    )

    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Regex pattern to force include layers (overrides --mode safe)",
    )

    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable basic calibration for input_scale",
    )

    parser.add_argument(
        "--calibrate-steps",
        type=int,
        default=8,
        help="Number of calibration steps (default: 8)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for stochastic rounding (default: 0 = deterministic)",
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for conversion (default: cuda)",
    )

    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Compute dtype (default: bfloat16)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed progress"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be quantized without converting",
    )

    parser.add_argument(
        "--use-fp8",
        action="store_true",
        help="Use FP8 for intermediate precision layers (Q6_K equivalent in GGUF). "
        "Creates mixed NVFP4/FP8 output for better quality with moderate size increase.",
    )

    args = parser.parse_args()

    try:
        convert_to_nvfp4(
            input_path=args.input,
            output_path=args.output,
            mode=args.mode,
            preset=args.preset,
            exclude_patterns=args.exclude,
            include_patterns=args.include,
            calibrate=args.calibrate,
            calibrate_steps=args.calibrate_steps,
            seed=args.seed,
            device=args.device,
            dtype=args.dtype,
            dry_run=args.dry_run,
            verbose=args.verbose,
            use_fp8=args.use_fp8,
        )
    except KeyboardInterrupt:
        print("\nConversion cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
