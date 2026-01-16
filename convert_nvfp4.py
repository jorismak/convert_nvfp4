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

    # Disable input_scale tensors (use ComfyUI dynamic quantization)
    python convert_nvfp4.py model.safetensors model_nvfp4.safetensors --no-input-scale

    # Use comfy_kitchen quantizer for exact NVFP4 packing/layout
    python convert_nvfp4.py model.safetensors model_nvfp4.safetensors --use-ck-quant

    # Fixed input_scale for activations (avoids clamping issues)
    python convert_nvfp4.py model.safetensors model_nvfp4.safetensors --input-scale-value 48.0

    # Copy input_scale from a reference NVFP4 model (same architecture)
    python convert_nvfp4.py model.safetensors model_nvfp4.safetensors --input-scale-from ref_nvfp4.safetensors

    # Calibrate input_scale from FP16/FP32 WAN model activations
    python convert_nvfp4.py model.safetensors model_nvfp4.safetensors --input-scale-from-fp16 wan_fp16.safetensors

    # Use per-layer input_scale from summary JSON (from analyze_input_scale_log.py)
    python convert_nvfp4.py model.safetensors model_nvfp4.safetensors --input-scale-summary-json nvfp4_scales_summary.json --input-scale-summary-percentile 99

    # Add margin to summary-derived input_scale
    python convert_nvfp4.py model.safetensors model_nvfp4.safetensors --input-scale-summary-json nvfp4_scales_summary.json --input-scale-summary-percentile 99 --input-scale-summary-multiplier 1.05

    # Print per-layer input_scale used
    python convert_nvfp4.py model.safetensors model_nvfp4.safetensors --input-scale-layer-summary
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
    # Tensors to keep in FP32 (not convert to BF16) - matches GGUF behavior
    "fp32_keep_patterns": [
        r"\.bias$",  # All biases - GGUF keeps these F32
    ],
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
    # Tensors to keep in FP32 (not convert to BF16) - matches GGUF behavior
    "fp32_keep_patterns": [
        r"\.bias$",  # All biases - GGUF keeps these F32
    ],
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
    # Tensors to keep in FP32 (not convert to BF16) - matches GGUF behavior
    "fp32_keep_patterns": [
        r"^patch_embedding\.weight$",  # Input Conv3d - GGUF keeps this F32
    ],
}

# =============================================================================
# Quant Test Preset - Matches GGUF Q4_0 quantization layout
# =============================================================================

# Test preset that exactly replicates GGUF Q4_0/Q4_1 quantization behavior
# for WAN models to match its quality output.
#
# GGUF Q4_0 analysis (WAN 2.2 TI2V 5B):
#   - F32 (1 tensor): patch_embedding.weight only
#   - F16 (524 tensors): All biases, norms, modulation, head, embeddings, time layers
#   - Q4_0 (270 tensors): All Q/K/V/O projections + FFN.0 (attention + FFN up)
#   - Q4_1 (30 tensors): FFN.2 down projections (slightly higher precision)
#
# NOTE: When used with --gguf, this preset will exactly mirror the GGUF
# precision choices (Q4_0/Q4_1 -> NVFP4, F16/BF16 -> FP16/BF16, F32 -> FP32).

QUANT_TEST_WAN_SKIP_PATTERNS = [
    # === Critical I/O layers - must stay at BF16 ===
    r"^head\.head\b",  # Output projection head
    r"^text_embedding\.",  # Text input embeddings
    r"^time_embedding\.",  # Time conditioning
    r"^time_projection\.",  # Time projection
    # === Normalization layers - stay at BF16 ===
    r"\.norm.*\.weight$",  # All norm layers (norm3, norm_k, norm_q)
    # === Modulation parameters - stay at BF16 ===
    r"\.modulation$",  # Adaptive norm modulation
    # NOTE: V projections and FFN.2 are NOT skipped - GGUF Q4_0 quantizes them to Q4_0/Q4_1
    # We quantize them to NVFP4 to match GGUF exactly
]

QUANT_TEST_WAN_CONFIG = {
    "name": "quant-test",
    "type": "static",
    "description": (
        "Exact GGUF Q4_0/Q4_1 match for WAN models. "
        "Quantizes Q/K/V/O projections, FFN.0, and FFN.2 to NVFP4. "
        "Keeps only head/embeddings/norms/time at BF16, patch_embedding at F32."
    ),
    "skip_patterns": QUANT_TEST_WAN_SKIP_PATTERNS,
    "fp8_patterns": [],  # No FP8 - everything is either NVFP4 or BF16
    "fp32_keep_patterns": [r"^patch_embedding\.weight$"],
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
    "quant-test": QUANT_TEST_WAN_CONFIG,
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
# Activation Calibration (WAN models)
# =============================================================================


class _ActivationCollector:
    """Collects input activation statistics for linear layers."""

    def __init__(self):
        self.amax_values: Dict[str, List[float]] = {}
        self.hooks = []

    def make_hook(self, name):
        def hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            if x is not None and isinstance(x, torch.Tensor):
                amax = x.abs().amax().item()
                self.amax_values.setdefault(name, []).append(amax)

        return hook

    def register_hooks(self, model, layer_names: Set[str]):
        import torch.nn as nn

        for name, module in model.named_modules():
            if name in layer_names and isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self.make_hook(name))
                self.hooks.append(hook)
        print(f"Registered {len(self.hooks)} activation hooks")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_input_scales(self, method: str = "max") -> Dict[str, float]:
        input_scales = {}
        for name, amaxes in self.amax_values.items():
            if not amaxes:
                continue
            if method == "max":
                amax = max(amaxes)
            elif method == "mean":
                amax = sum(amaxes) / len(amaxes)
            elif method == "percentile_99":
                sorted_amaxes = sorted(amaxes)
                idx = int(len(sorted_amaxes) * 0.99)
                amax = sorted_amaxes[min(idx, len(sorted_amaxes) - 1)]
            else:
                raise ValueError(f"Unknown input_scale method: {method}")

            # NVFP4 input quantization scale
            scale = amax / (F8_E4M3_MAX * F4_E2M1_MAX)
            input_scales[name] = max(scale, 1e-12)

        return input_scales


def _resolve_comfyui_root(user_root: Optional[str]) -> Path:
    if user_root:
        return Path(user_root)
    # Default to parent of this repo (nvfp4-conv -> ComfyUI)
    return Path(__file__).resolve().parents[1]


def _load_state_dict_for_calibration(model_path: str) -> Dict[str, torch.Tensor]:
    path = Path(model_path)
    input_type, base_path, index_data = detect_input_type(path)
    if input_type in ("sharded_index", "sharded_dir") and index_data is not None:
        tensors, _, _ = load_sharded_tensors(base_path, index_data, device="cpu")
        return tensors

    if path.suffix == ".safetensors":
        state_dict = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict

    return torch.load(path, map_location="cpu")


def _build_wan_model_for_calibration(
    model_path: str, device: str, comfyui_root: Optional[str]
):
    import sys

    root = _resolve_comfyui_root(comfyui_root)
    sys.path.insert(0, str(root))

    from comfy.ldm.wan.model import WanModel
    import comfy.ops as ops

    state_dict = _load_state_dict_for_calibration(model_path)

    hidden_size = state_dict["blocks.0.self_attn.q.weight"].shape[0]
    num_blocks = (
        max(int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("blocks."))
        + 1
    )
    text_dim = state_dict["text_embedding.0.weight"].shape[1]
    ffn_dim = state_dict["blocks.0.ffn.0.weight"].shape[0]
    in_dim = state_dict["patch_embedding.weight"].shape[1]
    head_out = state_dict["head.head.weight"].shape[0]
    out_dim = head_out // 4

    model = WanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        in_dim=in_dim,
        dim=hidden_size,
        ffn_dim=ffn_dim,
        text_dim=text_dim,
        out_dim=out_dim,
        num_heads=hidden_size // 128,
        num_layers=num_blocks,
        dtype=torch.bfloat16,
        device="cpu",
        operations=ops.manual_cast,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys during calibration load: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys during calibration load: {len(unexpected)}")

    model = model.to(device=device)
    model.eval()
    return model


def _run_wan_calibration_passes(
    model,
    in_dim: int,
    num_samples: int,
    device: str,
    batch_size: int = 1,
):
    text_dim = model.text_dim
    text_len = 512

    latent_channels = in_dim
    latent_t = 5
    latent_h = 30
    latent_w = 52

    for i in range(num_samples):
        x = torch.randn(
            batch_size,
            latent_channels,
            latent_t,
            latent_h,
            latent_w,
            device=device,
            dtype=torch.bfloat16,
        )
        timestep = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)
        context = torch.randn(
            batch_size, text_len, text_dim, device=device, dtype=torch.bfloat16
        )

        with torch.no_grad():
            try:
                _ = model(x, timestep, context)
            except Exception as e:
                print(f"Calibration forward error: {e}")
                break

        if i % 2 == 0 and device == "cuda":
            torch.cuda.empty_cache()


def compute_input_scales_from_fp16_model(
    fp16_model_path: str,
    layer_names: Set[str],
    method: str,
    num_samples: int,
    device: str,
    comfyui_root: Optional[str],
) -> Dict[str, torch.Tensor]:
    print("Computing input_scale from FP16/FP32 WAN model activations...")
    model = _build_wan_model_for_calibration(fp16_model_path, device, comfyui_root)

    collector = _ActivationCollector()
    collector.register_hooks(model, layer_names)

    in_dim = model.patch_embedding.weight.shape[1]
    _run_wan_calibration_passes(model, in_dim, num_samples, device)

    collector.remove_hooks()
    input_scales = collector.get_input_scales(method=method)

    if not input_scales:
        print("Warning: No activation scales collected from FP16 model.")
        return {}

    scale_values = list(input_scales.values())
    print(
        f"input_scale range: {min(scale_values):.6f} - {max(scale_values):.6f} "
        f"(mean {sum(scale_values) / len(scale_values):.6f})"
    )

    return {k: torch.tensor([v], dtype=torch.float32) for k, v in input_scales.items()}


# =============================================================================
# GGUF Precision Parity Helpers (for quant-test)
# =============================================================================


def load_gguf_tensor_types(gguf_path: str) -> Dict[str, str]:
    """
    Load GGUF tensor types by name.

    Returns:
        Dict mapping tensor name -> GGUF tensor type name (e.g., Q4_0, Q4_1, F16, F32)
    """
    try:
        import gguf  # type: ignore
    except Exception as e:
        raise ImportError(
            "The 'gguf' package is required for --gguf precision parity. "
            "Install it in your environment and retry."
        ) from e

    reader = gguf.GGUFReader(gguf_path)
    gguf_types: Dict[str, str] = {}
    for tensor in reader.tensors:
        tensor_type = tensor.tensor_type
        type_name = tensor_type.name if hasattr(tensor_type, "name") else str(tensor_type)
        gguf_types[tensor.name] = type_name
    return gguf_types


def classify_layers_from_gguf(
    tensor_names: List[str],
    tensors: Dict[str, torch.Tensor],
    gguf_types: Dict[str, str],
    verbose: bool = False,
) -> Tuple[Set[str], Set[str], Set[str], Dict, Set[str]]:
    """
    Classify layers using exact GGUF precision choices.

    Returns:
        (quantize_layers, fp8_layers, skip_layers, info_dict, fp32_keep_names)
    """
    q4_types = {"Q4_0", "Q4_1"}
    f16_types = {"F16", "BF16"}
    f32_types = {"F32"}

    quantize_layers: Set[str] = set()
    fp8_layers: Set[str] = set()
    skip_layers: Set[str] = set()
    fp32_keep_names: Set[str] = set()

    missing_in_gguf = []
    q4_non_linear = []
    unknown_types = []

    # Track FP32 tensors by exact name (can include non-weights)
    for name, ttype in gguf_types.items():
        if ttype in f32_types:
            fp32_keep_names.add(name)
        elif ttype not in q4_types and ttype not in f16_types:
            unknown_types.append((name, ttype))

    # Classify only weight tensors for quantization/skip
    for name in tensor_names:
        if not name.endswith(".weight"):
            continue

        if name not in gguf_types:
            missing_in_gguf.append(name)
            continue

        ttype = gguf_types[name]
        if ttype in q4_types:
            tensor = tensors.get(name)
            if tensor is not None and tensor.dim() == 2:
                layer = name[: -len(".weight")]
                quantize_layers.add(layer)
            else:
                q4_non_linear.append(name)
        elif ttype in f16_types or ttype in f32_types:
            tensor = tensors.get(name)
            if tensor is not None and tensor.dim() == 2:
                layer = name[: -len(".weight")]
                skip_layers.add(layer)

    info = {
        "preset": "quant-test",
        "model_type": None,
        "num_layers": 0,
        "gguf_q4_layers": len(quantize_layers),
        "gguf_f16_layers": len(skip_layers),
        "gguf_f32_tensors": len(fp32_keep_names),
        "gguf_missing_in_source": len(missing_in_gguf),
        "gguf_q4_non_linear": len(q4_non_linear),
        "gguf_unknown_types": len(unknown_types),
    }

    if verbose:
        if missing_in_gguf:
            print(f"[GGUF] Missing in GGUF (weights): {len(missing_in_gguf)}")
        if q4_non_linear:
            print(f"[GGUF] Q4 tensors not 2D weights: {len(q4_non_linear)}")
        if unknown_types:
            print(f"[GGUF] Unknown GGUF tensor types: {len(unknown_types)}")

    return quantize_layers, fp8_layers, skip_layers, info, fp32_keep_names


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


def _n_ones(n: int) -> int:
    """Return a number with n bits set to 1."""
    return (1 << n) - 1


# FP32 constants for conversion
EBITS_F32, MBITS_F32 = 8, 23
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)  # 127


def _f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """Convert FP32 numbers to sub-byte floating point numbers.

    This is the correct IEEE-754 style conversion with proper rounding,
    derived from PyTorch AO (comfy_kitchen).

    Args:
        x: Input tensor of dtype torch.float32
        ebits: Number of exponent bits (2 for FP4 E2M1)
        mbits: Number of mantissa bits (1 for FP4 E2M1)

    Returns:
        torch.Tensor of dtype torch.uint8, where the bit encoding is stored
        in the least significant bits.
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    # Calculate constants
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # All E bits and M bits are 1s
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))

    # E bits = 1, M bits = 0
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (
        # Exp bias conversion between formats
        (F32_EXP_BIAS - exp_bias)
        # Mantissa length difference between formats
        + (MBITS_F32 - mbits)
        # Add one to encoded exponent for denormalized numbers
        + 1
    )
    denorm_mask_int = denorm_exp << MBITS_F32

    # Reinterpret int32 as float32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(
        torch.float32
    )

    # Save the sign
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # Set everything to positive, will add sign back at the end
    x = x ^ sign

    # Convert back to float for comparisons
    x = x.view(torch.float)

    # Rewrite saturate/denorm/norm branches without explicit data dependent
    # control flow, to be more compiler friendly
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    #
    # Branch 1: saturate to max val - handled later in combining branches
    #

    #
    # Branch 2: conversion to denormal as well as rounding up to normal
    #
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    #
    # Branch 3: stay in normal range, adjust the exponent and round
    #
    normal_x = x.view(torch.int32)
    # Resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    # Update exponent, rounding bias part 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    # Rounding bias part 2
    normal_x += mant_odd
    # Take the bits!
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    #
    # Combine the branches
    #
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # Add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Mask out any filled bits from signed right shift
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


def pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit values into one uint8 byte."""
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    # Pack: first value in high nibble, second in low nibble
    return (uint8_data[::2] << 4 | uint8_data[1::2]).view(*shape[:-1], shape[-1] // 2)


def float_to_fp4_e2m1_packed(x: torch.Tensor) -> torch.Tensor:
    """
    Convert float tensor to packed FP4 E2M1 format with correct IEEE-754 rounding.

    FP4 E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
    Range: [-6.0, 6.0]

    Args:
        x: Input float tensor (values should be pre-scaled to FP4 range)

    Returns:
        Packed uint8 tensor (2 FP4 values per byte)
    """
    # Convert to float32 for the conversion algorithm
    x_f32 = x.to(torch.float32)

    # Convert float32 to unpacked FP4 codes (one code per byte)
    fp4_unpacked = _f32_to_floatx_unpacked(x_f32, ebits=2, mbits=1)

    # Pack two FP4 values into one byte
    return pack_uint4(fp4_unpacked)


def quantize_nvfp4_block(
    x: torch.Tensor, per_tensor_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor block to NVFP4 format.

    Args:
        x: Input tensor of shape (rows, cols)
        per_tensor_scale: Global scale factor

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

    # Quantize to FP4 using correct IEEE-754 rounding
    data_fp4 = float_to_fp4_e2m1_packed(x)

    return data_fp4, scaled_block_scales_fp8


def quantize_nvfp4(
    weight: torch.Tensor, block_size: int = 4096 * 4096
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor to NVFP4 format.

    Args:
        weight: Input weight tensor of shape (out_features, in_features)
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

    # Process in blocks for memory efficiency
    num_slices = max(1, x.numel() // block_size)
    slice_size = max(1, round(x.shape[0] / num_slices))

    for i in range(0, x.shape[0], slice_size):
        fp4, block = quantize_nvfp4_block(x[i : i + slice_size], tensor_scale)
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

    # Compute input scale for NVFP4 quantization
    # When inputs are quantized to NVFP4, we need to use the combined scale formula
    input_scale = amax / (F8_E4M3_MAX * F4_E2M1_MAX)

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
    gguf_path: Optional[str] = None,
    add_input_scale: bool = True,
    input_scale_value: Optional[float] = None,
    input_scale_from: Optional[str] = None,
    input_scale_from_fp16: Optional[str] = None,
    input_scale_method: str = "max",
    input_scale_samples: int = 8,
    comfyui_root: Optional[str] = None,
    input_scale_summary_json: Optional[str] = None,
    input_scale_summary_percentile: int = 99,
    input_scale_summary_multiplier: float = 1.0,
    input_scale_layer_summary: bool = False,
    use_ck_quant: bool = False,
    calibrate: bool = False,
    calibrate_steps: int = 8,
    device: str = "cuda",
    dtype: str = "bfloat16",
    quant_dtype: str = "bfloat16",
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
        gguf_path: Optional path to GGUF for exact precision parity (quant-test)
        add_input_scale: Whether to write input_scale tensors for NVFP4 layers
        input_scale_value: Fixed input_scale value for activations (if not calibrating)
        input_scale_from: Optional NVFP4 safetensors file to copy input_scale values
        input_scale_from_fp16: Optional FP16/FP32 WAN model to measure activations
        input_scale_method: Method for input_scale aggregation (max/mean/percentile_99)
        input_scale_samples: Number of activation samples to run
        comfyui_root: Optional ComfyUI root path for WAN model loading
        input_scale_summary_json: Optional summary JSON from analyze_input_scale_log.py
        input_scale_summary_percentile: Percentile to use from summary JSON
        input_scale_summary_multiplier: Multiplier applied to summary scales
        input_scale_layer_summary: Print per-layer input_scale values after conversion
        use_ck_quant: Use comfy_kitchen quantize_nvfp4 for backend-compatible packing
        calibrate: Whether to run calibration
        calibrate_steps: Number of calibration steps
        device: Device to use (cuda/cpu)
        dtype: Output dtype for non-quantized tensors (bfloat16/float16)
        quant_dtype: Input dtype for quantization (bfloat16/float16/float32).
                     Using bfloat16 matches working HuggingFace models.
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

    if preset == "quant-test" and not gguf_path:
        raise ValueError(
            "Preset 'quant-test' requires --gguf to mirror GGUF precision exactly."
        )
    if gguf_path and preset != "quant-test":
        print("Warning: --gguf is only used with preset 'quant-test'.")
    if calibrate and not add_input_scale:
        print("Warning: --calibrate ignored because --no-input-scale was set.")
    if input_scale_value is not None and not add_input_scale:
        print("Warning: --input-scale-value ignored because --no-input-scale was set.")
    if input_scale_from and not add_input_scale:
        print("Warning: --input-scale-from ignored because --no-input-scale was set.")
    if input_scale_from_fp16 and not add_input_scale:
        print("Warning: --input-scale-from-fp16 ignored because --no-input-scale was set.")
    if input_scale_summary_json and not add_input_scale:
        print("Warning: --input-scale-summary-json ignored because --no-input-scale was set.")
    if input_scale_summary_multiplier <= 0:
        raise ValueError("--input-scale-summary-multiplier must be > 0")

    input_scale_map: Dict[str, torch.Tensor] = {}
    input_scale_source: Dict[str, str] = {}

    ck = None
    if use_ck_quant:
        try:
            import comfy_kitchen as ck  # type: ignore
        except Exception:
            print("Warning: comfy_kitchen not available; falling back to local quantizer.")
            use_ck_quant = False

    # Setup device and dtype
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    compute_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    # Determine quantization input dtype
    if quant_dtype == "bfloat16":
        quantize_input_dtype = torch.bfloat16
    elif quant_dtype == "float16":
        quantize_input_dtype = torch.float16
    else:
        quantize_input_dtype = torch.float32

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
    print(f"Output dtype: {dtype}")
    print(f"Quantization input dtype: {quant_dtype}")

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
    fp32_keep_names: Set[str] = set()
    if preset == "quant-test" and gguf_path:
        gguf_types = load_gguf_tensor_types(gguf_path)
        quantize_layers, fp8_layers, skip_layers, classify_info, fp32_keep_names = (
            classify_layers_from_gguf(tensor_names, tensors, gguf_types, verbose)
        )
        print(
            f"GGUF parity mode: Q4_0/Q4_1 -> NVFP4, F16/BF16 -> {dtype}, F32 -> FP32"
        )
    else:
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

    if add_input_scale and input_scale_from_fp16:
        input_scale_map = compute_input_scales_from_fp16_model(
            input_scale_from_fp16,
            set(quantize_layers),
            input_scale_method,
            input_scale_samples,
            device,
            comfyui_root,
        )

    if add_input_scale and input_scale_summary_json:
        summary_path = Path(input_scale_summary_json)
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Input scale summary not found: {input_scale_summary_json}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        percentile_key = f"scale_p{input_scale_summary_percentile}"
        summary_layers = summary.get("layers", [])
        summary_map: Dict[str, torch.Tensor] = {}
        for row in summary_layers:
            layer = row.get("layer")
            if not layer:
                continue
            if percentile_key in row:
                val = float(row[percentile_key])
            elif "scale_max" in row:
                val = float(row["scale_max"])
            elif "scale_mean" in row:
                val = float(row["scale_mean"])
            else:
                continue
            val *= input_scale_summary_multiplier
            summary_map[layer] = torch.tensor([val], dtype=torch.float32)
        input_scale_map = summary_map
        if input_scale_map:
            vals = [float(v.item()) for v in input_scale_map.values()]
            print(
                f"Input scale summary ({percentile_key}, x{input_scale_summary_multiplier}): "
                f"count={len(vals)}, min={min(vals):.6f}, max={max(vals):.6f}, "
                f"mean={(sum(vals)/len(vals)):.6f}"
            )
            input_scale_source = {k: "summary" for k in input_scale_map.keys()}

    if add_input_scale and input_scale_from:
        input_scale_path = Path(input_scale_from)
        if not input_scale_path.exists():
            raise FileNotFoundError(f"Input scale source not found: {input_scale_from}")
        with safe_open(input_scale_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name.endswith(".input_scale"):
                    base = name[: -len(".input_scale")]
                    input_scale_map[base] = f.get_tensor(name).to(torch.float32)
                    input_scale_source[base] = "file"

    if add_input_scale and input_scale_from_fp16:
        for k in input_scale_map.keys():
            if k not in input_scale_source:
                input_scale_source[k] = "fp16"

    if add_input_scale and input_scale_map:
        missing = set(quantize_layers) - set(input_scale_map.keys())
        if missing:
            print(
                f"Warning: input_scale missing for {len(missing)} layers; using fallback." 
                f"Example: {sorted(missing)[:3]}"
            )

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
    quantized_layer_configs = {}  # For _quantization_metadata
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
    for layer in fp8_layers:
        processed_layer_tensors.add(f"{layer}.weight")

    print("\nQuantizing layers...")

    # Process NVFP4 quantized layers
    for layer in tqdm(sorted(quantize_layers), desc="Quantizing (NVFP4)"):
        weight_name = f"{layer}.weight"
        bias_name = f"{layer}.bias"

        weight = tensors[weight_name]
        stats["original_bytes"] += weight.numel() * weight.element_size()

        # Move to device and convert to quantization input dtype
        # Using BF16/FP16 matches the behavior of working NVFP4 models from HuggingFace
        # The FP32->BF16 conversion loses some precision but matches what the CUDA
        # quantization backend expects and produces compatible results
        weight_device = weight.to(device=device, dtype=quantize_input_dtype)

        # Quantize (internally converts to FP32 for precise quantization math)
        try:
            if use_ck_quant and ck is not None:
                # Compute per-tensor scale in FP32 for accuracy
                tensor_scale = torch.amax(weight.float().abs()) / (
                    F8_E4M3_MAX * F4_E2M1_MAX
                )
                tensor_scale = tensor_scale.to(torch.float32)

                # comfy_kitchen handles padding and blocked scale layout internally
                quantized_weight, block_scale = ck.quantize_nvfp4(
                    weight_device, tensor_scale, pad_16x=True
                )
            else:
                quantized_weight, block_scale, tensor_scale = quantize_nvfp4(weight_device)
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
        # IMPORTANT: Keep block_scale as float8_e4m3fn, NOT uint8!
        # ComfyUI expects float8_e4m3fn dtype for proper dequantization
        output_tensors[f"{layer}.weight_scale"] = block_scale.cpu()
        output_tensors[f"{layer}.weight_scale_2"] = tensor_scale.cpu()

        # Track layer for _quantization_metadata
        quantized_layer_configs[layer] = {"format": "nvfp4"}

        # Optional: add input_scale for NVFP4 layers
        if add_input_scale:
            # Without it, ComfyUI may use dynamic quantization (can be unstable)
            if calibrate:
                # Use calibration for accurate input_scale (measures actual activation range)
                input_scale = calibrate_layer(
                    weight_device,
                    num_steps=calibrate_steps,
                )
                output_tensors[f"{layer}.input_scale"] = input_scale.cpu()
                stats["calibrated_layers"] += 1
            else:
                if layer in input_scale_map:
                    output_tensors[f"{layer}.input_scale"] = input_scale_map[layer]
                if input_scale_value is None:
                    # Fallback heuristic (kept for compatibility)
                    activation_amax_estimate = 10.0
                    input_scale_value = activation_amax_estimate / (
                        F8_E4M3_MAX * F4_E2M1_MAX
                    )
                input_scale = torch.tensor(
                    [input_scale_value], dtype=torch.float32, device="cpu"
                )
                output_tensors[f"{layer}.input_scale"] = input_scale

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

            # Move to device and convert to quantization input dtype
            weight_device = weight.to(device=device, dtype=quantize_input_dtype)

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

            # Track layer for _quantization_metadata
            quantized_layer_configs[layer] = {"format": "float8_e4m3fn"}

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

    # Copy non-quantized tensors, converting FP32 to BF16/FP16 where appropriate
    # Get FP32 keep patterns from preset (if any)
    fp32_keep_patterns = []
    if preset:
        preset_config = PRESETS.get(preset, {})
        fp32_keep_patterns = preset_config.get("fp32_keep_patterns", [])

    def should_keep_fp32(tensor_name: str) -> bool:
        """Check if tensor should stay in FP32 based on preset patterns or GGUF parity."""
        if tensor_name in fp32_keep_names:
            return True
        for pattern in fp32_keep_patterns:
            if re.search(pattern, tensor_name):
                return True
        return False

    print("Copying non-quantized tensors...")
    fp32_converted = 0
    fp32_kept = 0
    for name in tqdm(tensor_names, desc="Copying"):
        if name not in output_tensors and name not in processed_layer_tensors:
            tensor = tensors[name]
            original_size = tensor.numel() * tensor.element_size()
            stats["original_bytes"] += original_size

            # Convert FP32 tensors to target dtype (BF16/FP16) unless they should stay FP32
            if tensor.dtype == torch.float32 and not should_keep_fp32(name):
                tensor = tensor.to(dtype=compute_dtype)
                fp32_converted += 1
            elif tensor.dtype == torch.float32:
                fp32_kept += 1

            output_tensors[name] = tensor
            stats["quantized_bytes"] += tensor.numel() * tensor.element_size()

    if fp32_converted > 0 or fp32_kept > 0:
        print(f"  FP32 tensors converted to {compute_dtype}: {fp32_converted}")
        print(f"  FP32 tensors kept as FP32: {fp32_kept}")

    # Add skipped layer count
    stats["skipped_layers"] = len(skip_layers)

    # Save output
    print(f"\nSaving to: {output_path_obj}")

    # Prepare metadata - safetensors requires all values to be strings
    output_metadata = {}
    if isinstance(metadata, dict):
        for k, v in metadata.items():
            output_metadata[str(k)] = str(v) if v is not None else ""

    # Add _quantization_metadata - this is required for ComfyUI to recognize quantized layers
    quantization_metadata = {
        "format_version": "1.0",
        "layers": quantized_layer_configs,
    }
    output_metadata["_quantization_metadata"] = json.dumps(quantization_metadata)

    # Add our converter metadata
    output_metadata["nvfp4_converter"] = "convert_nvfp4.py"
    output_metadata["nvfp4_mode"] = mode
    output_metadata["nvfp4_quant_dtype"] = quant_dtype  # Track quantization input dtype
    output_metadata["nvfp4_quantized_layers"] = str(stats["quantized_layers"])
    if gguf_path:
        output_metadata["nvfp4_gguf"] = str(gguf_path)
    if input_scale_from:
        output_metadata["nvfp4_input_scale_from"] = str(input_scale_from)
    if input_scale_from_fp16:
        output_metadata["nvfp4_input_scale_from_fp16"] = str(input_scale_from_fp16)
        output_metadata["nvfp4_input_scale_method"] = str(input_scale_method)
        output_metadata["nvfp4_input_scale_samples"] = str(input_scale_samples)
    if input_scale_summary_json:
        output_metadata["nvfp4_input_scale_summary_json"] = str(input_scale_summary_json)
        output_metadata["nvfp4_input_scale_summary_percentile"] = str(
            input_scale_summary_percentile
        )
        output_metadata["nvfp4_input_scale_summary_multiplier"] = str(
            input_scale_summary_multiplier
        )
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

    if input_scale_layer_summary and add_input_scale:
        print("\n--- Input scale per layer ---")
        for layer in sorted(quantize_layers):
            key = f"{layer}.input_scale"
            if key in output_tensors:
                try:
                    val = float(output_tensors[key].item())
                except Exception:
                    val = float(output_tensors[key].mean().item())
                print(f"  {layer}: {val:.6f}")

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

    # Exact GGUF parity for Wan 2.2 TI2V (Q4_0/Q4_1 -> NVFP4, F16 -> BF16, F32 -> FP32)
    python convert_nvfp4.py \
            "D:\\comfy2\\ComfyUI\\nvfp4-conv\\wan2.2-ti2v-5b\\diffusion_pytorch_model.safetensors.index.json" \
            "D:\\ComfyUI\\ComfyUI\\models\\diffusion_models\\wan2.2-ti2v-5b-nvfp4-quant-test.safetensors" \
            --preset quant-test \
            --gguf "D:\\ComfyUI\\ComfyUI\\models\\diffusion_models\\Wan2.2-TI2V-5B-Q4_0.gguf"

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

  quant-test      - Exact GGUF Q4_0 match for WAN models.
                    Quantizes Q/K/V/O projections, FFN.0, and FFN.2 to NVFP4.
                    Keeps only head/embeddings/norms/time at BF16, patch_embedding at F32.
                    Use this to exactly replicate GGUF Q4_0 quantization layout.
                    NOTE: Requires --gguf to mirror GGUF precision exactly.

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
        "--gguf",
        default=None,
        metavar="PATH",
        help="Path to GGUF file for exact precision parity (required for 'quant-test').",
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
        "--no-input-scale",
        action="store_true",
        help="Do not write input_scale tensors for NVFP4 layers",
    )

    parser.add_argument(
        "--input-scale-value",
        type=float,
        default=None,
        help="Fixed input_scale value for activations (used when not calibrating)",
    )

    parser.add_argument(
        "--input-scale-from",
        default=None,
        metavar="PATH",
        help="Copy input_scale tensors from a reference NVFP4 safetensors file",
    )

    parser.add_argument(
        "--input-scale-from-fp16",
        default=None,
        metavar="PATH",
        help="Calibrate input_scale from FP16/FP32 WAN model activations",
    )

    parser.add_argument(
        "--input-scale-method",
        choices=["max", "mean", "percentile_99"],
        default="max",
        help="Method for input_scale aggregation (default: max)",
    )

    parser.add_argument(
        "--input-scale-samples",
        type=int,
        default=8,
        help="Number of activation samples for input_scale calibration",
    )

    parser.add_argument(
        "--comfyui-root",
        default=None,
        metavar="PATH",
        help="Path to ComfyUI root for WAN model loading (optional)",
    )

    parser.add_argument(
        "--input-scale-summary-json",
        default=None,
        metavar="PATH",
        help="Use per-layer input_scale from summary JSON",
    )

    parser.add_argument(
        "--input-scale-summary-percentile",
        type=int,
        default=99,
        help="Percentile to use from summary JSON (default: 99)",
    )

    parser.add_argument(
        "--input-scale-summary-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to summary-derived input_scale values (default: 1.0)",
    )

    parser.add_argument(
        "--input-scale-layer-summary",
        action="store_true",
        help="Print per-layer input_scale values after conversion",
    )

    parser.add_argument(
        "--use-ck-quant",
        action="store_true",
        help="Use comfy_kitchen quantize_nvfp4 for backend-compatible packing",
    )

    parser.add_argument(
        "--calibrate-steps",
        type=int,
        default=8,
        help="Number of calibration steps (default: 8)",
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
        help="Output dtype for non-quantized tensors (default: bfloat16)",
    )

    parser.add_argument(
        "--quant-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Input dtype for quantization. Use bfloat16/float16 to match working models "
        "from HuggingFace. Use float32 for maximum precision (default: bfloat16)",
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
            gguf_path=args.gguf,
            add_input_scale=not args.no_input_scale,
            input_scale_value=args.input_scale_value,
            input_scale_from=args.input_scale_from,
            input_scale_from_fp16=args.input_scale_from_fp16,
            input_scale_method=args.input_scale_method,
            input_scale_samples=args.input_scale_samples,
            comfyui_root=args.comfyui_root,
            input_scale_summary_json=args.input_scale_summary_json,
            input_scale_summary_percentile=args.input_scale_summary_percentile,
            input_scale_summary_multiplier=args.input_scale_summary_multiplier,
            input_scale_layer_summary=args.input_scale_layer_summary,
            use_ck_quant=args.use_ck_quant,
            calibrate=args.calibrate,
            calibrate_steps=args.calibrate_steps,
            device=args.device,
            dtype=args.dtype,
            quant_dtype=args.quant_dtype,
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
