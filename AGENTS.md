# NVFP4 Mixed Quantization Project

# Notes about workspace directory

You might need to edit or scan/analyze the ComfyUI sourcecode. This is available in `comfyui-source` but is a symlink / junction. THIS CAN LOOP if you go into `comfyui-source\nvfp4-conv` so keep out!

You might need the comfy-kitchen sourecode (this contains multiple backends for tensor handling in modern comfyui, enabling nvfp4 code). This is checked out in `comfy-kitchen` in the workspace. This is just for you to reference / analyze, not modify.

# Notes about python environment

We are working in the `comfy2` venv that I use to run my ComfyUI installation. This means it has a lot of libraries ready to go like pytorch, diffuser, safetensors, etc...

Do _not_ start another venv, and always use this one to execute scripts or snippets.

Do not install new dependencies without asking first (or better yet, give the user the command and let them do it).

## Project Goal

Create a script to quantize diffusion models from FP32/FP16 to **mixed-precision NVFP4** format that works with ComfyUI and produces clean, artifact-free output.

**NVFP4 is the goal** - we're not interested in pure FP8 or FP16 models.

---

## Target Models

Models we want to support:
- **WAN 2.2 TI2V 5B** (current test model - both i2v and t2v, small and sensitive to quantization)
- **WAN 2.2 I2V 14B**
- **WAN 2.2 T2V 14B**
- **Qwen Image 2512**
- **Qwen Image Edit 2511**

**We're NOT interested in Flux models** - they're only used as format references.

---

## Quantization Strategy

We determine which layers to quantize by **analyzing GGUF versions** of the same models:

1. **Look at GGUF files** to see which layers they quantized to what precision
2. **Create presets** based on GGUF layer selection:
   - Layers at F32 in GGUF → keep as F32
   - Layers at F16 in GGUF → keep as FP16/BF16
   - Layers at Q4_0/Q4_1 in GGUF → quantize to NVFP4

3. **Preset types**:
   - Model-specific presets (based on GGUF analysis)
   - General "smart" preset (tries to work on any model)
   - Older presets and "all" mode (may exist from before GGUF analysis)

---

## Current Test Setup

### Source Model
**Location**: `D:\comfy2\ComfyUI\nvfp4-conv\wan2.2-ti2v-5b\`
- Sharded FP32 format (3 safetensors + index.json)
- WAN 2.2 TI2V 5B architecture
- 30 transformer blocks
- Small and sensitive to quantization errors

### Reference Models (Working)
1. **GGUF Q4_0**: `D:\ComfyUI\ComfyUI\models\diffusion_models\Wan2.2-TI2V-5B-Q4_0.gguf`
   - Clean output, occasional stable artifacts
   - Used as reference for layer selection

2. **GGUF Q5_K_M**: `D:\ComfyUI\ComfyUI\models\diffusion_models\Wan2.2-TI2V-5B-Q5_K_M.gguf`
   - Alternative GGUF reference

3. **FP8 Scaled (KJ)**: `D:\ComfyUI\ComfyUI\models\diffusion_models\Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors`
   - External reference (not created by us)
   - Exceptional quality
   - Pure FP8, not mixed

4. **FP16**: `D:\ComfyUI\ComfyUI\models\diffusion_models\wan2.2_ti2v_5B_fp16.safetensors`
   - Full precision reference
   - Not created by us

### Output Location
All converted models are written to: `D:\ComfyUI\ComfyUI\models\diffusion_models\`

User tests them in ComfyUI after conversion.

---

## The Problem

**Current Issue**: NVFP4 mixed models produce noisy results - far worse quality than GGUF versions of the same model.

**Symptoms**:
- Heavy random noise in every frame
- Noise appears in block patterns
- Different noise pattern each frame (unstable)
- Subjects/movement correct but heavily corrupted

**What We Know**:
- The issue is NOT how we quantize values to NVFP4 (the algorithm works)
- We suspect a **scaling, metadata, or compatibility issue**
- Something about how the model is loaded/interpreted in ComfyUI

---

## Recent Success: Pure FP8 Quantization

We **finally** managed to create a working pure FP8 model (for testing/comparison purposes).

**Key Discoveries**:
1. **Scale Convention**: Pure FP8 needs tensors stored as values between **-1.0 and 1.0**, then add scaling metadata/blocks
2. **Marker Tensor**: Needs a special dummy block with zeros called `scaled_fp8` to indicate to ComfyUI it's a scaled-fp8 version
3. **Result**: This produced output in line with our reference KJ model

**FP8 Format Requirements** (from successful conversion):
- Each quantized layer has TWO tensors:
  - `layer.weight` - FP8 E4M3 tensor (full dimensions, NOT packed)
  - `layer.scale_weight` - F32 scalar (shape `[1]`)
- Marker: `scaled_fp8` tensor with 2 FP8 elements (dummy, can be zeros)
- Metadata: `format: "pt"`, `model_type: "<model_name>"`
- Scale convention: `scale = amax` (stores FP8 values in [-1, +1] range)

---

## BREAKTHROUGH: The Missing `input_scale` Fix

### The Problem Identified

**Root cause**: Our NVFP4 models were missing `input_scale` tensors, causing noisy output!

**What we found**:
- Working Flux NVFP4: 110 quantized layers → 110 `input_scale` tensors ✓
- Broken WAN NVFP4: 210 quantized layers → 0 `input_scale` tensors ✗

### Why `input_scale` Matters

From `comfy-kitchen` code analysis:

1. **Without `input_scale`**: ComfyUI uses **dynamic quantization** (recalculates scale from input each forward pass)
   - Causes numerical instability
   - Different scales across forward passes
   - Results in heavy noise/artifacts

2. **With `input_scale`**: Input activations are quantized using a **fixed, pre-computed scale**
   - Consistent quantization across forward passes
   - Stable numerical behavior
   - Clean output

### The Fix

**Modified `convert_nvfp4.py`** to ALWAYS add `input_scale` for NVFP4 layers:

```python
# Use calibration if requested (accurate but slow)
if calibrate:
    input_scale = calibrate_layer(weight_device, num_steps=calibrate_steps)
else:
    # Use reasonable default based on weight statistics
    # Formula: scale = amax / (F8_E4M3_MAX * F4_E2M1_MAX)
    amax = torch.amax(weight_device.abs())
    input_scale_value = amax / (448.0 * 6.0)
    input_scale = torch.tensor([input_scale_value], dtype=torch.float32)
```

### Test Models Created

1. **`wan2.2-ti2v-5b-nvfp4-WITH-INPUT-SCALE.safetensors`** (FAILED - wrong values)
   - 210 `input_scale` tensors computed from WEIGHT statistics
   - Values: mean ~0.0001 (**65x too small!**)
   - Result: Produced unusable garbage (confirmed `input_scale` affects output)

2. **`wan2.2-ti2v-5b-nvfp4-FIXED-INPUT-SCALE.safetensors`** (Heuristic - untested)
   - 210 `input_scale` tensors using fixed heuristic (amax=10.0)
   - Values: constant ~0.0037 (**1.7x too large** vs calibrated)
   - May work but not optimal

3. **`wan2.2-ti2v-5b-nvfp4-CALIBRATED.safetensors`** (RECOMMENDED)
   - 210 `input_scale` tensors measured from **actual activation statistics**
   - Values: mean ~0.002, range 0.002-0.0025
   - Uses 16 steps of random input calibration to measure realistic activation ranges
   - **This should produce the best results!**

**Next step**: User test the CALIBRATED model in ComfyUI!

---

## NVFP4 Format Reference

**Tensor Structure** (per quantized layer):
- `{layer}.weight` - Packed uint8 (2 FP4 values per byte)
- `{layer}.weight_scale` - Block scales (float8_e4m3fn, one per 16 values)
- `{layer}.weight_scale_2` - Per-tensor scale (float32 scalar)
- `{layer}.input_scale` - **REQUIRED** for stable quantization (float32 scalar)
- `{layer}.bias` - Kept at original precision (usually BF16 or F32)

**Format Details**:
- Block size: 16 values per quantization block
- FP4 range: E2M1 format, max value = 6.0
- Block scale range: FP8 E4M3, max value = 448.0
- Input scale: Computed as `amax / (448.0 * 6.0)` or via calibration

---

## Calibration (Parked for Now)

There is code for calibrating quantized models (`calibrate_real.py` and related scripts), but **we've parked this for now**. 

Calibration is not important until we get a basic NVFP4-mixed version working. The current issue is about format/loading compatibility, not calibration.

---

## Key Scripts

### Main Converter
**File**: `convert_nvfp4.py` (~1900+ lines)

**Components**:
1. NVFP4 quantization algorithm
2. Preset system (model-specific configs, skip patterns)
3. Layer classification (which layers to quantize)
4. Sharded model support

**Usage**:
```bash
python convert_nvfp4.py \
    "D:\comfy2\ComfyUI\nvfp4-conv\wan2.2-ti2v-5b\diffusion_pytorch_model.safetensors.index.json" \
    "D:\ComfyUI\ComfyUI\models\diffusion_models\output-name.safetensors" \
    --preset <preset_name> \
    --mode <all|safe>
```

---

## CRITICAL: ComfyUI Code Locations

**IMPORTANT**: When searching for ComfyUI loading/quantization code, you MUST search BOTH locations:

1. **Base ComfyUI**: `D:\ComfyUI\ComfyUI\comfy\`
   - Core operations in `ops.py`
   - Basic FP8 support in `quant_ops.py`

2. **Comfy-Kitchen (NVFP4 support)**: `D:\comfy2\ComfyUI\nvfp4-conv\comfy-kitchen\`
   - **This is where NVFP4 format support lives!**
   - Extended quantization formats
   - NVFP4 layout classes and operations
   - Mixed-precision loading logic

**DO NOT assume something is missing just because it's not in base ComfyUI** - always check comfy-kitchen!

---

## What We're NOT Doing

**Don't suggest**:
- Switching to pure FP8 or FP16 (NVFP4 mixed is the goal)
- Post-quantization calibration (doesn't fix format/loading issues)
- Different quantization algorithms (the quantization itself works)
- Working on Flux models (only used as reference)

**Focus on**:
- Understanding what ComfyUI expects from NVFP4 mixed models
- Comparing with working reference models
- Finding missing metadata, tensors, or format issues
- Fixing compatibility, not the quantization algorithm itself
