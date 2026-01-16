# CRITICAL FINDINGS: Model Structure Comparison

## Summary
I have analyzed all three models and identified the ROOT CAUSE of the noise issues in our converted model.

---

## Model Overview

### 1. flux1-dev-nvfp4.safetensors (WORKING - Flux architecture)
- **Total tensors:** 1464
- **Architecture:** Flux (different from WAN)
- **Quantization:** Mixed nvfp4 + float8_e4m3fn
- **Dtypes:** 514 bfloat16, 532 float32, 266 float8_e4m3fn, 152 uint8

### 2. Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors (WORKING - WAN reference)
- **Total tensors:** 1126
- **Architecture:** WAN
- **Quantization:** FP8 only (float8_e4m3fn)
- **Dtypes:** 825 float32, 301 float8_e4m3fn, 0 uint8

### 3. wan2.2-ti2v-5b-nvfp4-ck.safetensors (BROKEN - Our conversion)
- **Total tensors:** 1437
- **Architecture:** WAN
- **Quantization:** NVFP4 (attempted)
- **Dtypes:** 519 float16, 306 float32, 306 float8_e4m3fn, 306 uint8

---

## CRITICAL DIFFERENCES

### Issue 1: Naming Convention Mismatch

**Working FP8 model (reference):**
```
blocks.0.cross_attn.k.weight: float8_e4m3fn, shape=[3072, 3072]
blocks.0.cross_attn.k.scale_weight: float32, shape=[1]
blocks.0.cross_attn.k.bias: float32, shape=[3072]
```

**Our NVFP4 model (broken):**
```
blocks.0.cross_attn.k.weight: uint8, shape=[3072, 1536]  <-- WRONG SHAPE!
blocks.0.cross_attn.k.weight_scale: float8_e4m3fn, shape=[3072, 192]
blocks.0.cross_attn.k.weight_scale_2: float32, shape=[]
blocks.0.cross_attn.k.bias: float16, shape=[3072]
```

### Issue 2: Scale Naming Pattern

**Working FP8:**
- Uses `.scale_weight` suffix
- Single scale per weight tensor
- Scale shape: [1] (scalar for entire tensor)

**Our NVFP4 (broken):**
- Uses `.weight_scale` and `.weight_scale_2` suffixes
- Two-level scaling hierarchy
- Scale shape: [3072, 192] (per-channel/grouped scaling)

### Issue 3: Weight Matrix Dimensions

**Working FP8:**
- `blocks.0.cross_attn.k.weight`: [3072, 3072] - FULL precision dimension
- `blocks.0.ffn.0.weight`: [14336, 3072] - FULL precision dimension

**Our NVFP4 (broken):**
- `blocks.0.cross_attn.k.weight`: [3072, 1536] - COMPRESSED dimension (divided by 2!)
- `blocks.0.ffn.0.weight`: [14336, 1536] - COMPRESSED dimension (divided by 2!)

### Issue 4: Base Dtype for Non-Quantized Layers

**Working FP8:**
- All non-quantized tensors: **float32**
- Normalization layers: float32
- Biases: float32
- Modulation: float32

**Our NVFP4 (broken):**
- All non-quantized tensors: **float16**
- Normalization layers: float16
- Biases: float16
- Modulation: float16

---

## ROOT CAUSES

### 1. **Weight Packing Issue**
Our NVFP4 conversion is packing TWO 4-bit values into each uint8 byte, which:
- Reduces the weight matrix dimension from [3072, 3072] to [3072, 1536]
- This is CORRECT for storage efficiency
- BUT the unpacking/dequantization is likely WRONG

### 2. **Scale Hierarchy Mismatch**
The working FP8 model uses:
- Simple per-tensor scaling: `scale_weight` [1]

Our NVFP4 uses:
- Complex grouped scaling: `weight_scale` [3072, 192] + `weight_scale_2` []
- This suggests 192 groups of 8 values each (3072/192 = 16, and 1536/192 = 8)

### 3. **Precision Loss in Base Dtype**
Using float16 instead of float32 for:
- Biases
- Normalization weights
- Modulation parameters

This could accumulate errors during inference.

### 4. **Missing Input Scales**
Our NVFP4 model has NO `.input_scale` tensors at all!
- This means activation quantization is not being tracked
- The working Flux nvfp4 model HAS input_scale for every quantized layer

---

## QUANTIZATION STRUCTURE ANALYSIS

### Working FP8 Pattern (Simple):
```
layer.weight: float8_e4m3fn [out, in]
layer.scale_weight: float32 [1]
layer.bias: float32 [out]
```

### Our NVFP4 Pattern (Complex):
```
layer.weight: uint8 [out, in/2]  <-- Packed 4-bit values
layer.weight_scale: float8_e4m3fn [out, in/16]  <-- First level scale
layer.weight_scale_2: float32 []  <-- Second level scale
layer.bias: float16 [out]
NO input_scale!  <-- MISSING!
```

### Flux NVFP4 Pattern (Reference):
```
layer.weight: uint8 [out, in/2]  <-- Packed 4-bit values
layer.weight_scale: float8_e4m3fn [out, in/16]  <-- First level scale
layer.weight_scale_2: float32 []  <-- Second level scale
layer.input_scale: float32 []  <-- INPUT SCALE PRESENT!
layer.bias: bfloat16 [out]
```

---

## SPECIFIC LAYER COMPARISON

### Cross Attention K Layer

**Working FP8:**
```
blocks.0.cross_attn.k.weight: float8_e4m3fn [3072, 3072]
blocks.0.cross_attn.k.scale_weight: float32 [1]
blocks.0.cross_attn.k.bias: float32 [3072]
```

**Our NVFP4:**
```
blocks.0.cross_attn.k.weight: uint8 [3072, 1536]  <-- Half width due to packing
blocks.0.cross_attn.k.weight_scale: float8_e4m3fn [3072, 192]
blocks.0.cross_attn.k.weight_scale_2: float32 []
blocks.0.cross_attn.k.bias: float16 [3072]
```

**Inference:**
- Weight unpacking: Each uint8 contains 2x 4-bit values
- Groups: 3072 / 192 = 16 scale groups
- Per group: 1536 / 192 = 8 uint8 values = 16 fp4 values
- Scale hierarchy: weight = uint8_to_fp4 * weight_scale * weight_scale_2

---

## QUANTIZATION METADATA

### Working FP8 Model:
- **NO metadata present** - simple structure, no special handling needed

### Our NVFP4 Model:
- **Metadata present but likely IGNORED by ComfyUI**
- Contains quantization format info that may not be read

### Flux NVFP4 Model:
- **Has extensive metadata** including format version and per-layer quantization info
- ComfyUI knows how to read this

---

## TENSOR COUNT DIFFERENCES

- Working FP8: 1126 tensors
- Our NVFP4: 1437 tensors (311 MORE)
- Extra tensors from: weight_scale + weight_scale_2 for each quantized layer

---

## CONCLUSIONS

### The noise in our model is caused by:

1. **Missing input_scale tensors** - Activation quantization not tracked
2. **Incompatible scale naming** - ComfyUI expects `.scale_weight` not `.weight_scale`
3. **Wrong base dtype** - float16 instead of float32 for non-quantized layers
4. **Possible unpacking errors** - The uint8 -> fp4 unpacking may not match expected layout

### To fix:

1. Either convert back to simple FP8 format (matching working reference)
2. Or ensure NVFP4 format matches Flux reference exactly:
   - Add input_scale tensors
   - Change naming to match expected pattern
   - Use float32 for non-quantized layers
   - Ensure unpacking logic is correct

### Recommendation:

**Use simple FP8 quantization** matching the working reference model:
- Single scale per tensor
- float8_e4m3fn for weights
- float32 for everything else
- No complex grouped scaling
- Much simpler and proven to work
