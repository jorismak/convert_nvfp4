# ANALYSIS SUMMARY

## CRITICAL ISSUE IDENTIFIED

Your NVFP4-converted WAN model has **fundamental structural incompatibilities** with the working FP8 reference model.

---

## KEY FINDINGS

### 1. INCOMPATIBLE QUANTIZATION SCHEME

**Working FP8 Model (Simple):**
- Weight: `float8_e4m3fn [3072, 3072]` - FULL dimensions
- Scale: `.scale_weight` (single float32 scalar)
- No weight packing

**Your NVFP4 Model (Complex):**  
- Weight: `uint8 [3072, 1536]` - **HALF width** (packed 4-bit)
- Scale: `.weight_scale` (grouped fp8) + `.weight_scale_2` (scalar fp32)
- Two-level scale hierarchy
- **MISSING `.input_scale`** for activation quantization

### 2. WRONG BASE PRECISION

| Tensor Type | Working FP8 | Your NVFP4 |
|-------------|-------------|------------|
| Biases | float32 | **float16** |
| Norms | float32 | **float16** |
| Modulation | float32 | **float16** |

### 3. NAMING CONVENTION MISMATCH

```
Working FP8:  .scale_weight
Your NVFP4:   .weight_scale  <-- ComfyUI may not recognize this!
```

### 4. TENSOR COUNT

- Working FP8: 1,126 tensors
- Your NVFP4: 1,437 tensors (+311 extra scale tensors)

---

## ROOT CAUSE

ComfyUI's WAN loader expects **simple FP8 format** with:
- Float8_e4m3fn weights (full dimensions)
- Single scalar scale per weight
- Float32 for non-quantized tensors
- Naming pattern: `.scale_weight`

Your conversion created **complex NVFP4 format** with:
- Packed uint8 weights (half dimensions)
- Grouped + hierarchical scaling
- Float16 base precision
- Different naming: `.weight_scale` + `.weight_scale_2`
- Missing input scales

**Result:** ComfyUI cannot properly dequantize your model, causing noise.

---

## SOLUTION

### Option 1: RECOMMENDED - Convert to Simple FP8
Match the working reference exactly:
```python
# Use float8_e4m3fn for weights
weight_quantized = weight.to(torch.float8_e4m3fn)
scale = weight.abs().max()  # Single scalar scale
```

### Option 2: Fix NVFP4 Format
Would require:
1. Adding input_scale tensors for all layers
2. Changing naming to match ComfyUI expectations
3. Using float32 for non-quantized layers
4. Ensuring unpacking logic matches ComfyUI's NVFP4 implementation

**However:** ComfyUI may not even support NVFP4 for WAN models!

---

## VERIFICATION COMMANDS

Compare layer structure:
```bash
# Working FP8
blocks.0.cross_attn.k.weight: float8 [3072, 3072]
blocks.0.cross_attn.k.scale_weight: float32 [1]

# Your NVFP4 (broken)
blocks.0.cross_attn.k.weight: uint8 [3072, 1536]  
blocks.0.cross_attn.k.weight_scale: float8 [3072, 192]
blocks.0.cross_attn.k.weight_scale_2: float32 []
```

---

## NEXT STEPS

1. **Check if ComfyUI supports NVFP4 for WAN** (likely NO)
2. **Convert using simple FP8 quantization** matching working model
3. **Use float32 for all non-quantized tensors**
4. **Use `.scale_weight` naming convention**
5. **Test with same prompts to verify quality**

The working FP8 model proves that FP8 quantization works well for WAN - you don't need the complexity of NVFP4!
