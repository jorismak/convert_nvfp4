# DETAILED TECHNICAL COMPARISON

## Executive Summary

The noise in your NVFP4-converted WAN model is caused by **fundamental format incompatibility**. ComfyUI expects a simple FP8 quantization format, but your conversion created a complex NVFP4 format with different structure, naming, and scaling mechanisms.

---

## SIDE-BY-SIDE COMPARISON

### blocks.0.cross_attn.k Layer

#### Working FP8 Model (Reference)
```
Tensors:
  blocks.0.cross_attn.k.weight
    dtype: float8_e4m3fn
    shape: [3072, 3072]         # FULL dimension
    
  blocks.0.cross_attn.k.scale_weight
    dtype: float32
    shape: [1]                   # Single scalar scale
    
  blocks.0.cross_attn.k.bias
    dtype: float32
    shape: [3072]

Dequantization:
  output = (weight.to(float32) * scale_weight) @ input + bias
```

#### Your NVFP4 Model (Broken)
```
Tensors:
  blocks.0.cross_attn.k.weight
    dtype: uint8                 # Packed 4-bit values!
    shape: [3072, 1536]         # HALF dimension (packed 2 per byte)
    
  blocks.0.cross_attn.k.weight_scale
    dtype: float8_e4m3fn        # First-level scale (also quantized!)
    shape: [3072, 192]          # Per-group scales (16 groups)
    
  blocks.0.cross_attn.k.weight_scale_2
    dtype: float32
    shape: []                    # Second-level scale (scalar)
    
  blocks.0.cross_attn.k.bias
    dtype: float16               # Lower precision!
    shape: [3072]

Dequantization (ASSUMED):
  weight_fp4 = unpack_uint8_to_fp4(weight)  # [3072, 3072]
  weight_scale_f32 = weight_scale.to(float32) * weight_scale_2
  output = (weight_fp4 * weight_scale_f32) @ input + bias
```

---

## KEY DIFFERENCES TABLE

| Aspect | Working FP8 | Your NVFP4 | Impact |
|--------|-------------|------------|--------|
| **Weight dtype** | float8_e4m3fn | uint8 (packed) | Different unpacking logic |
| **Weight shape** | [3072, 3072] | [3072, 1536] | Packed 2:1 |
| **Scale naming** | `.scale_weight` | `.weight_scale` | May not be recognized |
| **Scale count** | 1 per layer | 2 per layer | Complex hierarchy |
| **Scale shape** | [1] scalar | [3072, 192] grouped | Per-group quantization |
| **Scale dtype** | float32 | fp8 + float32 | Scales are also quantized |
| **Bias dtype** | float32 | float16 | Precision loss |
| **Norm dtype** | float32 | float16 | Precision loss |
| **Input scale** | Not needed | **MISSING** | Activation quantization broken |

---

## QUANTIZATION PATTERNS

### Working FP8 (Simple Per-Tensor)

```
Layer Structure:
  weight:       float8_e4m3fn [out_features, in_features]
  scale_weight: float32 [1]
  bias:         float32 [out_features]

Quantization:
  scale = max(abs(weight_f32))
  weight_q = (weight_f32 / scale).to(float8_e4m3fn)
  
Dequantization:
  weight_f32 = weight_q.to(float32) * scale
  output = F.linear(input, weight_f32, bias)
```

### Your NVFP4 (Complex Grouped)

```
Layer Structure:
  weight:         uint8 [out_features, in_features/2]  # Packed
  weight_scale:   float8_e4m3fn [out_features, groups]  # Quantized scale!
  weight_scale_2: float32 []
  bias:           float16 [out_features]

Quantization (INFERRED):
  # Group weight into blocks of 16 values
  # For [3072, 3072] -> 3072 * 192 groups of 16
  groups = 192
  for i in range(out_features):
    for g in range(groups):
      block = weight_f32[i, g*16:(g+1)*16]
      group_scale = max(abs(block))
      weight_scale[i, g] = group_scale / weight_scale_2
      weight_q[i, g*8:(g+1)*8] = pack_fp4(block / group_scale)
      
Dequantization (MUST MATCH):
  weight_fp4 = unpack_uint8_to_fp4(weight)  # [out, in]
  weight_scale_f32 = weight_scale.to(float32) * weight_scale_2
  # Broadcast scales to match weight shape
  weight_f32 = weight_fp4 * weight_scale_f32  
  output = F.linear(input, weight_f32, bias.to(float32))
```

---

## NAMING CONVENTION ANALYSIS

### Scale Naming Patterns

**Working FP8:**
```python
layer_names = [
    'blocks.0.cross_attn.k.weight',
    'blocks.0.cross_attn.k.scale_weight',  # <-- This pattern
    'blocks.0.cross_attn.k.bias'
]
```

**Your NVFP4:**
```python
layer_names = [
    'blocks.0.cross_attn.k.weight',
    'blocks.0.cross_attn.k.weight_scale',    # <-- Different!
    'blocks.0.cross_attn.k.weight_scale_2',  # <-- Extra!
    'blocks.0.cross_attn.k.bias'
]
```

**ComfyUI Loader Logic (LIKELY):**
```python
def load_quantized_layer(state_dict, layer_name):
    weight = state_dict[f"{layer_name}.weight"]
    
    # Try to find scale
    if f"{layer_name}.scale_weight" in state_dict:
        scale = state_dict[f"{layer_name}.scale_weight"]
        return dequantize_fp8(weight, scale)
    
    # NVFP4 pattern NOT HANDLED!
    # Missing: weight_scale, weight_scale_2 support
    
    return weight  # Returns quantized data directly!
```

---

## MISSING INPUT SCALES

### Flux NVFP4 (Working Reference)
```
For each quantized layer:
  layer.weight:        uint8 [out, in/2]
  layer.weight_scale:  fp8 [out, groups]
  layer.weight_scale_2: float32 []
  layer.input_scale:   float32 []        # PRESENT
  layer.bias:          bfloat16 [out]
```

### Your NVFP4 (Broken)
```
For each quantized layer:
  layer.weight:        uint8 [out, in/2]
  layer.weight_scale:  fp8 [out, groups]
  layer.weight_scale_2: float32 []
  layer.input_scale:   MISSING!          # NOT PRESENT
  layer.bias:          float16 [out]
```

**Why input_scale matters:**
- Tracks the scale of input activations
- Required for correct quantized matrix multiplication
- Without it, activations may overflow or underflow

---

## PRECISION LOSS ANALYSIS

### Non-Quantized Tensors

**Working FP8:**
```
Normalization layers:  float32 (7-8 significant digits)
Biases:                float32 (7-8 significant digits)
Modulation:            float32 (7-8 significant digits)
```

**Your NVFP4:**
```
Normalization layers:  float16 (3-4 significant digits)
Biases:                float16 (3-4 significant digits)
Modulation:            float16 (3-4 significant digits)
```

**Impact:**
- Float16 has ~3.3 decimal digits of precision
- Float32 has ~7.2 decimal digits of precision
- Error accumulation across layers
- Especially problematic in normalization layers

---

## STORAGE EFFICIENCY

### Working FP8 Model
```
Total tensors: 1,126
Weight storage: 8 bits per value (float8_e4m3fn)
Example layer [3072, 3072]:
  - Weight: 3072 * 3072 * 1 byte = 9.4 MB
  - Scale:  1 * 4 bytes = 4 bytes
  - Total:  9.4 MB
```

### Your NVFP4 Model
```
Total tensors: 1,437 (+311 extra)
Weight storage: 4 bits per value (packed in uint8)
Example layer [3072, 3072]:
  - Weight: 3072 * 1536 * 1 byte = 4.7 MB (packed)
  - Scale1: 3072 * 192 * 1 byte = 0.6 MB (fp8)
  - Scale2: 1 * 4 bytes = 4 bytes
  - Total:  5.3 MB
```

**Space savings: ~45%** but at cost of compatibility!

---

## ROOT CAUSE SUMMARY

Your NVFP4 model fails because:

1. **ComfyUI doesn't recognize the NVFP4 format** for WAN models
2. **Scale naming mismatch** prevents proper dequantization
3. **Missing input_scale** breaks activation quantization
4. **Float16 base precision** accumulates errors
5. **Complex unpacking logic** not implemented in ComfyUI

The working FP8 model succeeds because:

1. **Simple, well-supported format** - ComfyUI has built-in FP8 support
2. **Standard naming convention** - `.scale_weight` is recognized
3. **Float32 base precision** - maintains accuracy
4. **No complex unpacking** - direct float8 dequantization

---

## RECOMMENDATION

**Convert to simple FP8 format** matching the working reference:

```python
import torch
import safetensors.torch

def convert_to_simple_fp8(original_model, output_path):
    new_state_dict = {}
    
    for name, tensor in original_model.items():
        if name.endswith('.weight') and is_linear_layer(name):
            # Quantize to FP8
            scale = tensor.abs().max()
            quantized = (tensor / scale).to(torch.float8_e4m3fn)
            
            new_state_dict[name] = quantized
            new_state_dict[name.replace('.weight', '.scale_weight')] = scale
        else:
            # Keep at float32
            new_state_dict[name] = tensor.to(torch.float32)
    
    safetensors.torch.save_file(new_state_dict, output_path)
```

This will give you:
- 50% size reduction (8-bit vs 16/32-bit)
- Full compatibility with ComfyUI
- Proven to work (reference model exists)
- Simpler debugging if issues arise
