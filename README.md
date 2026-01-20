# ⚠️ DISCLAIMER: VIBECODER / AI SLOP

**This project is a learning exercise written by an AI-assisted hobbyist. The author does not have real expertise in quantization, model optimization, or the “best” decisions for model quality. Use at your own risk. Do not assume the results are correct or optimal.**

---

# NVFP4 Converter + Analysis Scripts

This folder contains a set of scripts to **convert diffusion models to mixed-precision NVFP4** and to **analyze/inspect** model layers and quantization behavior.

If you’re new to quantization: think of it as *shrinking the numbers inside a model* so it uses less VRAM and runs faster, at the risk of lower quality. NVFP4 is a very small number format (4-bit) mixed with larger formats to keep quality stable.

Basically, you can do NVFP4 in two ways: W4A4, or W4A16.

See it like this: You can store the weights in nvfp4, but still 'run the layers' of the model in full precision (or fp16). That means you get the smaller size and RAM savings, but not really the speed improvements. This is what W4A16 is. And this is where the `--full-precision-mm` flags are for.

If you want to run the nvfp4 weights really as nvfp4 on Blackwell hardware, you don't want to upscale the nvfp4 weights to fp16 and run like that, you want to scale the input _down_ to the weights, down to nvfp4, and then run everything in nvfp4.

This is where the Blackwell speed comes in, and this is called W4A4. But this is also where precision problems start to come into play and you'll notice that it might be hard to get good quality out of it.

So, "input * weights = result".
 * Either, "input (fp16) * weights (nvfp4 -> fp16) = result (fp16)". Respectable quality, no speed improvement versus a GGUF version.
 * or, "input (fp16 -> nvfp4) * weights (nvfp4) = result (nvfp4)". Really twice as fast as fp8/fp16, but quality suffers.

---

## What each script does

### `convert_nvfp4.py`
**Main converter.**
- Reads a diffusion model (usually a sharded `.safetensors` model with an `index.json`).
- Decides which layers to keep in higher precision and which to convert to NVFP4.
- Writes a new `.safetensors` file you can load in ComfyUI.

**Typical use case:** Convert a large FP16/FP32 model into a mixed-precision NVFP4 model.

---

### `list_layers.py`
**Layer inventory tool.**
- Lists layers and their names, types, size.
- Useful when building or debugging layer-selection presets.

**Typical use case:** “What layers does my model actually have?”. Or "what did this other person do when creating their mixed fp8/fp16 of GGUF version?".

---

### `analyze_input_precision_sensitivity.py`
**Sensitivity analyzer.**
- Tests how sensitive different layers are when inputs are quantized.
- Produces reports to help decide which layers should *not* be NVFP4 (or NVFP4 but full-precision-mm)

**Typical use case:** “Which layers should stay in FP16/BF16 or NVFP4-W4A16?”

It 'runs' a real version of the model, and looks for differences when the inputs are fp16 and when the inputs are nvfp4-w4a4. Stats about this are written to a json file, which you can use to decide what you want to do.

---

### `analyze_input_scale_log.py`
**Input scale log analyzer.**
- Parses logs or saved stats about `input_scale` values.
- Helps diagnose noise or instability that comes from bad scaling.

NVFP4 (W4A4) has one 'scale' value per 16 float values. This is handled automatically. But it also has one scale value per whole tensor.

When you run NVFP4 W4A4 layers, the _input_ into those layers is scaled to nvfp4 during running of the model (during use in ComfyUI). We can let ComfyUI try to autodetect a scale value as it runs (`--no-input-scale` in convert_nvfp4.py), or you can specify a input-scale for ComfyUI or other tools to use when _they_ quantize their inputs to nvfp4 for running the model.

This is 'calibration', you need to have an idea of what the model expects and what realistic input values are, per step, per layer. `--calibrate-from-fp16` tries to emulate that by firing some random values into the model.

But the real test is to run calibration passes. Really run the model in a real workflow, with different prompts and inputs. I've patched ComfyUI (patch down below) to write it's autodetected values to a file + stdout when a certain environment variable is set. I then run lots of prompts with as varied set of inputs as possible, so ComfyUI writes it's autodetected input-scale per layer to a file.

That big file is then analyzed by this analyze_input_scale_log.py script, to get statistics about inputs / input_scale per layer. That can be used to write hard-bakes input_scale values per layer, in the hope of improving over the auto-detection.

**Typical use case:** “Are my `input_scale` values too small or too large?”

---

### `nvfp4_calibration.py`
**Calibration helper module**

Not meant to be used directly, is used by convert_nvfp4.py if you use the `--calibrate-from-fp16` option(s).

- Runs a small calibration loop to estimate more realistic activation scales.
- Produces more stable `input_scale` values than a naive guess.

**Typical use case:** “My NVFP4 model is noisy; I want better scaling.”

---

## Minimal workflow (simple mental model)

1. **Identify your model** (usually a sharded `.safetensors` with an `index.json`).
2. **List layers** with `list_layers.py` if you need to understand the model layout.
2b. **List layers** with `list_layres.py` take a sneak peek at other models to see what quantizatons they used for which layers. Identify which layers are apparently important.
3. **Analyze sensitivity** with `analyze_input_precision_sensitivity.py` if you want a smarter preset.
4. **Convert** using `convert_nvfp4.py` with a preset.
5. **If output is noisy**, analyze input scales with `analyze_input_scale_log.py` or try `nvfp4_calibration.py`.

---

## Notes for beginners

- **Mixed-precision** means some layers stay FP16/BF16 while others become NVFP4.
- **NVFP4 is tiny**. Without correct scaling, results will look very noisy.
- **`input_scale` is critical.** Missing or bad `input_scale` values often cause unstable output.

---

## ComfyUI patch for input_scale logging

```patch
diff --git a/comfy/ops.py b/comfy/ops.py
index 415c39e9..15377d67 100644
--- a/comfy/ops.py
+++ b/comfy/ops.py
@@ -18,6 +18,7 @@

 import torch
 import logging
+import os
 import comfy.model_management
 from comfy.cli_args import args, PerformanceFeature
 import comfy.float
@@ -543,6 +544,8 @@ def mixed_precision_ops(quant_config={}, compute_dtype=torch.bfloat16, full_prec

                 device = self.factory_kwargs["device"]
                 layer_name = prefix.rstrip('.')
+                self._quant_layer_name = layer_name
+                self._input_scale_log_count = 0
                 weight_key = f"{prefix}weight"
                 weight = state_dict.pop(weight_key, None)
                 if weight is None:
@@ -680,6 +683,56 @@ def mixed_precision_ops(quant_config={}, compute_dtype=torch.bfloat16, full_prec
                         scale = getattr(self, 'input_scale', None)
                         if scale is not None:
                             scale = comfy.model_management.cast_to_device(scale, input.device, None)
+                        if os.getenv("COMFY_NFP4_LOG_INPUT_SCALE", "") and getattr(self, "quant_format", None) in {"nvfp4", "float8_e4m3fn", "float8_e5m2"}:
+                            try:
+                                max_logs = int(os.getenv("COMFY_NFP4_LOG_INPUT_SCALE_MAX", "5"))
+                            except ValueError:
+                                max_logs = 5
+                            if max_logs < 0 or self._input_scale_log_count < max_logs:
+                                self._input_scale_log_count += 1
+                                layer = getattr(self, "_quant_layer_name", "<unknown>")
+                                fmt = getattr(self, "quant_format", "unknown")
+                                if scale is None:
+                                    try:
+                                        from comfy_kitchen.float_utils import F8_E4M3_MAX, F4_E2M1_MAX
+                                        denom = F8_E4M3_MAX * F4_E2M1_MAX
+                                    except Exception:
+                                        denom = 448.0 * 6.0
+                                    amax = float(input_reshaped.abs().amax().item())
+                                    dyn_scale = amax / denom
+                                    msg = (
+                                        f"[{fmt}] layer={layer} dynamic input_scale={dyn_scale:.6f} "
+                                        f"amax={amax:.6f} shape={tuple(input_reshaped.shape)}"
+                                    )
+                                    logging.info(msg)
+                                    log_path = os.getenv("COMFY_NFP4_LOG_INPUT_SCALE_PATH", "")
+                                    if log_path:
+                                        try:
+                                            with open(log_path, "a", encoding="utf-8") as f:
+                                                f.write(
+                                                    f"{fmt}\tdynamic\t{layer}\t{dyn_scale:.6f}\t{amax:.6f}\t{tuple(input_reshaped.shape)}\n"
+                                                )
+                                        except Exception:
+                                            pass
+                                else:
+                                    try:
+                                        scale_val = float(scale.item())
+                                    except Exception:
+                                        scale_val = float(scale.mean().item())
+                                    msg = (
+                                        f"[{fmt}] layer={layer} provided input_scale={scale_val:.6f} "
+                                        f"shape={tuple(input_reshaped.shape)}"
+                                    )
+                                    logging.info(msg)
+                                    log_path = os.getenv("COMFY_NFP4_LOG_INPUT_SCALE_PATH", "")
+                                    if log_path:
+                                        try:
+                                            with open(log_path, "a", encoding="utf-8") as f:
+                                                f.write(
+                                                    f"{fmt}\tprovided\t{layer}\t{scale_val:.6f}\t-\t{tuple(input_reshaped.shape)}\n"
+                                                )
+                                        except Exception:
+                                            pass
                         input = QuantizedTensor.from_float(input_reshaped, self.layout_type, scale=scale)

                 output = self.forward_comfy_cast_weights(input)
```

---

## Support / expectations

This project is experimental. Expect to iterate and test. If something looks wrong, it probably *is* wrong.

