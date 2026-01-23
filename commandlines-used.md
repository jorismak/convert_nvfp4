# Wan 2.2 I2V 14b

Highnoise and lownoise use basically the same commandline.

```bash
# lownoise
python convert_nvfp4.py wan2.2-i2v-a14b-low-noise\diffusion_pytorch_model.safetensors.index.json d:\comfyui\comfyui\models\diffusion_models\wan2.2-i2v-lownoise-nvfp4mixed.safetensors --gguf d:\comfyui\comfyui\models\diffusion_models\Wan2.2-I2V-A14B-lowNoise-Q5_K_M.gguf --gguf-nvfp4-max-bitdepth 5 --use-fp8 --low-mem --gguf-keep-edge-blocks-fp16 --quant-dtype float32 --dtype float16 --input-scale-summary-json scales_summary_wan22_14b_lownoise.json --input-scale-summary-percentile 99.9 --input-scale-summary-multiplier 1.08 --full-precision-mm-layers sensitive_layers_wan22_i2v_14b_low.txt --full-precision-mm-cross-attn-qkvo

# highnoise
python convert_nvfp4.py wan2.2-i2v-a14b-high-noise\diffusion_pytorch_model.safetensors.index.json wan2.2-i2v-highnoise-nvfp4mixed.safetensors --gguf Wan2.2-I2V-A14B-highNoise-Q5_K_M.gguf --gguf-nvfp4-max-bitdepth 5 --use-fp8 --low-mem --gguf-keep-edge-blocks-fp16 --quant-dtype float32 --dtype float16 --input-scale-summary-json scales_summary_wan22_14b_highnoise.json --input-scale-summary-percentile 99.9 --input-scale-summary-multiplier 1.08 --full-precision-mm-layers sensitive_layers_wan22_i2v_14b_high.txt --full-precision-mm-cross-attn-qkvo
```

I started with a Q5_K_M GGUF quant from QuantStack (`--gguf`). Everything that is Q5_K gets quantized into NVFP4 (`--gguf-nvfp4-max-bitdepth 5`). Every Q6 / Q7 / Q8 quant gets quantized to fp8 (`--use-fp8`). The first two blocks and last two blocks are kept as fp16 in their entirety (`--gguf-keep-edge-blocks-fp16`).

Input scales are applied to the NVFP4 layers by taking a high percentile from the debug-info made by running 32 prompts through the model in ComfyUI (`--input-scale-summary-json scales_summary_wan22_14b_lownoise.json --input-scale-summary-percentile 99.9 --input-scale-summary-multiplier 1.08`). The input scales are multiplied by 1.08 to get a bit of margin to prevent clipping.

Looking at the histograms of `scale_cv` made by `analyze_input_scale_log.py`, it seems we have a high number of layers that are just shy of 0.01 for the scale_cv statistic. So, I took every NVFP4 layer which has a scale_cv of 0.01 or higher. I noticed that a lot of those layers where cross_attn layers.

The same with the rel_rmse histogram of the input sensitivity analyzer. The layers that have seem to be sensitive to input values are the cross_attn layers.

So, I decided to put every cross_attn layer in the 'full precision' list (which means they are NVFP4 W4A16 instead of NVFP4 W4A4), together with the other layers which have a high spread of input values (scale_cv and kurtosis), to create `sensitive_layers_wan22_i2v_14b_low.txt` and `sensitive_layers_wan22_i2v_14b_high.txt`. They are marked as full-precision-mm by `--full-precision-mm-layers sensitive_layers_wan22_i2v_14b_high.txt --full-precision-mm-cross-attn-qkvo` in the CLI.

## Calibration

For a version used in ComfyUI calibration, you would leave out all the full-precision-mm options and leave out all the input-scale options. And then include `--no-input-scale`. That way no input-scales are baked into the model, and ComfyUI will try to determine them on the fly. This is logged into a file which creates the calibration data in `calib/`.

# Wan .2 TI2V 5b

calibraton command:

```bash
python convert_nvfp4.py wan2.2-ti2v-5b\diffusion_pytorch_model.safetensors.index.json d:\comfyui\comfyui\models\diffusion_models\ti2v-calib.safetensors --gguf d:\comfyui\comfyui\models\diffusion_models\Wan2.2-TI2V-5B-Q5_K_M.gguf --gguf-nvfp4-max-bitdepth 5 --low-mem --gguf-keep-edge-blocks-fp16 --quant-dtype float32 --dtype float16 --no-input-scale
```

final quant command:

```bash
python convert_nvfp4.py wan2.2-ti2v-5b\diffusion_pytorch_model.safetensors.index.json d:\comfyui\comfyui\models\diffusion_models\wan2.2-ti2v-5b-nvp4mixed.safetensors --gguf d:\comfyui\comfyui\models\diffusion_models\Wan2.2-TI2V-5B-Q5_K_M.gguf --gguf-nvfp4-max-bitdepth 5 --low-mem --gguf-keep-edge-blocks-fp16 --quant-dtype float32 --dtype float16 --input-scale-summary-json scales_summary_wan22_5b.json --input-scale-summary-percentile 99.9 --input-scale-summary-multiplier 1.08  --full-precision-mm-layers sensitive_layers_wan22_5b.txt --full-precision-mm-cross-attn-qkvo
```

