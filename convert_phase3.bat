@echo off
REM Phase 3: Full GGUF Q4_0 match - quantize V and FFN.2 to NVFP4
REM Run this if Phase 2 produces clean output

echo Converting WAN 2.2 TI2V 5B with Phase 3 preset (full GGUF Q4_0 match)...

python convert_nvfp4.py ^
    "D:\comfy2\ComfyUI\nvfp4-conv\wan2.2-ti2v-5b\diffusion_pytorch_model.safetensors.index.json" ^
    "D:\ComfyUI\ComfyUI\models\diffusion_models\wan2.2-ti2v-5b-nvfp4-phase3.safetensors" ^
    --preset quant-test-phase3 ^
    --mode all

echo.
echo Conversion complete!
echo Output: D:\ComfyUI\ComfyUI\models\diffusion_models\wan2.2-ti2v-5b-nvfp4-phase3.safetensors
echo.
echo This should match GGUF Q4_0 quantization exactly (except using NVFP4 instead of Q4_0)
echo Expected size: ~2.8-3.0 GB (maximum compression)
pause
