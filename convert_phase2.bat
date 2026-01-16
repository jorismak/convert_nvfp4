@echo off
REM Phase 2: Quantize V projections to NVFP4, keep FFN.2 at BF16
REM Run this if Phase 1 (quant-test) produces clean output

echo Converting WAN 2.2 TI2V 5B with Phase 2 preset (quantize V projections)...

python convert_nvfp4.py ^
    "D:\comfy2\ComfyUI\nvfp4-conv\wan2.2-ti2v-5b\diffusion_pytorch_model.safetensors.index.json" ^
    "D:\ComfyUI\ComfyUI\models\diffusion_models\wan2.2-ti2v-5b-nvfp4-phase2.safetensors" ^
    --preset quant-test-phase2 ^
    --mode all

echo.
echo Conversion complete!
echo Output: D:\ComfyUI\ComfyUI\models\diffusion_models\wan2.2-ti2v-5b-nvfp4-phase2.safetensors
echo.
echo Next: Test in ComfyUI and compare quality to Phase 1
pause
