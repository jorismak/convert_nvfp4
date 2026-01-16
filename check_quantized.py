from safetensors import safe_open
import json

f = safe_open(
    "D:/ComfyUI/ComfyUI/models/diffusion_models/wan2.2-ti2v-5b-nvfp4-quant-test-fixed.safetensors",
    framework="pt",
    device="cpu",
)
meta = f.metadata()
qmeta = json.loads(meta.get("_quantization_metadata", "{}"))
layers = qmeta.get("layers", {})

v_quant = [k for k in layers.keys() if ".v.weight" in k]
ffn2_quant = [k for k in layers.keys() if ".ffn.2.weight" in k]

print(f"Total quantized: {len(layers)}")
print(f"V projections quantized: {len(v_quant)}")
print(f"FFN.2 quantized: {len(ffn2_quant)}")
print(f"\nFirst 3 V projections:")
for k in sorted(v_quant)[:3]:
    print(f"  {k}")
print(f"\nFirst 3 FFN.2:")
for k in sorted(ffn2_quant)[:3]:
    print(f"  {k}")
