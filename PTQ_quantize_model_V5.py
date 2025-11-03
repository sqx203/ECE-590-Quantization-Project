import os
import platform
import time
import torch
import copy
from pathlib import Path
from PIL import Image
from torchvision import transforms
from stable_diffusion_pytorch import util, pipeline, model_loader

# =========================================================
#                   CONFIGURATION
# =========================================================
DEVICE = "cpu"  # Quantized INT8 kernels only run on CPU (fbgemm backend)
OUTPUT_DIR = Path("data_PTQ")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPTS = ["a photograph of an astronaut riding a horse"]
CALIBRATE_PROMPTS = PROMPTS

SEED = 42
N_STEPS = 30
SAMPLER = "k_lms"
CFG_SCALE = 7.5


# =========================================================
#                    HELPERS
# =========================================================
def measure_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    size_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    return total_params, size_bytes


# =========================================================
#             FX-BASED POST TRAINING QUANTIZATION
# =========================================================
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

def build_ptq_model_fx(unet_fp32):
    """
    FX-based Post-Training Quantization (works on Windows, PyTorch 2.5+)
    """
    unet = copy.deepcopy(unet_fp32).eval().to("cpu")
    qmap = get_default_qconfig_mapping("x86")

    example_lat = torch.randn(1, 4, 64, 64)
    example_t = util.get_time_embedding(500, torch.float32)
    example_ctx = torch.randn(1, 77, 768)

    prepared = prepare_fx(unet, qmap, example_inputs=(example_lat, example_t, example_ctx))
    print("📏 Calibrating quantization observers …")
    with torch.inference_mode():
        for prompt in CALIBRATE_PROMPTS:
            _ = pipeline.generate(
                [prompt],
                models={"diffusion": prepared},
                seed=SEED,
                n_inference_steps=5,
                sampler=SAMPLER,
                device="cpu"
            )

    quantized = convert_fx(prepared)
    print("✅ FX quantization complete.")
    return quantized


# =========================================================
#              PT2E QUANTIZATION (Linux only)
# =========================================================
def build_ptq_model_pt2e(unet_fp32):
    """
    PT2E quantization (requires Linux/WSL, PyTorch 2.5+)
    """
    import torch._export
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer

    unet_fp32 = unet_fp32.eval().to("cpu")

    print("🔧 Capturing pre-autograd graph for Diffusion model …")
    example_lat = torch.randn(1, 4, 64, 64)
    example_t = util.get_time_embedding(500, torch.float32)
    example_ctx = torch.randn(1, 77, 768)
    example_inputs = (example_lat, example_t, example_ctx)

    # Capture model graph
    exported_model = torch._export.capture_pre_autograd_graph(unet_fp32, example_inputs)

    quantizer = X86InductorQuantizer()
    quantizer.set_global(None)

    prepared = prepare_pt2e(exported_model, quantizer)
    print("📏 Calibrating PT2E observers …")
    with torch.inference_mode():
        for prompt in CALIBRATE_PROMPTS:
            _ = pipeline.generate(
                [prompt],
                models={"diffusion": prepared},
                seed=SEED,
                n_inference_steps=5,
                sampler=SAMPLER,
                device="cpu",
            )

    quantized = convert_pt2e(prepared)
    print("✅ PT2E quantization complete.")
    return quantized


# =========================================================
#                        MAIN
# =========================================================
def main():
    torch.manual_seed(SEED)
    models = model_loader.preload_models("cpu")

    os_type = platform.system().lower()
    if "windows" in os_type:
        print("🪟 Windows detected → Using FX-based PTQ.")
        unet_q = build_ptq_model_fx(models["diffusion"])
    else:
        print("🐧 Linux detected → Using PT2E quantization.")
        unet_q = build_ptq_model_pt2e(models["diffusion"])

    ckpt_dir = OUTPUT_DIR / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save quantized model
    torch.save(unet_q.state_dict(), "data_PTQ/ckpt/diffusion_int8_state.pt")
    torch.save(unet_q, "data_PTQ/ckpt/diffusion_int8_full.pt", _use_new_zipfile_serialization=True)
    # scripted = torch.jit.script(unet_q)
    # torch.jit.save(scripted, "data_PTQ/ckpt/diffusion_int8_torchscript.pt")

    print(f"✅ Saved quantized model to {ckpt_dir/'diffusion_int8.pt'}")

    for name, mod in unet_q.named_modules():
        if "quantized" in str(type(mod)).lower():
            print(f"✅ Quantized module: {name} → {type(mod)}")
    
    # Report model stats
    params, size = measure_model_stats(unet_q)
    print(f"📊 Params: {params/1e6:.1f}M | Size: {size/1e6:.2f} MB")

if __name__ == "__main__":
    main()
