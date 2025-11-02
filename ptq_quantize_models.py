"""
ptq_quantize_models.py
-----------------------------------
Copies Stable Diffusion model files and applies
Post-Training Quantization (FP32 → INT8) to each checkpoint.
"""

import os
import shutil
import torch
from pathlib import Path

from stable_diffusion_pytorch import model_loader
from stable_diffusion_pytorch.encoder import Encoder
from stable_diffusion_pytorch.decoder import Decoder
from stable_diffusion_pytorch.diffusion import Diffusion
from stable_diffusion_pytorch.clip import CLIP

# === 1️⃣ PATH CONFIGURATION ===
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SOURCE_CKPT = DATA_DIR / "ckpt"
DEST_DIR = BASE_DIR / "data_PTQ"
DEST_CKPT = DEST_DIR / "ckpt"
DEST_CKPT_STATE_DICT = DEST_DIR / "ckpt_state_dict"

# Create output folders
DEST_DIR.mkdir(parents=True, exist_ok=True)
DEST_CKPT.mkdir(parents=True, exist_ok=True)
DEST_CKPT_STATE_DICT.mkdir(parents=True, exist_ok=True)

# === 2️⃣ COPY STATIC FILES (merges, vocab) ===
for filename in ["merges.txt", "vocab.json"]:
    src = DATA_DIR / filename
    dst = DEST_DIR / filename
    if src.exists():
        shutil.copy(src, dst)
        print(f"📁 Copied {filename} → {dst}")
    else:
        print(f"⚠️ Missing {filename}, skipping...")

# === 3️⃣ COPY ORIGINAL clip.pt (FP32) ===
print("\n🔄 Copying original clip.pt without quantization...")
src_clip = SOURCE_CKPT / "clip.pt"
dst_clip = DEST_CKPT / "clip.pt"
dst_clip_sd = DEST_CKPT_STATE_DICT / "clip.pt"

# Copy full model
shutil.copy(src_clip, dst_clip)
print(f"✅ Copied clip.pt → {dst_clip}")

# Also extract and save state_dict
clip_model = CLIP()
clip_model.load_state_dict(torch.load(src_clip, map_location="cpu"))
torch.save(clip_model.state_dict(), dst_clip_sd)
print(f"🧾 Saved state_dict for clip.pt → {dst_clip_sd}")

# === 4️⃣ STATIC QUANTIZATION FUNCTION ===
def quantize_static_model(model, dummy_input, out_path: Path, model_name: str):
    print(f"⚙️ Statically quantizing {model_name}...")
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    with torch.no_grad():
        if isinstance(dummy_input, tuple):
            model(*dummy_input)
        else:
            model(dummy_input)

    quantized_model = torch.quantization.convert(model, inplace=False)

    # ✅ Save the quantized model
    torch.save(quantized_model, out_path)

    # ✅ Also save a *dequantized* state_dict for use in notebooks
    dequantized_state_dict = {}
    for k, v in quantized_model.state_dict().items():
        if isinstance(v, torch.Tensor):
            if v.is_quantized:
                dequantized_state_dict[k] = v.dequantize()
            else:
                dequantized_state_dict[k] = v
        else:
            # keep non‑tensor objects (dtype, ints, etc.) as is
            dequantized_state_dict[k] = v



    state_dict_path = DEST_CKPT_STATE_DICT / model_name
    torch.save(dequantized_state_dict, state_dict_path)
    print(f"🧾 Saved *dequantized* state_dict for {model_name} → {state_dict_path}")


# === 5️⃣ DEFINE DUMMY INPUTS FOR CALIBRATION ===
dummy_inputs = {
    "encoder": (torch.randn(1, 3, 512, 512), torch.randn(1, 4, 64, 64)),
    "decoder": torch.randn(1, 4, 64, 64),
    "diffusion": (torch.randn(1, 4, 64, 64),
                  torch.randn(1, 77, 768),
                  torch.randn(1, 320)),
}

# === 6️⃣ APPLY STATIC QUANTIZATION ===

# Encoder
encoder = Encoder()
encoder.load_state_dict(torch.load(SOURCE_CKPT / "encoder.pt", map_location="cpu"))
quantize_static_model(
    encoder,
    dummy_input=dummy_inputs["encoder"],
    out_path=DEST_CKPT / "encoder.pt",
    model_name="encoder.pt"
)

# Decoder
decoder = Decoder()
decoder.load_state_dict(torch.load(SOURCE_CKPT / "decoder.pt", map_location="cpu"))
quantize_static_model(
    decoder,
    dummy_input=dummy_inputs["decoder"],
    out_path=DEST_CKPT / "decoder.pt",
    model_name="decoder.pt"
)

# Diffusion
diffusion = Diffusion()
diffusion.load_state_dict(torch.load(SOURCE_CKPT / "diffusion.pt", map_location="cpu"))
quantize_static_model(
    diffusion,
    dummy_input=dummy_inputs["diffusion"],
    out_path=DEST_CKPT / "diffusion.pt",
    model_name="diffusion.pt"
)

print("\n🎉 Done! decoder.pt, encoder.pt, diffusion.pt quantized. clip.pt copied without quantization.")
