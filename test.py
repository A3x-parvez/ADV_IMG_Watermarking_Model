import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from config import (
    TEST_COVER_DIR,
    TEST_WATERMARK_DIR,
    DEVICE
)

from dataset import WatermarkDataset

from models.encoder import Encoder
from models.decoder import Decoder

from metrics import *

from utils import *

from security.dna import dna_encode
from security.chaos import (
    hybrid_encrypt,
    hybrid_decrypt
)


# =========================================================
# DEVICE
# =========================================================

device = DEVICE

print_device_info(device)

create_dir("test_results")

# =========================================================
# DATASET
# =========================================================

dataset = WatermarkDataset(
    TEST_COVER_DIR,
    TEST_WATERMARK_DIR
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

# =========================================================
# LOAD MODELS
# =========================================================

encoder = Encoder().to(device)

decoder = Decoder().to(device)

load_model(
    encoder,
    "checkpoints/encoder_best.pth",
    device
)

load_model(
    decoder,
    "checkpoints/decoder_best.pth",
    device
)

encoder.eval()
decoder.eval()

# =========================================================
# TEST
# =========================================================

all_psnr = []

all_ssim = []

all_ber = []

all_acc = []

with torch.no_grad():

    for idx, (cover, watermark) in enumerate(tqdm(loader)):

        cover = cover.to(device)

        watermark = watermark.to(device)

        watermark = dna_encode(
            watermark
        )

        encrypted_wm, chaos_key = \
            hybrid_encrypt(watermark)

        # =================================================
        # ENCODER
        # =================================================

        stego, blueprint, auth_key = encoder(
            cover,
            encrypted_wm
        )

        # =================================================
        # DECODER
        # =================================================

        recovered_cover, recovered_watermark = \
            decoder(
                stego,
                blueprint
            )

        recovered_prob = torch.sigmoid(
            recovered_watermark
        )

        decrypted = hybrid_decrypt(
            (recovered_prob > 0.5).float()
        )

        # =================================================
        # METRICS
        # =================================================

        psnr = calculate_psnr(
            stego,
            cover
        )

        ssim = calculate_ssim(
            stego,
            cover
        )

        ber = calculate_ber(
            decrypted,
            watermark
        )

        acc = watermark_accuracy(
            decrypted,
            watermark
        )

        all_psnr.append(psnr)

        all_ssim.append(ssim)

        all_ber.append(ber)

        all_acc.append(acc)

        # =================================================
        # VISUALIZATION
        # =================================================

        save_visualization(

            cover,

            watermark,

            stego,

            stego,

            recovered_cover,

            recovered_watermark,

            f"test_results/result_{idx}.png",

            psnr=psnr,

            ssim=ssim,

            ber=ber
        )

# =========================================================
# FINAL RESULTS
# =========================================================

print("\n===================================")

print("FINAL TEST RESULTS")

print("===================================")

print(f"PSNR : {sum(all_psnr)/len(all_psnr):.2f}")

print(f"SSIM : {sum(all_ssim)/len(all_ssim):.4f}")

print(f"BER  : {sum(all_ber)/len(all_ber):.6f}")

print(f"ACC  : {sum(all_acc)/len(all_acc):.2f}%")

print("===================================")