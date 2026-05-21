import os
import random
import numpy as np

import torch
import matplotlib.pyplot as plt

from torchvision.utils import save_image


# =========================================================
# RANDOM SEED
# =========================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


# =========================================================
# CREATE DIRECTORY
# =========================================================

def create_dir(path):

    os.makedirs(
        path,
        exist_ok=True
    )


# =========================================================
# SAFE IMAGE
# =========================================================

def safe_image(x):

    return torch.clamp(
        x,
        0,
        1
    )


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(model, path):

    torch.save(
        model.state_dict(),
        path
    )

    print(f"Saved: {path}")


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(model, path, device):

    model.load_state_dict(
        torch.load(
            path,
            map_location=device
        )
    )

    print(f"Loaded: {path}")

    return model


# =========================================================
# DIFFERENCE MAP
# =========================================================

def generate_difference_map(
    cover,
    stego
):

    diff = torch.abs(
        cover - stego
    )

    diff = diff / (
        diff.max() + 1e-8
    )

    return diff


# =========================================================
# SAVE VISUALIZATION
# =========================================================

def save_visualization(

    cover,
    watermark,
    stego,
    attacked_stego,
    recovered_cover,
    recovered_watermark,
    save_path,
    psnr=None,
    ssim=None,
    ber=None
):

    cover = safe_image(cover)
    stego = safe_image(stego)
    attacked_stego = safe_image(attacked_stego)
    recovered_cover = safe_image(recovered_cover)

    cover_img = cover[0][0].detach().cpu().numpy()

    watermark_img = watermark[0][0].detach().cpu().numpy()

    stego_img = stego[0][0].detach().cpu().numpy()

    attacked_img = attacked_stego[0][0].detach().cpu().numpy()

    recovered_cover_img = \
        recovered_cover[0][0].detach().cpu().numpy()

    recovered_wm_img = torch.sigmoid(
        recovered_watermark
    )[0][0].detach().cpu().numpy()

    recovered_wm_img = (
        recovered_wm_img > 0.5
    ).astype(float)

    diff_map = np.abs(
        cover_img - stego_img
    )

    overlay = np.clip(
        0.7 * cover_img +
        0.3 * diff_map,
        0,
        1
    )

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 4, 1)
    plt.imshow(cover_img, cmap="gray")
    plt.title("Cover")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(watermark_img, cmap="gray")
    plt.title("Original Watermark")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(stego_img, cmap="gray")

    title = "Stego"

    if psnr is not None:
        title += f"\nPSNR={psnr:.2f}"

    plt.title(title)
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(diff_map, cmap="hot")
    plt.title("Difference Map")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(attacked_img, cmap="gray")
    plt.title("Attacked Stego")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(recovered_cover_img, cmap="gray")
    plt.title("Recovered Cover")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(recovered_wm_img, cmap="gray")

    title = "Recovered Watermark"

    if ber is not None:
        title += f"\nBER={ber:.6f}"

    plt.title(title)
    plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.imshow(overlay, cmap="gray")

    title = "Embedding Overlay"

    if ssim is not None:
        title += f"\nSSIM={ssim:.4f}"

    plt.title(title)

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# SAVE IMAGE
# =========================================================

def save_stego_image(stego, path):

    save_image(
        stego,
        path
    )


# =========================================================
# PARAM COUNT
# =========================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# =========================================================
# MODEL INFO
# =========================================================

def print_model_info(
    encoder,
    decoder
):

    enc_params = count_parameters(
        encoder
    )

    dec_params = count_parameters(
        decoder
    )

    total = enc_params + dec_params

    print("\n===================================")

    print("MODEL INFORMATION")

    print("===================================")

    print(f"Encoder Params : {enc_params:,}")

    print(f"Decoder Params : {dec_params:,}")

    print(f"Total Params   : {total:,}")

    print("===================================\n")


# =========================================================
# DEVICE INFO
# =========================================================

def print_device_info(device):

    print("\n===================================")

    print("DEVICE INFORMATION")

    print("===================================")

    print(f"Using Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        memory = torch.cuda.get_device_properties(
            0
        ).total_memory / 1024**3

        print(f"GPU Memory: {memory:.2f} GB")

    print("===================================\n")