import torch
import numpy as np

from torchmetrics.functional.image import \
    structural_similarity_index_measure


# =========================================================
# PSNR
# =========================================================

def calculate_psnr(img1, img2):

    mse = torch.mean(
        (img1 - img2) ** 2
    )

    if mse.item() == 0:
        return 100.0

    psnr = 20 * torch.log10(
        1.0 / torch.sqrt(mse)
    )

    return psnr.item()


# =========================================================
# SSIM
# =========================================================

def calculate_ssim(img1, img2):

    return structural_similarity_index_measure(
        img1,
        img2,
        data_range=1.0
    ).item()


# =========================================================
# WATERMARK ACCURACY
# =========================================================

def watermark_accuracy(pred, target):

    pred = (pred > 0.5).float()

    correct = (
        pred == target
    ).float().mean()

    return correct.item() * 100


# =========================================================
# BER
# =========================================================

def calculate_ber(pred, target):

    pred = (pred > 0.5).float()

    errors = (
        pred != target
    ).float().sum()

    total = target.numel()

    ber = errors / total

    return ber.item()


# =========================================================
# NORMALIZED CORRELATION
# =========================================================

def normalized_correlation(img1, img2):

    numerator = torch.sum(
        img1 * img2
    )

    denominator = torch.sqrt(
        torch.sum(img1 ** 2)
        *
        torch.sum(img2 ** 2)
    )

    nc = numerator / (
        denominator + 1e-8
    )

    return nc.item()


# =========================================================
# MSE
# =========================================================

def calculate_mse(img1, img2):

    return torch.mean(
        (img1 - img2) ** 2
    ).item()


# =========================================================
# RMSE
# =========================================================

def calculate_rmse(img1, img2):

    mse = calculate_mse(
        img1,
        img2
    )

    return np.sqrt(mse)


# =========================================================
# MAE
# =========================================================

def calculate_mae(img1, img2):

    return torch.mean(
        torch.abs(img1 - img2)
    ).item()


# =========================================================
# NPCR
# =========================================================

def calculate_npcr(img1, img2):

    img1 = img1.detach().cpu().numpy()

    img2 = img2.detach().cpu().numpy()

    diff = np.abs(img1 - img2) > 1e-5

    npcr = np.sum(diff) / diff.size

    return npcr * 100


# =========================================================
# UACI
# =========================================================

def calculate_uaci(img1, img2):

    img1 = img1.detach().cpu().numpy()

    img2 = img2.detach().cpu().numpy()

    uaci = np.mean(
        np.abs(img1 - img2)
    )

    return (uaci / 1.0) * 100


# =========================================================
# ENTROPY
# =========================================================

def calculate_entropy(img):

    img = img.detach().cpu().numpy()

    hist, _ = np.histogram(
        img.flatten(),
        bins=256,
        range=(0, 1)
    )

    hist = hist / hist.sum()

    hist = hist[hist > 0]

    entropy = -np.sum(
        hist * np.log2(hist)
    )

    return entropy


# =========================================================
# PIXEL CORRELATION
# =========================================================

def pixel_correlation(img):

    img = img.detach().cpu().numpy().flatten()

    x = img[:-1]
    y = img[1:]

    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0

    correlation = np.corrcoef(
        x,
        y
    )[0, 1]

    return correlation


# =========================================================
# CAPACITY BPP
# =========================================================

def calculate_capacity_bpp(secret, cover):

    bits = secret.numel()

    pixels = (
        cover.shape[-1]
        *
        cover.shape[-2]
    )

    return bits / pixels


# =========================================================
# SECURITY SCORE
# =========================================================

def calculate_security_score(
    entropy,
    npcr,
    uaci
):

    entropy_norm = entropy / 8.0

    score = (

        0.4 * entropy_norm

        +

        0.3 * (npcr / 100)

        +

        0.3 * (uaci / 100)
    )

    return score * 100


# =========================================================
# ROBUSTNESS SCORE
# =========================================================

def calculate_robustness_score(
    ber_list
):

    avg_ber = np.mean(ber_list)

    robustness = 1.0 - avg_ber

    return robustness * 100