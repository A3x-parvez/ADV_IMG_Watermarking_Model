import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ssim

from config import (
    STEGO_LOSS_WEIGHT,
    COVER_LOSS_WEIGHT,
    WATERMARK_LOSS_WEIGHT,
    BLUEPRINT_LOSS_WEIGHT,
    EDGE_LOSS_WEIGHT,
    SSIM_LOSS_WEIGHT
)

# =========================================================
# BASIC LOSSES
# =========================================================

mse = nn.MSELoss()

l1 = nn.L1Loss()

bce = nn.BCEWithLogitsLoss()


# =========================================================
# GLOBAL SOBEL FILTERS
# =========================================================

sobel_x = torch.tensor(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]],
    dtype=torch.float32
).view(1, 1, 3, 3)

sobel_y = torch.tensor(
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]],
    dtype=torch.float32
).view(1, 1, 3, 3)


# =========================================================
# EDGE LOSS
# =========================================================

def edge_loss(img1, img2):

    sx = sobel_x.to(img1.device)

    sy = sobel_y.to(img1.device)

    edge1_x = F.conv2d(
        img1,
        sx,
        padding=1
    )

    edge1_y = F.conv2d(
        img1,
        sy,
        padding=1
    )

    edge2_x = F.conv2d(
        img2,
        sx,
        padding=1
    )

    edge2_y = F.conv2d(
        img2,
        sy,
        padding=1
    )

    edge1 = torch.sqrt(
        edge1_x ** 2 +
        edge1_y ** 2 +
        1e-8
    )

    edge2 = torch.sqrt(
        edge2_x ** 2 +
        edge2_y ** 2 +
        1e-8
    )

    return l1(edge1, edge2)


# =========================================================
# SSIM LOSS
# =========================================================

def ssim_loss(img1, img2):

    return 1 - ssim(
        img1,
        img2,
        data_range=1.0,
        size_average=True
    )


# =========================================================
# TOTAL LOSS
# =========================================================

def total_loss(
    cover,
    stego,
    recovered_cover,
    watermark,
    recovered_watermark,
    blueprint
):

    # =====================================================
    # STEGO INVISIBILITY
    # =====================================================

    stego_mse = mse(
        stego,
        cover
    )

    stego_ssim = ssim_loss(
        stego,
        cover
    )

    stego_edge = edge_loss(
        stego,
        cover
    )

    stego_loss = (

        STEGO_LOSS_WEIGHT * stego_mse

        +

        SSIM_LOSS_WEIGHT * stego_ssim

        +

        EDGE_LOSS_WEIGHT * stego_edge
    )

    # =====================================================
    # COVER RECONSTRUCTION
    # =====================================================

    cover_loss = (

        l1(recovered_cover, cover)

        +

        mse(recovered_cover, cover)
    )

    # =====================================================
    # WATERMARK RECOVERY
    # =====================================================

    watermark_loss = bce(
        recovered_watermark,
        watermark
    )

    # =====================================================
    # BLUEPRINT REGULARIZATION
    # =====================================================

    blueprint_loss = l1(
        blueprint,
        torch.zeros_like(blueprint)
    )

    # =====================================================
    # FINAL TOTAL LOSS
    # =====================================================

    total = (

        2.0 * stego_loss

        +

        COVER_LOSS_WEIGHT * cover_loss

        +

        WATERMARK_LOSS_WEIGHT * watermark_loss

        +

        BLUEPRINT_LOSS_WEIGHT * blueprint_loss
    )

    return total