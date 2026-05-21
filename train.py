import os
import csv
import torch

from tqdm import tqdm

from torch.utils.data import DataLoader

from torch.cuda.amp import (
    autocast,
    GradScaler
)

# =========================================================
# CONFIG
# =========================================================

from config import (

    DEVICE,

    TRAIN_COVER_DIR,
    TRAIN_WATERMARK_DIR,

    BATCH_SIZE,
    NUM_WORKERS,

    EPOCHS,

    LEARNING_RATE,
    WEIGHT_DECAY,

    SAVE_EVERY,
    VISUALIZE_EVERY,

    CHECKPOINT_DIR,
    VISUALS_DIR,
    METRICS_DIR,

    USE_AMP,
    GRADIENT_CLIP,

    ENCODER_CHECKPOINT,
    DECODER_CHECKPOINT,

    PIN_MEMORY
)

# =========================================================
# DATASET
# =========================================================

from dataset import WatermarkDataset

# =========================================================
# MODEL
# =========================================================

from models.full_model import WatermarkSystem

# =========================================================
# LOSSES
# =========================================================

from losses import total_loss

# =========================================================
# METRICS
# =========================================================

from metrics import (

    calculate_psnr,
    calculate_ssim,
    calculate_ber,
    watermark_accuracy,

    calculate_mse,
    calculate_rmse,
    calculate_mae,

    normalized_correlation,

    calculate_entropy,

    calculate_npcr,
    calculate_uaci
)

# =========================================================
# UTILS
# =========================================================

from utils import (

    create_dir,

    save_visualization,

    print_device_info,

    set_seed
)

# =========================================================
# SECURITY
# =========================================================

from security.dna import (
    dna_encode
)

from security.chaos import (
    hybrid_encrypt
)

# =========================================================
# SEED
# =========================================================

set_seed()

# =========================================================
# DEVICE
# =========================================================

device = DEVICE

print_device_info(device)

# =========================================================
# DIRECTORIES
# =========================================================

create_dir(CHECKPOINT_DIR)

create_dir(VISUALS_DIR)

create_dir(METRICS_DIR)

# =========================================================
# DATASET
# =========================================================

train_dataset = WatermarkDataset(

    TRAIN_COVER_DIR,

    TRAIN_WATERMARK_DIR
)

loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

    drop_last=True
)

# =========================================================
# MODEL
# =========================================================

model = WatermarkSystem().to(device)

# =========================================================
# OPTIONAL PYTORCH COMPILE
# =========================================================

try:

    model = torch.compile(model)

    print("Torch Compile Enabled")

except:

    print("Torch Compile Not Available")

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)

# =========================================================
# SCHEDULER
# =========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode='min',

    factor=0.5,

    patience=5
)

# =========================================================
# AMP
# =========================================================

# scaler = GradScaler(enabled=USE_AMP)
scaler = GradScaler(
    "cuda",
    enabled=USE_AMP
)

# =========================================================
# BEST LOSS
# =========================================================

best_loss = float("inf")

# =========================================================
# CSV LOGGER
# =========================================================

csv_path = os.path.join(
    METRICS_DIR,
    "training_metrics.csv"
)

with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([

        "epoch",

        "loss",

        "psnr",
        "ssim",

        "ber",
        "accuracy",

        "mse",
        "rmse",
        "mae",

        "nc",

        "entropy",

        "npcr",
        "uaci"
    ])

# =========================================================
# TRAINING LOOP
# =========================================================

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    epoch_psnr = 0
    epoch_ssim = 0

    epoch_ber = 0
    epoch_acc = 0

    epoch_mse = 0
    epoch_rmse = 0
    epoch_mae = 0

    epoch_nc = 0

    epoch_entropy = 0

    epoch_npcr = 0
    epoch_uaci = 0

    loop = tqdm(loader)

    for step, (cover, watermark) in enumerate(loop):

        # =================================================
        # DEVICE
        # =================================================

        cover = cover.to(device)

        watermark = watermark.to(device)

        # =================================================
        # DNA ENCODING
        # =================================================

        watermark = dna_encode(
            watermark
        )

        # =================================================
        # CHAOTIC ENCRYPTION
        # =================================================

        encrypted_watermark, chaos_key = \
            hybrid_encrypt(
                watermark,
                cover
            )

        # =================================================
        # ZERO GRAD
        # =================================================

        optimizer.zero_grad(
            set_to_none=True
        )

        # =================================================
        # FORWARD
        # =================================================

        # with autocast(enabled=USE_AMP):
        with autocast("cuda",enabled=USE_AMP):

            (

                stego,

                attacked_stego,

                blueprint,

                recovered_cover,

                recovered_watermark,

                auth_key

            ) = model(

                cover,

                encrypted_watermark,

                apply_attack=True
            )

            # =============================================
            # LOSS
            # =============================================

            loss = total_loss(

                cover,

                stego,

                recovered_cover,

                encrypted_watermark,

                recovered_watermark,

                blueprint
            )

        # =================================================
        # NAN CHECK
        # =================================================

        if torch.isnan(loss):

            print("NaN Loss Detected")

            continue

        # =================================================
        # BACKPROP
        # =================================================

        scaler.scale(loss).backward()

        # =================================================
        # UN-SCALE BEFORE CLIP
        # =================================================

        scaler.unscale_(optimizer)

        # =================================================
        # GRADIENT CLIPPING
        # =================================================

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            GRADIENT_CLIP
        )

        # =================================================
        # OPTIMIZER STEP
        # =================================================

        scaler.step(optimizer)

        scaler.update()

        # =================================================
        # METRICS
        # =================================================

        recovered_prob = torch.sigmoid(
            recovered_watermark
        )

        psnr = calculate_psnr(
            stego,
            cover
        )

        ssim = calculate_ssim(
            stego,
            cover
        )

        ber = calculate_ber(
            recovered_prob,
            encrypted_watermark
        )

        acc = watermark_accuracy(
            recovered_prob,
            encrypted_watermark
        )

        mse = calculate_mse(
            stego,
            cover
        )

        rmse = calculate_rmse(
            stego,
            cover
        )

        mae = calculate_mae(
            stego,
            cover
        )

        nc = normalized_correlation(
            recovered_prob,
            encrypted_watermark
        )

        entropy = calculate_entropy(
            stego
        )

        npcr = calculate_npcr(
            cover,
            stego
        )

        uaci = calculate_uaci(
            cover,
            stego
        )

        # =================================================
        # ACCUMULATION
        # =================================================

        epoch_loss += loss.item()

        epoch_psnr += psnr
        epoch_ssim += ssim

        epoch_ber += ber
        epoch_acc += acc

        epoch_mse += mse
        epoch_rmse += rmse
        epoch_mae += mae

        epoch_nc += nc

        epoch_entropy += entropy

        epoch_npcr += npcr
        epoch_uaci += uaci

        # =================================================
        # PROGRESS BAR
        # =================================================

        loop.set_postfix(

            loss=f"{loss.item():.4f}",

            psnr=f"{psnr:.2f}",

            acc=f"{acc:.2f}%"
        )

        # =================================================
        # VISUALIZATION
        # =================================================

        if step == 0 and epoch % VISUALIZE_EVERY == 0:

            save_visualization(

                cover,

                encrypted_watermark,

                stego,

                attacked_stego,

                recovered_cover,

                recovered_watermark,

                os.path.join(

                    VISUALS_DIR,

                    f"epoch_{epoch+1}.png"
                ),

                psnr=psnr,

                ssim=ssim,

                ber=ber
            )

    # =====================================================
    # AVERAGES
    # =====================================================

    n = len(loader)

    avg_loss = epoch_loss / n

    avg_psnr = epoch_psnr / n
    avg_ssim = epoch_ssim / n

    avg_ber = epoch_ber / n
    avg_acc = epoch_acc / n

    avg_mse = epoch_mse / n
    avg_rmse = epoch_rmse / n
    avg_mae = epoch_mae / n

    avg_nc = epoch_nc / n

    avg_entropy = epoch_entropy / n

    avg_npcr = epoch_npcr / n
    avg_uaci = epoch_uaci / n

    # =====================================================
    # LR SCHEDULER
    # =====================================================

    scheduler.step(avg_loss)

    # =====================================================
    # PRINT
    # =====================================================

    print("\n===================================")

    print(f"Epoch {epoch+1}/{EPOCHS}")

    print(f"Loss     : {avg_loss:.4f}")

    print(f"PSNR     : {avg_psnr:.2f}")

    print(f"SSIM     : {avg_ssim:.4f}")

    print(f"BER      : {avg_ber:.6f}")

    print(f"ACC      : {avg_acc:.2f}%")

    print(f"NC       : {avg_nc:.4f}")

    print(f"Entropy  : {avg_entropy:.4f}")

    print(f"NPCR     : {avg_npcr:.2f}")

    print(f"UACI     : {avg_uaci:.2f}")

    print("===================================\n")

    # =====================================================
    # CSV LOGGING
    # =====================================================

    with open(csv_path, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([

            epoch + 1,

            avg_loss,

            avg_psnr,
            avg_ssim,

            avg_ber,
            avg_acc,

            avg_mse,
            avg_rmse,
            avg_mae,

            avg_nc,

            avg_entropy,

            avg_npcr,
            avg_uaci
        ])

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(

            model.encoder.state_dict(),

            ENCODER_CHECKPOINT
        )

        torch.save(

            model.decoder.state_dict(),

            DECODER_CHECKPOINT
        )

        print("Best Model Saved")

    # =====================================================
    # PERIODIC CHECKPOINTS
    # =====================================================

    if (epoch + 1) % SAVE_EVERY == 0:

        torch.save(

            model.encoder.state_dict(),

            os.path.join(

                CHECKPOINT_DIR,

                f"encoder_epoch_{epoch+1}.pth"
            )
        )

        torch.save(

            model.decoder.state_dict(),

            os.path.join(

                CHECKPOINT_DIR,

                f"decoder_epoch_{epoch+1}.pth"
            )
        )

        print(f"Checkpoint Saved @ Epoch {epoch+1}")

print("Training Complete")