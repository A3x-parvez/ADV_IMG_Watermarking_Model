import os
import torch


VISUALIZE_ON = False

# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

USE_CUDA = torch.cuda.is_available()

# =========================================================
# RANDOM SEED
# =========================================================

SEED = 42

# =========================================================
# DATASET
# =========================================================

IMAGE_SIZE = 128

CHANNELS = 1

BATCH_SIZE = 2

NUM_WORKERS = 4

PIN_MEMORY = True

DROP_LAST = False

# =========================================================
# TRAINING
# =========================================================

EPOCHS = 100

LEARNING_RATE = 1e-4

MIN_LEARNING_RATE = 1e-6

WEIGHT_DECAY = 1e-5

GRADIENT_CLIP = 1.0

USE_AMP = True

SAVE_EVERY = 5

VISUALIZE_EVERY = 2

VALIDATION_EVERY = 1

SCHEDULER_PATIENCE = 5

SCHEDULER_FACTOR = 0.5

EARLY_STOPPING_PATIENCE = 20

# =========================================================
# MODEL
# =========================================================

BASE_CHANNELS = 32

BLUEPRINT_CHANNELS = 16

DROPOUT = 0.05

USE_CHECKPOINTING = False

# =========================================================
# SECURITY
# =========================================================

SECRET_KEY = "WMNET_SECURE"

# =========================================================
# CHAOTIC SYSTEM
# =========================================================

CHAOS_LOGISTIC_R = 3.99

CHAOS_HENON_A = 1.4

CHAOS_HENON_B = 0.3

CHAOS_TENT_MU = 1.99

CHAOS_X0 = 0.51

# =========================================================
# ATTACK PROBABILITIES
# =========================================================

CLEAN_PROB = 0.40

JPEG_PROB = 0.20

GAUSSIAN_PROB = 0.15

BLUR_PROB = 0.10

CROP_PROB = 0.10

ROTATION_PROB = 0.05

SALT_PEPPER_PROB = 0.10

MULTI_ATTACK_PROB = 0.30

# =========================================================
# ROBUSTNESS SETTINGS
# =========================================================

JPEG_QUALITY = 50

JPEG_QUALITY_MIN = 30

JPEG_QUALITY_MAX = 90

GAUSSIAN_STD = 0.02

GAUSSIAN_STD_MAX = 0.05

ROTATION_ANGLE = 15

CROP_RATIO = 0.80

BLUR_KERNEL = 3

# =========================================================
# LOSS WEIGHTS
# =========================================================

STEGO_LOSS_WEIGHT = 2.0

COVER_LOSS_WEIGHT = 5.0

WATERMARK_LOSS_WEIGHT = 8.0

BLUEPRINT_LOSS_WEIGHT = 0.1

EDGE_LOSS_WEIGHT = 1.0

SSIM_LOSS_WEIGHT = 2.0

PERCEPTUAL_LOSS_WEIGHT = 0.5

ATTACK_ROBUSTNESS_WEIGHT = 3.0

# =========================================================
# PATHS
# =========================================================

TRAIN_COVER_DIR = (
    "train_data/cover"
)

TRAIN_WATERMARK_DIR = (
    "train_data/watermark"
)

TEST_COVER_DIR = (
    "test_data/cover"
)

TEST_WATERMARK_DIR = (
    "test_data/watermark"
)

RESULTS_DIR = "results"

CHECKPOINT_DIR = "checkpoints"

VISUALS_DIR = "visualizations"

METRICS_DIR = "metrics"

LOG_DIR = "logs"

ATTACK_RESULTS_DIR = (
    "attack_results"
)

# =========================================================
# CHECKPOINT FILES
# =========================================================

ENCODER_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "encoder_best.pth"
)

DECODER_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "decoder_best.pth"
)

FULL_MODEL_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "full_model_best.pth"
)

# =========================================================
# VISUALIZATION
# =========================================================

SAVE_DIFFERENCE_MAP = True

SAVE_ATTACK_RESULTS = True

SAVE_RECOVERED_IMAGES = True

SAVE_BLUEPRINT_VIS = False

# =========================================================
# METRICS
# =========================================================

METRIC_DECIMALS = 4

EPSILON = 1e-8

MAX_PSNR = 100.0

# =========================================================
# TESTING
# =========================================================

TEST_BATCH_SIZE = 1

RUN_ATTACK_BENCHMARK = True

SAVE_TEST_IMAGES = True

# =========================================================
# SECURITY VALIDATION
# =========================================================

VERIFY_AUTHENTICATION = True

VERIFY_BLUEPRINT = True

VERIFY_CHAOS_REVERSIBILITY = True

# =========================================================
# AUTOMATIC DIRECTORY CREATION
# =========================================================

DIRECTORIES = [

    RESULTS_DIR,

    CHECKPOINT_DIR,

    VISUALS_DIR,

    METRICS_DIR,

    LOG_DIR,

    ATTACK_RESULTS_DIR
]

for directory in DIRECTORIES:

    os.makedirs(
        directory,
        exist_ok=True
    )

# =========================================================
# DEVICE INFO
# =========================================================

print("\n===================================")
print("CONFIG LOADED")
print("===================================")

print(f"DEVICE            : {DEVICE}")
print(f"IMAGE_SIZE        : {IMAGE_SIZE}")
print(f"BATCH_SIZE        : {BATCH_SIZE}")
print(f"EPOCHS            : {EPOCHS}")
print(f"BASE_CHANNELS     : {BASE_CHANNELS}")
print(f"BLUEPRINT_CHANNELS: {BLUEPRINT_CHANNELS}")

print("===================================\n")