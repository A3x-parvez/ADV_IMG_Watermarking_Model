import torch
import hashlib


# =========================================================
# KEY GENERATION
# =========================================================

def generate_chaos_key(cover):

    value = torch.mean(
        cover
    ).item()

    key = hashlib.sha256(
        str(value).encode()
    ).hexdigest()

    return key


# =========================================================
# LOGISTIC MAP
# =========================================================

def logistic_map(
    size,
    r,
    x0
):

    x = torch.zeros(size)

    x[0] = x0

    for i in range(1, size):

        x[i] = (
            r *
            x[i - 1] *
            (1 - x[i - 1])
        )

    return x


# =========================================================
# HENON MAP
# =========================================================

def henon_map(
    size,
    a,
    b
):

    x = torch.zeros(size)

    y = torch.zeros(size)

    x[0] = 0.1

    y[0] = 0.3

    for i in range(1, size):

        x[i] = (
            1
            - a * x[i - 1] ** 2
            + y[i - 1]
        )

        y[i] = b * x[i - 1]

    return x


# =========================================================
# TENT MAP
# =========================================================

def tent_map(
    size,
    mu=1.99,
    x0=0.37
):

    x = torch.zeros(size)

    x[0] = x0

    for i in range(1, size):

        if x[i - 1] < 0.5:

            x[i] = mu * x[i - 1]

        else:

            x[i] = mu * (
                1 - x[i - 1]
            )

    return x


# =========================================================
# HYBRID CHAOS GENERATION
# =========================================================

def generate_hybrid_chaos(
    batch_size,
    h,
    w,
    key
):

    total = h * w

    seed = int(
        key[:8],
        16
    )

    generator = torch.Generator()

    generator.manual_seed(seed)

    r = 3.9 + torch.rand(
        1,
        generator=generator
    ).item() * 0.09

    a = 1.3 + torch.rand(
        1,
        generator=generator
    ).item() * 0.2

    b = 0.2 + torch.rand(
        1,
        generator=generator
    ).item() * 0.2

    x0 = torch.rand(
        1,
        generator=generator
    ).item()

    logistic = logistic_map(
        total,
        r,
        x0
    )

    henon = henon_map(
        total,
        a,
        b
    )

    tent = tent_map(
        total
    )

    # =====================================================
    # HYBRID FUSION
    # =====================================================

    chaos = (
        0.4 * logistic
        +
        0.4 * henon
        +
        0.2 * tent
    )

    chaos = (
        chaos - chaos.min()
    ) / (
        chaos.max() - chaos.min() + 1e-8
    )

    chaos = chaos.reshape(
        1,
        1,
        h,
        w
    )

    # =====================================================
    # BATCH REPLICATION
    # =====================================================

    chaos = chaos.repeat(
        batch_size,
        1,
        1,
        1
    )

    return chaos


# =========================================================
# CHAOTIC PERMUTATION
# =========================================================

def chaotic_permutation(
    watermark,
    chaos
):

    b, c, h, w = watermark.shape

    permuted_batches = []

    for i in range(b):

        single_watermark = watermark[i].reshape(-1)

        single_chaos = chaos[i].reshape(-1)

        indices = torch.argsort(
            single_chaos
        )

        permuted = single_watermark[
            indices
        ]

        permuted = permuted.reshape(
            c,
            h,
            w
        )

        permuted_batches.append(
            permuted
        )

    return torch.stack(
        permuted_batches,
        dim=0
    )


# =========================================================
# REVERSE PERMUTATION
# =========================================================

def reverse_permutation(
    image,
    chaos
):

    b, c, h, w = image.shape

    recovered_batches = []

    for i in range(b):

        single_image = image[i].reshape(-1)

        single_chaos = chaos[i].reshape(-1)

        indices = torch.argsort(
            single_chaos
        )

        recovered = torch.zeros_like(
            single_image
        )

        recovered[indices] = single_image

        recovered = recovered.reshape(
            c,
            h,
            w
        )

        recovered_batches.append(
            recovered
        )

    return torch.stack(
        recovered_batches,
        dim=0
    )


# =========================================================
# XOR DIFFUSION
# =========================================================

def xor_diffusion(
    image,
    chaos
):

    binary_chaos = (
        chaos > 0.5
    ).float()

    encrypted = torch.logical_xor(
        image.bool(),
        binary_chaos.bool()
    ).float()

    return encrypted


# =========================================================
# FULL HYBRID ENCRYPTION
# =========================================================

def hybrid_encrypt(
    watermark,
    cover
):

    b, c, h, w = watermark.shape

    # =====================================================
    # KEY GENERATION
    # =====================================================

    key = generate_chaos_key(
        cover
    )

    # =====================================================
    # CHAOS MAP
    # =====================================================

    chaos = generate_hybrid_chaos(
        b,
        h,
        w,
        key
    ).to(watermark.device)

    # =====================================================
    # SCRAMBLING
    # =====================================================

    scrambled = chaotic_permutation(
        watermark,
        chaos
    )

    # =====================================================
    # XOR ENCRYPTION
    # =====================================================

    encrypted = xor_diffusion(
        scrambled,
        chaos
    )

    return encrypted, key


# =========================================================
# FULL HYBRID DECRYPTION
# =========================================================

def hybrid_decrypt(
    encrypted,
    cover,
    key=None
):

    b, c, h, w = encrypted.shape

    # =====================================================
    # KEY REGENERATION
    # =====================================================

    if key is None:

        key = generate_chaos_key(
            cover
        )

    # =====================================================
    # SAME CHAOS
    # =====================================================

    chaos = generate_hybrid_chaos(
        b,
        h,
        w,
        key
    ).to(encrypted.device)

    # =====================================================
    # REVERSE XOR
    # =====================================================

    descrambled = xor_diffusion(
        encrypted,
        chaos
    )

    # =====================================================
    # REVERSE PERMUTATION
    # =====================================================

    recovered = reverse_permutation(
        descrambled,
        chaos
    )

    return recovered