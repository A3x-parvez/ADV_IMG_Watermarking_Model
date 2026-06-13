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

    # Legacy: replaced by randomized generator in hybrid fusion
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

    # Legacy: keep behavior but not used in vectorized hybrid generation
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

    # Legacy: keep behavior but not used in vectorized hybrid generation
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

    # Vectorized, deterministic pseudo-chaos using seeded RNGs to avoid Python loops
    total = h * w

    seed = int(key[:8], 16)

    g1 = torch.Generator(); g1.manual_seed(seed)
    g2 = torch.Generator(); g2.manual_seed(seed ^ 0x9e3779b9)
    g3 = torch.Generator(); g3.manual_seed((seed << 13) & 0xffffffff)

    logistic = torch.rand(total, generator=g1)
    henon = torch.rand(total, generator=g2)
    tent = torch.rand(total, generator=g3)

    chaos = (0.4 * logistic + 0.4 * henon + 0.2 * tent)

    chaos = (chaos - chaos.min()) / (chaos.max() - chaos.min() + 1e-8)

    chaos = chaos.reshape(1, 1, h, w)

    chaos = chaos.repeat(batch_size, 1, 1, 1)

    return chaos


# =========================================================
# CHAOTIC PERMUTATION
# =========================================================

def chaotic_permutation(
    watermark,
    chaos
):

    b, c, h, w = watermark.shape

    n = c * h * w

    flat = watermark.view(b, n)

    chaos_flat = chaos.view(b, -1)

    indices = torch.argsort(chaos_flat, dim=1)

    permuted_flat = torch.gather(flat, 1, indices)

    return permuted_flat.view(b, c, h, w)


# =========================================================
# REVERSE PERMUTATION
# =========================================================

def reverse_permutation(
    image,
    chaos
):

    b, c, h, w = image.shape

    n = c * h * w

    flat = image.view(b, n)

    chaos_flat = chaos.view(b, -1)

    indices = torch.argsort(chaos_flat, dim=1)

    # inverse permutation: for each row produce indices_inv such that indices_inv[indices[row]] = arange
    arange = torch.arange(n, device=indices.device).unsqueeze(0).expand(b, n)

    inv = torch.zeros_like(indices)

    inv.scatter_(1, indices, arange)

    recovered_flat = torch.gather(flat, 1, inv)

    return recovered_flat.view(b, c, h, w)


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