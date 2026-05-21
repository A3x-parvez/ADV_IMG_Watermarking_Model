import torch
import hashlib


# =========================================================
# BLUEPRINT KEY
# =========================================================

def generate_blueprint_key(
    blueprint
):

    value = torch.mean(
        blueprint
    ).item()

    key = hashlib.sha256(
        str(value).encode()
    ).hexdigest()

    return key


# =========================================================
# CHAOTIC MASK
# =========================================================

def generate_chaotic_mask(
    shape,
    key
):

    total = 1

    for s in shape:
        total *= s

    seed = int(
        key[:8],
        16
    )

    generator = torch.Generator()

    generator.manual_seed(seed)

    x = torch.zeros(total)

    r = 3.99

    x[0] = 0.51

    for i in range(1, total):

        x[i] = (
            r *
            x[i - 1] *
            (1 - x[i - 1])
        )

    x = (
        x - x.min()
    ) / (
        x.max() - x.min() + 1e-8
    )

    return x.reshape(shape)


# =========================================================
# PERMUTATION
# =========================================================

def chaotic_permutation(
    tensor,
    mask
):

    flat = tensor.reshape(-1)

    indices = torch.argsort(
        mask.reshape(-1)
    )

    permuted = flat[indices]

    return permuted.reshape_as(
        tensor
    )


# =========================================================
# REVERSE PERMUTATION
# =========================================================

def reverse_permutation(
    tensor,
    mask
):

    flat = tensor.reshape(-1)

    indices = torch.argsort(
        mask.reshape(-1)
    )

    recovered = torch.zeros_like(flat)

    recovered[indices] = flat

    return recovered.reshape_as(
        tensor
    )


# =========================================================
# ENCRYPT BLUEPRINT
# =========================================================

def encrypt_blueprint(
    blueprint
):

    key = generate_blueprint_key(
        blueprint
    )

    mask = generate_chaotic_mask(
        blueprint.shape,
        key
    ).to(blueprint.device)

    permuted = chaotic_permutation(
        blueprint,
        mask
    )

    encrypted = permuted + mask

    return encrypted, key


# =========================================================
# DECRYPT BLUEPRINT
# =========================================================

def decrypt_blueprint(
    encrypted_blueprint,
    key
):

    mask = generate_chaotic_mask(
        encrypted_blueprint.shape,
        key
    ).to(encrypted_blueprint.device)

    permuted = (
        encrypted_blueprint - mask
    )

    recovered = reverse_permutation(
        permuted,
        mask
    )

    return recovered