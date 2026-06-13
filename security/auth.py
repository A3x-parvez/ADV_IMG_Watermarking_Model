import hashlib
import torch


# =========================================================
# AUTH KEY
# =========================================================

def generate_authentication_key(
    blueprint,
    stego=None,
    secret_key="WMNET_SECURE"
):

    # Avoid moving full tensors to CPU / numpy; use compact deterministic summary
    def tensor_summary(t):
        t = t.detach()
        mean = torch.mean(t).item()
        std = torch.std(t).item()
        mn = torch.min(t).item()
        mx = torch.max(t).item()
        shape = tuple(t.shape)
        return f"{mean:.8f}:{std:.8f}:{mn:.8f}:{mx:.8f}:{shape}".encode()

    payload = tensor_summary(blueprint)

    if stego is not None:

        payload += tensor_summary(stego)

    payload += secret_key.encode()

    auth_key = hashlib.sha256(payload).hexdigest()

    return auth_key


# =========================================================
# VERIFY
# =========================================================

def verify_authentication(
    blueprint,
    received_key,
    stego=None,
    secret_key="WMNET_SECURE"
):

    generated_key = generate_authentication_key(
        blueprint,
        stego,
        secret_key
    )

    return generated_key == received_key