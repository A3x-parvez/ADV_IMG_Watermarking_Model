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

    blueprint_data = blueprint.detach() \
        .cpu() \
        .numpy() \
        .tobytes()

    payload = blueprint_data

    if stego is not None:

        stego_data = stego.detach() \
            .cpu() \
            .numpy() \
            .tobytes()

        payload += stego_data

    payload += secret_key.encode()

    auth_key = hashlib.sha256(
        payload
    ).hexdigest()

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