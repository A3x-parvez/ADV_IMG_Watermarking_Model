import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder

from attacks import AttackLayer

from security.encryption import (
    encrypt_blueprint,
    decrypt_blueprint
)

from security.auth import (
    verify_authentication
)


class WatermarkSystem(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = Encoder()

        self.attack_layer = AttackLayer()

        self.decoder = Decoder()

    def forward(

        self,

        cover,

        watermark,

        apply_attack=True
    ):

        # =================================================
        # ENCODER
        # =================================================

        stego, blueprint, auth_key = \
            self.encoder(
                cover,
                watermark
            )

        # =================================================
        # ENCRYPT BLUEPRINT
        # =================================================

        encrypted_blueprint, blueprint_key = \
            encrypt_blueprint(
                blueprint
            )

        # =================================================
        # ATTACK SIMULATION
        # =================================================

        if apply_attack:

            attacked_stego = self.attack_layer(
                stego
            )

        else:

            attacked_stego = stego

        # =================================================
        # DECRYPT BLUEPRINT
        # =================================================

        decrypted_blueprint = decrypt_blueprint(

            encrypted_blueprint,

            blueprint_key
        )

        # =================================================
        # AUTHENTICATION VERIFY
        # =================================================

        auth_valid = verify_authentication(

            decrypted_blueprint,

            auth_key
        )

        # =================================================
        # DECODER
        # =================================================

        recovered_cover, recovered_watermark = \
            self.decoder(

                attacked_stego,

                decrypted_blueprint
            )

        return (

            stego,

            attacked_stego,

            encrypted_blueprint,

            recovered_cover,

            recovered_watermark,

            auth_key,

            auth_valid
        )