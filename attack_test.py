import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from config import (
    TEST_COVER_DIR,
    TEST_WATERMARK_DIR,
    DEVICE
)

from dataset import WatermarkDataset

from models.encoder import Encoder
from models.decoder import Decoder

from attacks import AttackLayer

from metrics import *

from security.dna import dna_encode
from security.chaos import (
    hybrid_encrypt,
    hybrid_decrypt
)


device = DEVICE

dataset = WatermarkDataset(
    TEST_COVER_DIR,
    TEST_WATERMARK_DIR
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

encoder = Encoder().to(device)

decoder = Decoder().to(device)

encoder.load_state_dict(
    torch.load(
        "checkpoints/encoder_best.pth",
        map_location=device
    )
)

decoder.load_state_dict(
    torch.load(
        "checkpoints/decoder_best.pth",
        map_location=device
    )
)

encoder.eval()
decoder.eval()

attack_layer = AttackLayer()

all_ber = []

with torch.no_grad():

    for cover, watermark in tqdm(loader):

        cover = cover.to(device)

        watermark = watermark.to(device)

        watermark = dna_encode(
            watermark
        )

        encrypted_wm, chaos_key = \
            hybrid_encrypt(watermark)

        stego, blueprint, auth_key = encoder(
            cover,
            encrypted_wm
        )

        attacked = attack_layer(
            stego
        )

        recovered_cover, recovered_watermark = \
            decoder(
                attacked,
                blueprint
            )

        recovered_prob = torch.sigmoid(
            recovered_watermark
        )

        decrypted = hybrid_decrypt(
            (recovered_prob > 0.5).float()
        )

        ber = calculate_ber(
            decrypted,
            watermark
        )

        all_ber.append(ber)

avg_ber = sum(all_ber) / len(all_ber)

print("\n===================================")

print("ATTACK ROBUSTNESS TEST")

print("===================================")

print(f"Average BER: {avg_ber:.6f}")

print(
    f"Robustness Score: {(1-avg_ber)*100:.2f}%"
)

print("===================================")