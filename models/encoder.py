import torch
import torch.nn as nn

from config import (
    BASE_CHANNELS,
    BLUEPRINT_CHANNELS
)

from models.blocks import (

    ResidualBlock,
    SwinBlock,
    CouplingBlock,
    FeatureFusion,
    EdgeEnhancement
)

from models.attention import (

    CoordinateAttention,
    MDTA
)

from security.auth import (
    generate_authentication_key
)


class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        ch = BASE_CHANNELS

        # =====================================================
        # COVER FEATURE STEM
        # =====================================================

        self.cover_stem = nn.Sequential(

            nn.Conv2d(
                1,
                ch,
                3,
                padding=1
            ),

            ResidualBlock(ch),

            ResidualBlock(ch)
        )

        # =====================================================
        # WATERMARK FEATURE STEM
        # =====================================================

        self.watermark_stem = nn.Sequential(

            nn.Conv2d(
                1,
                ch,
                3,
                padding=1
            ),

            ResidualBlock(ch),

            ResidualBlock(ch)
        )

        # =====================================================
        # FEATURE FUSION
        # =====================================================

        self.fusion = FeatureFusion(ch)

        # =====================================================
        # DEEP EMBEDDING
        # =====================================================

        self.embedding = nn.Sequential(

            ResidualBlock(ch),

            MDTA(ch),

            SwinBlock(ch),

            ResidualBlock(ch),

            CoordinateAttention(ch),

            SwinBlock(ch),

            CouplingBlock(ch),

            EdgeEnhancement(ch)
        )

        # =====================================================
        # STEGO IMAGE HEAD
        # =====================================================

        self.stego_head = nn.Sequential(

            nn.Conv2d(
                ch,
                ch // 2,
                3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                ch // 2,
                1,
                1
            ),

            nn.Sigmoid()
        )

        # =====================================================
        # BLUEPRINT GENERATOR
        # =====================================================

        self.blueprint_head = nn.Sequential(

            nn.Conv2d(
                ch,
                ch,
                3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                ch,
                BLUEPRINT_CHANNELS,
                1
            )
        )

    def forward(self, cover, watermark):

        # =================================================
        # FEATURE EXTRACTION
        # =================================================

        cover_feat = self.cover_stem(
            cover
        )

        watermark_feat = self.watermark_stem(
            watermark
        )

        # =================================================
        # FEATURE FUSION
        # =================================================

        x = self.fusion(
            cover_feat,
            watermark_feat
        )

        # =================================================
        # DEEP EMBEDDING
        # =================================================

        x = self.embedding(x)

        # =================================================
        # OUTPUTS
        # =================================================

        stego = self.stego_head(x)

        blueprint = self.blueprint_head(x)

        # =================================================
        # AUTHENTICATION
        # =================================================

        auth_key = generate_authentication_key(
            blueprint
        )

        return (
            stego,
            blueprint,
            auth_key
        )