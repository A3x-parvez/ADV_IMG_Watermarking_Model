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
    MDTA,
    CrossAttention
)


class Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        ch = BASE_CHANNELS

        # =====================================================
        # STEGO FEATURE STEM
        # =====================================================

        self.stego_stem = nn.Sequential(

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
        # BLUEPRINT FEATURE STEM
        # =====================================================

        self.blueprint_stem = nn.Sequential(

            nn.Conv2d(
                BLUEPRINT_CHANNELS,
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
        # DEEP RECOVERY
        # =====================================================

        self.recovery = nn.Sequential(

            ResidualBlock(ch),

            CrossAttention(ch),

            MDTA(ch),

            SwinBlock(ch),

            ResidualBlock(ch),

            CoordinateAttention(ch),

            SwinBlock(ch),

            CouplingBlock(ch),

            EdgeEnhancement(ch)
        )

        # =====================================================
        # COVER RECOVERY HEAD
        # =====================================================

        self.cover_head = nn.Sequential(

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
        # WATERMARK RECOVERY HEAD
        # =====================================================

        self.watermark_head = nn.Sequential(

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
            )
        )

    def forward(self, stego, blueprint):

        # =================================================
        # FEATURE EXTRACTION
        # =================================================

        stego_feat = self.stego_stem(
            stego
        )

        blueprint_feat = self.blueprint_stem(
            blueprint
        )

        # =================================================
        # FEATURE FUSION
        # =================================================

        x = self.fusion(
            stego_feat,
            blueprint_feat
        )

        # =================================================
        # BLUEPRINT GUIDED RECOVERY
        # =================================================

        for layer in self.recovery:

            if isinstance(layer, CrossAttention):

                x = layer(
                    x,
                    blueprint_feat
                )

            else:

                x = layer(x)

        # =================================================
        # OUTPUTS
        # =================================================

        recovered_cover = self.cover_head(x)

        recovered_watermark = self.watermark_head(x)

        return (
            recovered_cover,
            recovered_watermark
        )