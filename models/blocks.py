import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    DROPOUT
)


# =========================================================
# RESIDUAL BLOCK
# =========================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            ),

            nn.InstanceNorm2d(channels),

            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            ),

            nn.InstanceNorm2d(channels)
        )

        self.scale = nn.Parameter(
            torch.ones(1)
        )

    def forward(self, x):

        residual = x

        out = self.block(x)

        out = residual + self.scale * out

        return F.gelu(out)


# =========================================================
# SWIN V2 STYLE BLOCK
# =========================================================

class SwinBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.depthwise = nn.Conv2d(
            channels,
            channels,
            7,
            padding=3,
            groups=channels
        )

        self.pointwise1 = nn.Conv2d(
            channels,
            channels * 2,
            1
        )

        self.pointwise2 = nn.Conv2d(
            channels * 2,
            channels,
            1
        )

        self.norm = nn.InstanceNorm2d(
            channels
        )

        self.gelu = nn.GELU()

        self.dropout = nn.Dropout2d(
            DROPOUT
        )

    def forward(self, x):

        residual = x

        x = self.norm(x)

        shifted = torch.roll(
            x,
            shifts=(2, 2),
            dims=(2, 3)
        )

        x = self.depthwise(shifted)

        x = self.pointwise1(x)

        x = self.gelu(x)

        x = self.dropout(x)

        x = self.pointwise2(x)

        return x + residual


# =========================================================
# COUPLING BLOCK
# =========================================================

class CouplingBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        half = channels // 2

        self.F = nn.Sequential(

            nn.Conv2d(
                half,
                half,
                3,
                padding=1
            ),

            nn.InstanceNorm2d(half),

            nn.GELU(),

            nn.Conv2d(
                half,
                half,
                3,
                padding=1
            )
        )

        self.G = nn.Sequential(

            nn.Conv2d(
                half,
                half,
                3,
                padding=1
            ),

            nn.InstanceNorm2d(half),

            nn.GELU(),

            nn.Conv2d(
                half,
                half,
                3,
                padding=1
            )
        )

        self.scale = nn.Parameter(
            torch.ones(1)
        )

    def forward(self, x):

        x1, x2 = torch.chunk(
            x,
            2,
            dim=1
        )

        y1 = x1 + self.scale * self.F(x2)

        y2 = x2 + self.scale * self.G(y1)

        return torch.cat(
            [y1, y2],
            dim=1
        )


# =========================================================
# FEATURE FUSION
# =========================================================

class FeatureFusion(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                channels * 2,
                channels,
                1
            ),

            nn.InstanceNorm2d(channels),

            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            )
        )

    def forward(self, x1, x2):

        x = torch.cat(
            [x1, x2],
            dim=1
        )

        return self.conv(x)


# =========================================================
# EDGE ENHANCEMENT
# =========================================================

class EdgeEnhancement(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            )
        )

    def forward(self, x):

        edge = torch.abs(

            x -

            F.avg_pool2d(
                x,
                3,
                stride=1,
                padding=1
            )
        )

        enhanced = self.conv(edge)

        return x + 0.1 * enhanced