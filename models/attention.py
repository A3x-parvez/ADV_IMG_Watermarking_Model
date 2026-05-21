import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# COORDINATE ATTENTION
# =========================================================

class CoordinateAttention(nn.Module):

    def __init__(
        self,
        channels,
        reduction=32
    ):

        super().__init__()

        mid = max(
            8,
            channels // reduction
        )

        self.conv1 = nn.Conv2d(
            channels,
            mid,
            1
        )

        self.bn = nn.InstanceNorm2d(mid)

        self.act = nn.GELU()

        self.conv_h = nn.Conv2d(
            mid,
            channels,
            1
        )

        self.conv_w = nn.Conv2d(
            mid,
            channels,
            1
        )

    def forward(self, x):

        identity = x

        b, c, h, w = x.size()

        x_h = F.adaptive_avg_pool2d(
            x,
            (h, 1)
        )

        x_w = F.adaptive_avg_pool2d(
            x,
            (1, w)
        )

        x_w = x_w.permute(
            0,
            1,
            3,
            2
        )

        y = torch.cat(
            [x_h, x_w],
            dim=2
        )

        y = self.conv1(y)

        y = self.bn(y)

        y = self.act(y)

        x_h, x_w = torch.split(
            y,
            [h, w],
            dim=2
        )

        x_w = x_w.permute(
            0,
            1,
            3,
            2
        )

        a_h = torch.sigmoid(
            self.conv_h(x_h)
        )

        a_w = torch.sigmoid(
            self.conv_w(x_w)
        )

        return identity * a_h * a_w


# =========================================================
# MDTA
# =========================================================

class MDTA(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.norm = nn.InstanceNorm2d(
            channels
        )

        self.qkv = nn.Conv2d(
            channels,
            channels * 3,
            1
        )

        self.qkv_dwconv = nn.Conv2d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3
        )

        self.project = nn.Conv2d(
            channels,
            channels,
            1
        )

        self.temperature = nn.Parameter(
            torch.ones(1)
        )

    def forward(self, x):

        identity = x

        x = self.norm(x)

        qkv = self.qkv(x)

        qkv = self.qkv_dwconv(qkv)

        q, k, v = torch.chunk(
            qkv,
            3,
            dim=1
        )

        q = F.normalize(q, dim=1)

        k = F.normalize(k, dim=1)

        attention = torch.softmax(
            self.temperature * (q * k),
            dim=1
        )

        out = attention * v

        out = self.project(out)

        return out + identity


# =========================================================
# CROSS ATTENTION
# =========================================================

class CrossAttention(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.norm_x = nn.InstanceNorm2d(
            channels
        )

        self.norm_g = nn.InstanceNorm2d(
            channels
        )

        self.query = nn.Conv2d(
            channels,
            channels,
            1
        )

        self.key = nn.Conv2d(
            channels,
            channels,
            1
        )

        self.value = nn.Conv2d(
            channels,
            channels,
            1
        )

        self.depthwise = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels
        )

        self.project = nn.Conv2d(
            channels,
            channels,
            1
        )

        self.scale = nn.Parameter(
            torch.ones(1)
        )

    def forward(self, x, guide):

        identity = x

        x = self.norm_x(x)

        guide = self.norm_g(guide)

        q = self.query(x)

        k = self.key(guide)

        v = self.value(guide)

        q = F.normalize(q, dim=1)

        k = F.normalize(k, dim=1)

        attention = torch.softmax(
            self.scale * (q * k),
            dim=1
        )

        out = attention * v

        out = self.depthwise(out)

        out = self.project(out)

        return out + identity