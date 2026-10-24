import torch
import torch.nn as nn
import torch.nn.functional as F


def group_count(channels, maximum=8):
    """Choose the largest small group count that divides the channel count."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class GNResBlock(nn.Module):
    """Residual block whose normalization is independent of batch size."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = group_count(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.activation(out + identity)


class LiteResidualMSFF(nn.Module):
    """Lightweight residual multi-scale fusion for high-resolution skip features.

    A shared 1x1 projection reduces channels before three depthwise dilated
    branches.  A learnable residual scale starts small so the block initially
    behaves close to an identity mapping instead of replacing useful encoder
    features outright.
    """

    def __init__(self, channels, reduction=4, dilations=(1, 2, 3)):
        super().__init__()
        hidden_channels = max(16, channels // reduction)
        hidden_groups = group_count(hidden_channels)
        output_groups = group_count(channels)

        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(hidden_groups, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        groups=hidden_channels,
                        bias=False,
                    ),
                    nn.GroupNorm(hidden_groups, hidden_channels),
                    nn.ReLU(inplace=True),
                )
                for dilation in dilations
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * len(dilations), channels, 1, bias=False),
            nn.GroupNorm(output_groups, channels),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x):
        reduced = self.reduce(x)
        fused = self.fuse(torch.cat([branch(reduced) for branch in self.branches], dim=1))
        return x + self.residual_scale * fused


class AttentionGateV2(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Conv2d(gating_channels, inter_channels, 1)
        self.W_x = nn.Conv2d(in_channels, inter_channels, 1)
        self.psi = nn.Conv2d(inter_channels, 1, 1)

    def forward(self, x, g):
        attention = torch.sigmoid(self.psi(F.relu(self.W_g(g) + self.W_x(x))))
        return x * attention


class ImprovedAttentionUNetV2(nn.Module):
    """GroupNorm residual Attention U-Net with optional lightweight MSFF."""

    def __init__(self, in_channels=1, out_channels=1, use_msff=True):
        super().__init__()
        self.use_msff = use_msff

        self.e1 = GNResBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.e2 = GNResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.e3 = GNResBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.e4 = GNResBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.center = GNResBlock(512, 1024)

        if use_msff:
            self.msff1 = LiteResidualMSFF(64)
            self.msff2 = LiteResidualMSFF(128)
            self.msff3 = LiteResidualMSFF(256)
            self.msff4 = LiteResidualMSFF(512)
        else:
            self.msff1 = nn.Identity()
            self.msff2 = nn.Identity()
            self.msff3 = nn.Identity()
            self.msff4 = nn.Identity()

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.att4 = AttentionGateV2(512, 512, 256)
        self.d4 = GNResBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att3 = AttentionGateV2(256, 256, 128)
        self.d3 = GNResBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att2 = AttentionGateV2(128, 128, 64)
        self.d2 = GNResBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att1 = AttentionGateV2(64, 64, 32)
        self.d1 = GNResBlock(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    @staticmethod
    def _resize(x, reference):
        if x.shape[2:] != reference.shape[2:]:
            x = F.interpolate(x, size=reference.shape[2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool1(e1))
        e3 = self.e3(self.pool2(e2))
        e4 = self.e4(self.pool3(e3))
        center = self.center(self.pool4(e4))

        e1 = self.msff1(e1)
        e2 = self.msff2(e2)
        e3 = self.msff3(e3)
        e4 = self.msff4(e4)

        d4 = self._resize(self.up4(center), e4)
        d4 = self.d4(torch.cat([d4, self.att4(e4, d4)], dim=1))
        d3 = self._resize(self.up3(d4), e3)
        d3 = self.d3(torch.cat([d3, self.att3(e3, d3)], dim=1))
        d2 = self._resize(self.up2(d3), e2)
        d2 = self.d2(torch.cat([d2, self.att2(e2, d2)], dim=1))
        d1 = self._resize(self.up1(d2), e1)
        d1 = self.d1(torch.cat([d1, self.att1(e1, d1)], dim=1))
        return self.out_conv(d1)


if __name__ == "__main__":
    for enabled in (False, True):
        model = ImprovedAttentionUNetV2(use_msff=enabled)
        output = model(torch.randn(1, 1, 128, 128))
        print(enabled, output.shape)
