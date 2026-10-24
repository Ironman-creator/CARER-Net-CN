import torch
import torch.nn as nn
import torch.nn.functional as F


# ===============================
# Residual Block
# ===============================
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        return self.relu(out)


# ===============================
# Multi-Scale Feature Fusion
# ===============================
class MSFF(nn.Module):
    """Multi-kernel feature fusion used by the submitted full model.

    ``kernels`` exists for ablation experiments.  The default ``(1, 3, 5)``
    keeps the original architecture and state-dict names unchanged, so the
    three existing checkpoints remain compatible.
    """

    def __init__(self, in_channels, kernels=(1, 3, 5)):
        super().__init__()

        kernels = tuple(sorted(set(kernels)))
        invalid = set(kernels) - {1, 3, 5}
        if not kernels or invalid:
            raise ValueError(f"kernels must be a non-empty subset of (1, 3, 5), got {kernels}")

        self.kernels = kernels

        self.conv1 = nn.Conv2d(in_channels, in_channels, 1) if 1 in kernels else None
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1) if 3 in kernels else None
        self.conv5 = nn.Conv2d(in_channels, in_channels, 5, padding=2) if 5 in kernels else None

        self.fuse = nn.Conv2d(in_channels * len(kernels), in_channels, 1)

    def forward(self, x):
        branches = []
        if self.conv1 is not None:
            branches.append(self.conv1(x))
        if self.conv3 is not None:
            branches.append(self.conv3(x))
        if self.conv5 is not None:
            branches.append(self.conv5(x))

        out = torch.cat(branches, dim=1)
        return self.fuse(out)


# ===============================
# Attention Gate
# ===============================
class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()

        self.W_g = nn.Conv2d(gating_channels, inter_channels, 1)
        self.W_x = nn.Conv2d(in_channels, inter_channels, 1)
        self.psi = nn.Conv2d(inter_channels, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.sigmoid(self.psi(F.relu(g1 + x1)))

        return x * psi


# ===============================
# Improved Attention UNet
# ===============================
class ImprovedAttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, use_msff=True,
                 msff_kernels=(1, 3, 5)):
        super().__init__()

        self.use_msff = use_msff
        self.msff_kernels = tuple(msff_kernels)

        # Encoder
        self.e1 = ResBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.e2 = ResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.e3 = ResBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.e4 = ResBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.center = ResBlock(512, 1024)

        # MSFF（关键改进）
        if use_msff:
            self.msff1 = MSFF(64, self.msff_kernels)
            self.msff2 = MSFF(128, self.msff_kernels)
            self.msff3 = MSFF(256, self.msff_kernels)
            self.msff4 = MSFF(512, self.msff_kernels)
        else:
            self.msff1 = nn.Identity()
            self.msff2 = nn.Identity()
            self.msff3 = nn.Identity()
            self.msff4 = nn.Identity()

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.att4 = AttentionGate(512, 512, 256)
        self.d4 = ResBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att3 = AttentionGate(256, 256, 128)
        self.d3 = ResBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att2 = AttentionGate(128, 128, 64)
        self.d2 = ResBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att1 = AttentionGate(64, 64, 32)
        self.d1 = ResBlock(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(self.pool1(e1))
        e3 = self.e3(self.pool2(e2))
        e4 = self.e4(self.pool3(e3))

        center = self.center(self.pool4(e4))

        # MSFF增强
        e1 = self.msff1(e1)
        e2 = self.msff2(e2)
        e3 = self.msff3(e3)
        e4 = self.msff4(e4)

        # Decoder
        d4 = self.up4(center)
        e4_att = self.att4(e4, d4)
        d4 = self.d4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        e3_att = self.att3(e3, d3)
        d3 = self.d3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        e2_att = self.att2(e2, d2)
        d2 = self.d2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        e1_att = self.att1(e1, d1)
        d1 = self.d1(torch.cat([d1, e1_att], dim=1))

        return self.out_conv(d1)


# ===============================
# test
# ===============================
if __name__ == "__main__":
    model = ImprovedAttentionUNet()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("output shape:", y.shape)
