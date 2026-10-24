import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Attention_UNet_V2 import AttentionGateV2, GNResBlock, group_count


class DecoderProjection(nn.Module):
    """Project one decoder scale without changing the segmentation branch."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MultiLevelCenterlineHead(nn.Module):
    """Training-only centerline head that fuses three decoder resolutions."""

    def __init__(self, projection_channels=16):
        super().__init__()
        self.project_d1 = DecoderProjection(64, projection_channels)
        self.project_d2 = DecoderProjection(128, projection_channels)
        self.project_d3 = DecoderProjection(256, projection_channels)
        fused_channels = projection_channels * 3
        hidden_channels = projection_channels * 2
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, d1, d2, d3):
        target_size = d1.shape[2:]
        projected_d1 = self.project_d1(d1)
        projected_d2 = F.interpolate(
            self.project_d2(d2), size=target_size, mode="bilinear", align_corners=False
        )
        projected_d3 = F.interpolate(
            self.project_d3(d3), size=target_size, mode="bilinear", align_corners=False
        )
        return self.fuse(torch.cat([projected_d1, projected_d2, projected_d3], dim=1))


class ConnectivityAuxiliaryUNet(nn.Module):
    """GN residual Attention U-Net with training-only centerline supervision.

    The complete inference path, including module construction order, matches
    ``ImprovedAttentionUNetV2(use_msff=False)``.  The auxiliary head is created
    after the main output layer and is evaluated only when ``return_aux=True``.
    Consequently, a normal forward pass has the same computation graph as the
    existing GN baseline.
    """

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.e1 = GNResBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.e2 = GNResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.e3 = GNResBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.e4 = GNResBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.center = GNResBlock(512, 1024)

        # Keep the baseline state-dict names and construction order exactly.
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

        # This module is intentionally last so it cannot alter baseline RNG
        # initialization of any inference-path parameter for the same seed.
        self.centerline_head = MultiLevelCenterlineHead()

    @staticmethod
    def _resize(x, reference):
        if x.shape[2:] != reference.shape[2:]:
            x = F.interpolate(x, size=reference.shape[2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x, return_aux=False):
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
        logits = self.out_conv(d1)

        if not return_aux:
            return logits
        centerline_logits = self.centerline_head(d1, d2, d3)
        return logits, centerline_logits


if __name__ == "__main__":
    network = ConnectivityAuxiliaryUNet()
    sample = torch.randn(1, 1, 128, 128)
    segmentation, centerline = network(sample, return_aux=True)
    print(segmentation.shape, centerline.shape)
