import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Classic U-Net baseline using the same channel widths as the main model."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.e1 = DoubleConv(in_channels, 64)
        self.e2 = DoubleConv(64, 128)
        self.e3 = DoubleConv(128, 256)
        self.e4 = DoubleConv(256, 512)
        self.center = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.d4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.d3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    @staticmethod
    def _resize(x, reference):
        if x.shape[2:] != reference.shape[2:]:
            x = F.interpolate(x, size=reference.shape[2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        center = self.center(self.pool(e4))

        d4 = self._resize(self.up4(center), e4)
        d4 = self.d4(torch.cat([d4, e4], dim=1))
        d3 = self._resize(self.up3(d4), e3)
        d3 = self.d3(torch.cat([d3, e3], dim=1))
        d2 = self._resize(self.up2(d3), e2)
        d2 = self.d2(torch.cat([d2, e2], dim=1))
        d1 = self._resize(self.up1(d2), e1)
        d1 = self.d1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)


if __name__ == "__main__":
    model = UNet()
    output = model(torch.randn(1, 1, 256, 256))
    print(output.shape)
