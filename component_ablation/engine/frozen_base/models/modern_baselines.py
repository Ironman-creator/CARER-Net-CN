"""Self-contained retinal segmentation baselines for the JIIM comparison.

All models use GroupNorm because the frozen protocol uses batch size one.  The
architectural identities (nested U-Net++, recurrent residual R2U-Net and
spatial-attention U-Net) are preserved while avoiding external packages and
pretrained weights.  The paper must describe these as protocol-matched
reimplementations, not as copied official checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


def resize_like(inputs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        inputs, size=reference.shape[-2:], mode="bilinear", align_corners=False
    )


class UNetPlusPlusGN(nn.Module):
    """Nested U-Net++ reimplementation with dense decoder skip paths."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32):
        super().__init__()
        channels = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.x00 = ConvBlock(in_channels, channels[0])
        self.x10 = ConvBlock(channels[0], channels[1])
        self.x20 = ConvBlock(channels[1], channels[2])
        self.x30 = ConvBlock(channels[2], channels[3])
        self.x40 = ConvBlock(channels[3], channels[4])

        self.x01 = ConvBlock(channels[0] + channels[1], channels[0])
        self.x11 = ConvBlock(channels[1] + channels[2], channels[1])
        self.x21 = ConvBlock(channels[2] + channels[3], channels[2])
        self.x31 = ConvBlock(channels[3] + channels[4], channels[3])

        self.x02 = ConvBlock(channels[0] * 2 + channels[1], channels[0])
        self.x12 = ConvBlock(channels[1] * 2 + channels[2], channels[1])
        self.x22 = ConvBlock(channels[2] * 2 + channels[3], channels[2])

        self.x03 = ConvBlock(channels[0] * 3 + channels[1], channels[0])
        self.x13 = ConvBlock(channels[1] * 3 + channels[2], channels[1])
        self.x04 = ConvBlock(channels[0] * 4 + channels[1], channels[0])
        self.output = nn.Conv2d(channels[0], out_channels, 1)

    def _up(self, inputs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return resize_like(self.up(inputs), reference)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x00 = self.x00(inputs)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))

        x01 = self.x01(torch.cat([x00, self._up(x10, x00)], dim=1))
        x11 = self.x11(torch.cat([x10, self._up(x20, x10)], dim=1))
        x21 = self.x21(torch.cat([x20, self._up(x30, x20)], dim=1))
        x31 = self.x31(torch.cat([x30, self._up(x40, x30)], dim=1))

        x02 = self.x02(torch.cat([x00, x01, self._up(x11, x00)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self._up(x21, x10)], dim=1))
        x22 = self.x22(torch.cat([x20, x21, self._up(x31, x20)], dim=1))

        x03 = self.x03(torch.cat([x00, x01, x02, self._up(x12, x00)], dim=1))
        x13 = self.x13(torch.cat([x10, x11, x12, self._up(x22, x10)], dim=1))
        x04 = self.x04(
            torch.cat([x00, x01, x02, x03, self._up(x13, x00)], dim=1)
        )
        return self.output(x04)


class RecurrentUnit(nn.Module):
    def __init__(self, channels: int, steps: int = 2):
        super().__init__()
        self.steps = steps
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels), channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        state = None
        for _ in range(self.steps):
            state = self.conv(inputs if state is None else inputs + state)
        return state


class RRCNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, steps: int = 2):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.recurrent = nn.Sequential(
            RecurrentUnit(out_channels, steps),
            RecurrentUnit(out_channels, steps),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.project(inputs)
        return projected + self.recurrent(projected)


class R2UNetGN(nn.Module):
    """Recurrent residual U-Net (R2U-Net) with two recurrent steps."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2)
        self.e1 = RRCNNBlock(in_channels, c[0])
        self.e2 = RRCNNBlock(c[0], c[1])
        self.e3 = RRCNNBlock(c[1], c[2])
        self.e4 = RRCNNBlock(c[2], c[3])
        self.center = RRCNNBlock(c[3], c[4])
        self.d4 = RRCNNBlock(c[4] + c[3], c[3])
        self.d3 = RRCNNBlock(c[3] + c[2], c[2])
        self.d2 = RRCNNBlock(c[2] + c[1], c[1])
        self.d1 = RRCNNBlock(c[1] + c[0], c[0])
        self.output = nn.Conv2d(c[0], out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(inputs)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        center = self.center(self.pool(e4))
        d4 = self.d4(torch.cat([resize_like(center, e4), e4], dim=1))
        d3 = self.d3(torch.cat([resize_like(d4, e3), e3], dim=1))
        d2 = self.d2(torch.cat([resize_like(d3, e2), e2], dim=1))
        d1 = self.d1(torch.cat([resize_like(d2, e1), e1], dim=1))
        return self.output(d1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        average = inputs.mean(dim=1, keepdim=True)
        maximum = inputs.amax(dim=1, keepdim=True)
        attention = torch.sigmoid(self.conv(torch.cat([average, maximum], dim=1)))
        return inputs * attention


class SAUNetGN(nn.Module):
    """Protocol-matched spatial-attention U-Net reimplementation."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(in_channels, c[0])
        self.e2 = ConvBlock(c[0], c[1])
        self.e3 = ConvBlock(c[1], c[2])
        self.e4 = ConvBlock(c[2], c[3])
        self.center = ConvBlock(c[3], c[4])
        self.dropout = nn.Dropout2d(p=0.2)
        self.attention = SpatialAttention(7)
        self.d4 = ConvBlock(c[4] + c[3], c[3])
        self.d3 = ConvBlock(c[3] + c[2], c[2])
        self.d2 = ConvBlock(c[2] + c[1], c[1])
        self.d1 = ConvBlock(c[1] + c[0], c[0])
        self.output = nn.Conv2d(c[0], out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(inputs)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        center = self.attention(self.dropout(self.center(self.pool(e4))))
        d4 = self.d4(torch.cat([resize_like(center, e4), e4], dim=1))
        d3 = self.d3(torch.cat([resize_like(d4, e3), e3], dim=1))
        d2 = self.d2(torch.cat([resize_like(d3, e2), e2], dim=1))
        d1 = self.d1(torch.cat([resize_like(d2, e1), e1], dim=1))
        return self.output(d1)


if __name__ == "__main__":
    sample = torch.rand(1, 1, 128, 128)
    for model_class in (UNetPlusPlusGN, R2UNetGN, SAUNetGN):
        model = model_class()
        output = model(sample)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(model_class.__name__, output.shape, parameters)
