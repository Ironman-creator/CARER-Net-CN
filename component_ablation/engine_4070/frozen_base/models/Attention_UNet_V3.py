import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Attention_UNet_V2 import AttentionGateV2, GNResBlock, group_count


class DecoderConditionedOrientationAttention(nn.Module):
    """Enhance an attention-gated skip with decoder-conditioned vessel features.

    The ordinary attention-gated skip remains the main path.  A lightweight
    residual branch extracts local, horizontal, and vertical responses and
    uses decoder context to select them channel by channel.  Its learnable
    per-channel strength starts near zero, so the block initially behaves like
    the validated GroupNorm residual-attention baseline.
    """

    def __init__(
        self,
        in_channels,
        gating_channels,
        inter_channels,
        reduction=4,
        directional_kernel=7,
        initial_scale=0.05,
    ):
        super().__init__()
        if directional_kernel < 3 or directional_kernel % 2 == 0:
            raise ValueError("directional_kernel must be an odd integer >= 3")
        if not 0.0 < initial_scale < 1.0:
            raise ValueError("initial_scale must be between 0 and 1")

        # This is deliberately identical to the baseline spatial attention
        # path.  The proposed branch is added residually after this path.
        self.W_g = nn.Conv2d(gating_channels, inter_channels, 1)
        self.W_x = nn.Conv2d(in_channels, inter_channels, 1)
        self.psi = nn.Conv2d(inter_channels, 1, 1)

        hidden_channels = max(16, in_channels // reduction)
        hidden_groups = group_count(hidden_channels)
        output_groups = group_count(in_channels)
        selector_channels = max(8, hidden_channels // 4)
        padding = directional_kernel // 2

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(hidden_groups, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.decoder_context = nn.Sequential(
            nn.Conv2d(gating_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(hidden_groups, hidden_channels),
        )

        def directional_branch(kernel_size, branch_padding):
            return nn.Sequential(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=branch_padding,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.GroupNorm(hidden_groups, hidden_channels),
                nn.SiLU(inplace=True),
            )

        self.local_branch = directional_branch(3, 1)
        self.horizontal_branch = directional_branch(
            (1, directional_kernel), (0, padding)
        )
        self.vertical_branch = directional_branch(
            (directional_kernel, 1), (padding, 0)
        )

        # Decoder semantics choose the useful orientation independently for
        # each reduced feature channel.
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, selector_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(selector_channels, hidden_channels * 3, 1),
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False),
            nn.GroupNorm(output_groups, in_channels),
        )

        initial_logit = math.log(initial_scale / (1.0 - initial_scale))
        self.residual_logits = nn.Parameter(
            torch.full((1, in_channels, 1, 1), initial_logit, dtype=torch.float32)
        )

    def forward(self, x, g):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        spatial_attention = torch.sigmoid(
            self.psi(F.relu(self.W_g(g) + self.W_x(x), inplace=True))
        )
        baseline_skip = x * spatial_attention

        reduced = self.reduce(x)
        context = F.silu(reduced + self.decoder_context(g), inplace=True)
        batch_size, hidden_channels = reduced.shape[:2]
        branch_weights = self.selector(context).view(
            batch_size, 3, hidden_channels, 1, 1
        )
        branch_weights = torch.softmax(branch_weights, dim=1)

        branches = torch.stack(
            [
                self.local_branch(reduced),
                self.horizontal_branch(reduced),
                self.vertical_branch(reduced),
            ],
            dim=1,
        )
        directional_features = (branches * branch_weights).sum(dim=1)
        enhancement = self.project(directional_features)
        residual_scale = torch.sigmoid(self.residual_logits)
        return baseline_skip + residual_scale * spatial_attention * enhancement


class DCOAAttentionUNet(nn.Module):
    """GroupNorm residual Attention U-Net with DCOA high-resolution skips."""

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

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        # The deepest skip is predominantly semantic; retain the validated
        # baseline gate and apply directional enhancement only where vessel
        # detail is spatially represented.
        self.att4 = AttentionGateV2(512, 512, 256)
        self.d4 = GNResBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att3 = DecoderConditionedOrientationAttention(256, 256, 128)
        self.d3 = GNResBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att2 = DecoderConditionedOrientationAttention(128, 128, 64)
        self.d2 = GNResBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att1 = DecoderConditionedOrientationAttention(64, 64, 32)
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
    model = DCOAAttentionUNet()
    output = model(torch.randn(1, 1, 128, 128))
    print(output.shape)
