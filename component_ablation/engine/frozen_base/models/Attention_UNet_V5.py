import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Attention_UNet_V2 import ImprovedAttentionUNetV2, group_count


class GrayRGBControlUNet(ImprovedAttentionUNetV2):
    """GN baseline adapter for the four-channel gray+RGB dataset representation."""

    def __init__(self, out_channels=1):
        super().__init__(in_channels=1, out_channels=out_channels, use_msff=False)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError("GrayRGBControlUNet expects [gray, R, G, B] input")
        return super().forward(inputs[:, :1])


class GreenChannelUNet(ImprovedAttentionUNetV2):
    """Fixed green-channel input with the unchanged one-channel GN backbone."""

    def __init__(self, out_channels=1):
        super().__init__(in_channels=1, out_channels=out_channels, use_msff=False)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError("GreenChannelUNet expects [gray, R, G, B] input")
        return super().forward(inputs[:, 2:3])


class GrayCLAHEUNet(ImprovedAttentionUNetV2):
    """Fixed CLAHE input while retaining original grayscale for FOV evaluation."""

    def __init__(self, out_channels=1):
        super().__init__(in_channels=1, out_channels=out_channels, use_msff=False)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 5:
            raise ValueError(
                "GrayCLAHEUNet expects [gray, R, G, B, gray_CLAHE] input"
            )
        return super().forward(inputs[:, 4:5])


class LearnedRGBProjectionUNet(ImprovedAttentionUNetV2):
    """Generic learnable 1x1 RGB-to-grayscale projection control."""

    def __init__(self, out_channels=1):
        super().__init__(in_channels=1, out_channels=out_channels, use_msff=False)
        self.rgb_projection = nn.Conv2d(3, 1, 1, bias=True)
        with torch.no_grad():
            self.rgb_projection.weight.copy_(
                torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
            )
            self.rgb_projection.bias.zero_()

    def forward(self, inputs, return_aux=False):
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError(
                "LearnedRGBProjectionUNet expects [gray, R, G, B] input"
            )
        grayscale = inputs[:, :1]
        projected = self.rgb_projection(inputs[:, 1:4])
        logits = super().forward(projected)
        if not return_aux:
            return logits
        return logits, {
            "base_logits": logits,
            "input_delta": projected - grayscale,
            "logit_delta": torch.zeros_like(logits),
        }


class ColorDetailStem(nn.Module):
    """Learn a bounded correction to the exact grayscale baseline input."""

    def __init__(self, feature_channels=24, use_rgb=True, use_local_contrast=True):
        super().__init__()
        if not use_rgb and not use_local_contrast:
            raise ValueError("ColorDetailStem requires at least one input cue")
        self.use_rgb = bool(use_rgb)
        self.use_local_contrast = bool(use_local_contrast)
        input_channels = (3 if self.use_rgb else 0) + (
            1 if self.use_local_contrast else 0
        )
        groups = group_count(feature_channels)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, feature_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                feature_channels,
                feature_channels,
                3,
                padding=1,
                groups=feature_channels,
                bias=False,
            ),
            nn.GroupNorm(groups, feature_channels),
            nn.SiLU(inplace=True),
        )
        self.input_delta = nn.Conv2d(feature_channels, 1, 1)
        nn.init.zeros_(self.input_delta.weight)
        nn.init.zeros_(self.input_delta.bias)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError("ColorDetailStem expects [gray, R, G, B] input")
        grayscale = inputs[:, :1]
        rgb = inputs[:, 1:4]
        green = rgb[:, 1:2]
        local_contrast = green - F.avg_pool2d(
            green, kernel_size=9, stride=1, padding=4
        )
        cues = []
        if self.use_rgb:
            cues.append(rgb)
        if self.use_local_contrast:
            cues.append(local_contrast)
        features = self.features(torch.cat(cues, dim=1))
        adjusted_grayscale = grayscale + 0.1 * torch.tanh(self.input_delta(features))
        return adjusted_grayscale, features


class SignedErrorRefiner(nn.Module):
    """Predict two unconstrained residual responses with FN/FP auxiliary supervision."""

    def __init__(self, detail_channels=24, hidden_channels=32):
        super().__init__()
        input_channels = 64 + detail_channels + 2
        groups = group_count(hidden_channels)
        self.reduce = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.local = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            3,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        self.horizontal = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            (1, 7),
            padding=(0, 3),
            groups=hidden_channels,
            bias=False,
        )
        self.vertical = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            (7, 1),
            padding=(3, 0),
            groups=hidden_channels,
            bias=False,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, 1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.fn_head = self._make_head(hidden_channels)
        self.fp_head = self._make_head(hidden_channels)
        for head in (self.fn_head, self.fp_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _make_head(channels):
        hidden = max(8, channels // 2)
        return nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, decoder_feature, detail_feature, base_logits):
        # Auxiliary error gradients may train the new modules but cannot alter
        # the established GN representation through d1 or the base prediction.
        detached_decoder = decoder_feature.detach()
        base_probability = torch.sigmoid(base_logits.detach())
        uncertainty = 0.1 + 0.9 * (
            4.0 * base_probability * (1.0 - base_probability)
        )
        refiner_inputs = [detached_decoder]
        if detail_feature is not None:
            refiner_inputs.append(detail_feature)
        refiner_inputs.extend([base_probability, uncertainty])
        reduced = self.reduce(torch.cat(refiner_inputs, dim=1))
        fused = self.fuse(
            torch.cat(
                [self.local(reduced), self.horizontal(reduced), self.vertical(reduced)],
                dim=1,
            )
        )
        fn_correction = torch.tanh(self.fn_head(fused))
        fp_correction = torch.tanh(self.fp_head(fused))
        signed_delta = uncertainty * (fn_correction - fp_correction)
        return signed_delta, fn_correction, fp_correction, uncertainty


class CARERUNet(ImprovedAttentionUNetV2):
    """Color-aware, baseline-preserving signed error refinement of the GN model."""

    def __init__(self, out_channels=1, detail_channels=24):
        # Construct the complete GN inference path first so all inherited state
        # names and initialization order remain compatible with its checkpoint.
        super().__init__(in_channels=1, out_channels=out_channels, use_msff=False)
        self.color_detail_stem = ColorDetailStem(detail_channels)
        self.error_refiner = SignedErrorRefiner(detail_channels)

    def forward(self, inputs, return_aux=False):
        grayscale, detail_feature = self.color_detail_stem(inputs)

        e1 = self.e1(grayscale)
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
        base_logits = self.out_conv(d1)

        signed_delta, fn_correction, fp_correction, uncertainty = self.error_refiner(
            d1, detail_feature, base_logits
        )
        final_logits = base_logits + signed_delta
        if not return_aux:
            return final_logits
        return final_logits, {
            "base_logits": base_logits,
            "fn_correction": fn_correction,
            "fp_correction": fp_correction,
            "uncertainty": uncertainty,
            "input_delta": grayscale - inputs[:, :1],
            "logit_delta": signed_delta,
        }


class ColorDetailOnlyUNet(ImprovedAttentionUNetV2):
    """Ablation arm containing the color/detail input adapter but no refiner."""

    def __init__(self, out_channels=1, detail_channels=24):
        super().__init__(in_channels=1, out_channels=out_channels, use_msff=False)
        self.color_detail_stem = ColorDetailStem(detail_channels)

    def forward(self, inputs, return_aux=False):
        grayscale, _ = self.color_detail_stem(inputs)
        logits = super().forward(grayscale)
        if not return_aux:
            return logits
        return logits, {
            "base_logits": logits,
            "input_delta": grayscale - inputs[:, :1],
            "logit_delta": torch.zeros_like(logits),
        }


class RGBOnlyColorDetailUNet(ColorDetailOnlyUNet):
    """CARER-Core ablation using RGB cues without local green contrast."""

    def __init__(self, out_channels=1, detail_channels=24):
        ImprovedAttentionUNetV2.__init__(
            self, in_channels=1, out_channels=out_channels, use_msff=False
        )
        self.color_detail_stem = ColorDetailStem(
            detail_channels, use_rgb=True, use_local_contrast=False
        )


class ContrastOnlyColorDetailUNet(ColorDetailOnlyUNet):
    """CARER-Core ablation using local green contrast without full RGB."""

    def __init__(self, out_channels=1, detail_channels=24):
        ImprovedAttentionUNetV2.__init__(
            self, in_channels=1, out_channels=out_channels, use_msff=False
        )
        self.color_detail_stem = ColorDetailStem(
            detail_channels, use_rgb=False, use_local_contrast=True
        )


class SignedRefinerOnlyUNet(ImprovedAttentionUNetV2):
    """Ablation arm containing the signed FN/FP refiner but no color stem."""

    def __init__(self, out_channels=1):
        super().__init__(in_channels=1, out_channels=out_channels, use_msff=False)
        self.error_refiner = SignedErrorRefiner(detail_channels=0)

    def forward(self, inputs, return_aux=False):
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError("SignedRefinerOnlyUNet expects [gray, R, G, B] input")
        grayscale = inputs[:, :1]

        e1 = self.e1(grayscale)
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
        base_logits = self.out_conv(d1)

        signed_delta, fn_correction, fp_correction, uncertainty = self.error_refiner(
            d1, None, base_logits
        )
        final_logits = base_logits + signed_delta
        if not return_aux:
            return final_logits
        return final_logits, {
            "base_logits": base_logits,
            "fn_correction": fn_correction,
            "fp_correction": fp_correction,
            "uncertainty": uncertainty,
            "input_delta": torch.zeros_like(grayscale),
            "logit_delta": signed_delta,
        }


if __name__ == "__main__":
    sample = torch.rand(1, 4, 128, 128)
    model = CARERUNet()
    prediction, auxiliary = model(sample, return_aux=True)
    print(prediction.shape, {key: value.shape for key, value in auxiliary.items()})
