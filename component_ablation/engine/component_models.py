"""Single-factor CARER-Core variants using the frozen GN implementation.

No dataset discovery or checkpoint deserialization occurs in this module.
The caller must load its trusted checkpoint before calling load_gn_weights.
"""

from collections.abc import Mapping
from pathlib import Path
import sys

FROZEN_BASE = Path(__file__).resolve().parent / "frozen_base"
if str(FROZEN_BASE) in sys.path:
    sys.path.remove(str(FROZEN_BASE))
sys.path.insert(0, str(FROZEN_BASE))

import torch
from torch import nn
from torch.nn import functional as F

from models.Attention_UNet_V2 import ImprovedAttentionUNetV2, group_count
from models.Attention_UNet_V5 import ColorDetailOnlyUNet, ColorDetailStem

for _implementation in (ImprovedAttentionUNetV2, ColorDetailOnlyUNet):
    _module_file = Path(sys.modules[_implementation.__module__].__file__).resolve()
    if not _module_file.is_relative_to(FROZEN_BASE.resolve()):
        raise RuntimeError(
            "A different models package was imported before frozen_base: "
            f"{_module_file}. Start this package in a fresh Python process."
        )


ARMS = ("core", "no_bound", "random_init")
MODULE_PREFIX = "color_detail_stem."
TOTAL_PARAMETERS = 32_779_990
MODULE_PARAMETERS = 1_201


class ComponentColorDetailStem(ColorDetailStem):
    """Keep construction order and state names identical to frozen V5.

    Conv2d's last-layer constructor always consumes its default random draws.
    The core/no_bound arms then zero those values; random_init simply retains
    the defaults. No second reset or extra random draw is performed.
    """

    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; choose one of {ARMS}")
        nn.Module.__init__(self)
        self.arm = arm
        self.use_rgb = True
        self.use_local_contrast = True
        channels = 24
        groups = group_count(channels)
        self.features = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels, bias=False
            ),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
        )
        self.input_delta = nn.Conv2d(channels, 1, 1)
        if arm != "random_init":
            nn.init.zeros_(self.input_delta.weight)
            nn.init.zeros_(self.input_delta.bias)

    def forward_with_diagnostics(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError("ComponentColorDetailStem expects [gray, R, G, B]")
        grayscale = inputs[:, :1]
        rgb = inputs[:, 1:4]
        green = rgb[:, 1:2]
        local_contrast = green - F.avg_pool2d(
            green, kernel_size=9, stride=1, padding=4
        )
        features = self.features(torch.cat([rgb, local_contrast], dim=1))
        raw_residual = self.input_delta(features)
        if self.arm == "no_bound":
            correction = 0.1 * raw_residual
        else:
            correction = 0.1 * torch.tanh(raw_residual)
        adjusted_input = grayscale + correction
        # Deliberately do not clamp: the original Core bounds the increment,
        # not the sum of that increment and the original grayscale value.
        return adjusted_input, features, raw_residual

    def forward(self, inputs):
        adjusted, features, _ = self.forward_with_diagnostics(inputs)
        return adjusted, features


class ComponentUNet(ColorDetailOnlyUNet):
    """The exact Core architecture with one explicitly selected factor."""

    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; choose one of {ARMS}")
        ImprovedAttentionUNetV2.__init__(
            self, in_channels=1, out_channels=1, use_msff=False
        )
        self.arm = arm
        self.color_detail_stem = ComponentColorDetailStem(arm)

    def forward(self, inputs, return_aux=False):
        adjusted, _, raw_residual = (
            self.color_detail_stem.forward_with_diagnostics(inputs)
        )
        logits = ImprovedAttentionUNetV2.forward(self, adjusted)
        if not return_aux:
            return logits
        return logits, {
            "base_logits": logits,
            "input_delta": adjusted - inputs[:, :1],
            "logit_delta": torch.zeros_like(logits),
            "raw_residual": raw_residual,
            "adjusted_input": adjusted,
        }


def build_model(arm):
    """Construct one arm without setting/resetting the caller's random seed."""
    return ComponentUNet(arm)


def load_gn_weights(model, checkpoint):
    """Validate and load only an explicitly identified FIVES gray GN backbone.

    A raw state dict, a component checkpoint, a different dataset/model, or
    incomplete metadata is rejected. All key and shape checks happen before
    any parameter is copied. Only the new color_detail_stem keys may be absent.
    Returns their names for compatibility with the original ablation loader.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError("GN checkpoint must be a metadata-bearing mapping")
    if checkpoint.get("dataset") != "FIVES":
        raise ValueError("Expected checkpoint dataset=FIVES")
    if checkpoint.get("model_name") != "res_attention_unet_gn":
        raise ValueError("Expected checkpoint model_name=res_attention_unet_gn")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("GN checkpoint must contain config metadata")
    if config.get("image_size") != 512:
        raise ValueError("Expected config.image_size=512")
    if config.get("input_mode") != "grayscale":
        raise ValueError("Expected config.input_mode=grayscale")
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("GN checkpoint must contain state_dict")
    expected = model.state_dict()
    module_keys = {key for key in expected if key.startswith(MODULE_PREFIX)}
    if not module_keys:
        raise ValueError("Target model does not contain color_detail_stem")
    backbone_keys = set(expected) - module_keys
    missing = backbone_keys - set(state)
    unexpected = set(state) - backbone_keys
    if missing or unexpected:
        raise ValueError(
            "Backbone-only checkpoint keys mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for name in sorted(backbone_keys):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Checkpoint value is not a tensor: {name}")
        if value.shape != expected[name].shape:
            raise ValueError(
                f"Checkpoint shape mismatch for {name}: "
                f"{tuple(value.shape)} != {tuple(expected[name].shape)}"
            )
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or set(result.missing_keys) != module_keys:
        raise RuntimeError("Unexpected incompatibility after validated GN load")
    return list(result.missing_keys)


__all__ = ["ARMS", "build_model", "load_gn_weights"]
