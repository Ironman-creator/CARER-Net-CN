from models.Attention_UNet import AttentionUNet
from models.Attention_UNet_Plus import ImprovedAttentionUNet
from models.Attention_UNet_V2 import ImprovedAttentionUNetV2
from models.Attention_UNet_V3 import DCOAAttentionUNet
from models.Attention_UNet_V4 import ConnectivityAuxiliaryUNet
from models.Attention_UNet_V5 import (
    CARERUNet,
    ColorDetailOnlyUNet,
    GrayRGBControlUNet,
    SignedRefinerOnlyUNet,
)
from models.UNet import UNet
from models.modern_baselines import R2UNetGN, SAUNetGN, UNetPlusPlusGN


MODEL_CHOICES = (
    "unet",
    "attention_unet",
    "res_attention_unet",
    "improved_attention_unet",
    "res_attention_unet_gn",
    "improved_attention_unet_v2",
    "dcoa_unet",
    "connectivity_aux_unet",
    "gray_rgb_control_unet",
    "gray_rgb_control_30_unet",
    "carer_unet",
    "color_detail_only_unet",
    "signed_refiner_only_unet",
    "msff_no_k1",
    "msff_no_k3",
    "msff_no_k5",
    "unet_plus_plus_gn",
    "r2_unet_gn",
    "sa_unet_gn",
)


MODEL_TAGS = {
    "unet": "UNet",
    "attention_unet": "AttentionUNet",
    "res_attention_unet": "ResAttentionUNet",
    "improved_attention_unet": "ImprovedAttentionUNet",
    "res_attention_unet_gn": "ResAttentionUNetGN",
    "improved_attention_unet_v2": "ImprovedAttentionUNetV2",
    "dcoa_unet": "DCOAAttentionUNet",
    "connectivity_aux_unet": "ConnectivityAuxiliaryUNet",
    "gray_rgb_control_unet": "GrayRGBControlUNet",
    "gray_rgb_control_30_unet": "GrayRGBControl30UNet",
    "carer_unet": "CARERUNet",
    "color_detail_only_unet": "ColorDetailOnlyUNet",
    "signed_refiner_only_unet": "SignedRefinerOnlyUNet",
    "msff_no_k1": "ImprovedAttentionUNet_NoK1",
    "msff_no_k3": "ImprovedAttentionUNet_NoK3",
    "msff_no_k5": "ImprovedAttentionUNet_NoK5",
    "unet_plus_plus_gn": "UNetPlusPlusGN",
    "r2_unet_gn": "R2UNetGN",
    "sa_unet_gn": "SAUNetGN",
}


def build_model(name, in_channels=1, out_channels=1):
    if name == "unet":
        return UNet(in_channels, out_channels)
    if name == "attention_unet":
        return AttentionUNet(in_channels, out_channels)
    if name == "res_attention_unet":
        return ImprovedAttentionUNet(in_channels, out_channels, use_msff=False)
    if name == "improved_attention_unet":
        return ImprovedAttentionUNet(in_channels, out_channels)
    if name == "res_attention_unet_gn":
        return ImprovedAttentionUNetV2(in_channels, out_channels, use_msff=False)
    if name == "improved_attention_unet_v2":
        return ImprovedAttentionUNetV2(in_channels, out_channels, use_msff=True)
    if name == "dcoa_unet":
        return DCOAAttentionUNet(in_channels, out_channels)
    if name == "connectivity_aux_unet":
        return ConnectivityAuxiliaryUNet(in_channels, out_channels)
    if name in {"gray_rgb_control_unet", "gray_rgb_control_30_unet"}:
        return GrayRGBControlUNet(out_channels=out_channels)
    if name == "carer_unet":
        return CARERUNet(out_channels=out_channels)
    if name == "color_detail_only_unet":
        return ColorDetailOnlyUNet(out_channels=out_channels)
    if name == "signed_refiner_only_unet":
        return SignedRefinerOnlyUNet(out_channels=out_channels)
    if name == "msff_no_k1":
        return ImprovedAttentionUNet(in_channels, out_channels, msff_kernels=(3, 5))
    if name == "msff_no_k3":
        return ImprovedAttentionUNet(in_channels, out_channels, msff_kernels=(1, 5))
    if name == "msff_no_k5":
        return ImprovedAttentionUNet(in_channels, out_channels, msff_kernels=(1, 3))
    if name == "unet_plus_plus_gn":
        return UNetPlusPlusGN(in_channels, out_channels)
    if name == "r2_unet_gn":
        return R2UNetGN(in_channels, out_channels)
    if name == "sa_unet_gn":
        return SAUNetGN(in_channels, out_channels)
    raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODEL_CHOICES)}")


def checkpoint_tag(name):
    return MODEL_TAGS[name]


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
