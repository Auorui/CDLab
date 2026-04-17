"""
SiamUnet_diff   (Fully Convolutional Siamese Networks for Change Detection)
SNUNet     (SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images)
BIT        (Remote Sensing Image Change Detection with Transformers)
MSCANet    (A CNN-Transformer Network With Multiscale Context Aggregation for Fine-Grained Cropland Change Detection)
ChangeFormer    (A Transformer-Based Siamese Network for Change Detection)
DPCC-Net   (DPCC-Net: Dual-perspective change contextual network for change detection in high-resolution remote sensing images)
USSFCNet   (Ultralightweight Spatial–Spectral Feature Cooperation Network for Change Detection in Remote Sensing Images)
STNet      (STNet: Spatial and Temporal feature fusion network for change detection in remote sensing images)
MeGNet     (A Memory-Guided Network and a Novel Dataset for Cropland Semantic Change Detection)
HATNet     (Hybrid Attention-Aware Transformer Network Collaborative Multiscale Feature Alignment for Building Change Detection)
DCSI_UNet  (A Dual-Stream UNet With Parallel Channel–Spatial Interaction and Aggregation for Change Detection)
LENet      (A Remote Sensing Image Change Detection Method Integrating Layer-Exchange and Channel-Spatial Differences)
ISDANet    (Interactive and Supervised Dual-Mode Attention Network for Remote Sensing Image Change Detection)
LCD-Net    (LCD-Net: A Lightweight Remote Sensing Change Detection Network Combining Feature Fusion and Gating Mechanism)
CSDNet     (Synergy of Content and Style: Enhanced Remote Sensing Change Detection via Disentanglement and Refinement)
WDMFNet    (A Lightweight Wavelet-Aligned Difference and Mask-Guided Fusion Network for Change Detection)

"""
from models.baseline import *

MODEL_CLASSES = {
    "FC_SiamUnet_diff" : SiamUnet_diff,               # ICIP 2018
    "SNUNet"           : SNUNet_ECAM,                 # LGRS 2021
    "BIT"              : BIT,                         # TGRS 2021
    "MSCANet"          : MSCANet,                     # JSTARS 2022
    "ChangeFormer"     : ChangeFormerV6,              # IGARSS 2022
    "DPCCNet"          : DPCCNet,                     # JAG 2022
    "USSFCNet"         : USSFCNet,                    # TGRS 2023
    "STNet"            : STNet,                       # ICME 2023
    "MeGNet"           : MeGNetApt,                   # TGRS 2024
    "HATNet"           : HATNet,                      # TIM 2024
    "DCSI_UNet"        : DCSI_UNet,                   # TGRS 2025
    "LENet"            : LENet,                       # JSTARS 2025
    "ISDANet"          : ISDANet,                     # TGRS 2025
    "LCD_Net"          : LCD_Net,                     # JSTARS 2025
    "CSDNet"           : CSDNet,                      # TGRS 2026
    "WDMFNet"          : WDMFNet,                     # TGRS 2026
}

def get_change_networks(name, **kwargs):
    if name in MODEL_CLASSES:
        model = MODEL_CLASSES[name](**kwargs)
        return model
    else:
        raise ValueError(f"Model {name} not found in MODEL_CLASSES")


if __name__ == "__main__":
    import torch
    from utils.utils import load_config
    x1 = torch.rand(1, 3, 256, 256).cuda()
    x2 = torch.rand(1, 3, 256, 256).cuda()
    conf = load_config('E:\PythonProject\CDLab\config\MeGNet.yaml')
    Net = get_change_networks(conf.model.name, **conf.model.params).cuda()

    # Calculate GFLOPs & Parameters / 计算 FLOPs 与参数量
    from thop import profile

    flops, params = profile(Net, inputs=(x1, x2))
    print(f"Model FLOPs: {flops / 1e9:.4f} GFLOPs")
    print(f"Model Parameters: {params / 1e6:.2f} M")







