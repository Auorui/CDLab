"""
MSCANet (A CNN-Transformer Network With Multiscale Context Aggregation for Fine-Grained Cropland Change Detection)
MeGNet  (A Memory-Guided Network and a Novel Dataset for Cropland Semantic Change Detection)
STNet   (STNet: Spatial and Temporal feature fusion network for change detection in remote sensing images)
SNUNet  (SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images)
SiamUnet_diff (Fully Convolutional Siamese Networks for Change Detection)
ChangeFormer  (A Transformer-Based Siamese Network for Change Detection)
DCSI_UNet  (A Dual-Stream UNet With Parallel Channel–Spatial Interaction and Aggregation for Change Detection)
LENet      (A Remote Sensing Image Change Detection Method Integrating Layer-Exchange and Channel-Spatial Differences)
ISDANet    (Interactive and Supervised Dual-Mode Attention Network for Remote Sensing Image Change Detection)

"""
from models.baseline import *

MODEL_CLASSES = {
    "MSCANet"          : MSCANet,
    "MeGNet"           : MeGNetApt,
    "STNet"            : STNet,
    "FC_SiamUnet_diff" : SiamUnet_diff,
    "SNUNet"           : SNUNet_ECAM,
    "ChangeFormer"     : ChangeFormerV6,
    "DCSI_UNet"        : DCSI_UNet,
    "LENet"            : LENet,
    "ISDANet"          : ISDANet,
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







