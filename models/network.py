from .baseline import *

MODEL_CLASSES = {
    "MSCANet"          : MSCANet,
    "MeGNet"           : MeGNetApt,
    "STNet"            : STNet,
    "FC_SiamUnet_diff" : SiamUnet_diff,
    "SNUNet"           : SNUNet_ECAM,
    "ChangeFormer"     : ChangeFormerV6,
    "DCSI_UNet"        : DCSI_UNet
}

def get_change_networks(name, **kwargs):
    if name in MODEL_CLASSES:
        model = MODEL_CLASSES[name](**kwargs)
        return model
    else:
        raise ValueError(f"Model {name} not found in MODEL_CLASSES")










