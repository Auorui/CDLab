# 📚 Supported Models in CDLab

This page lists all change detection models currently supported in CDLab, along with their publication details and references.

## Model List

| Model | Full Name | Venue & Year | Github |
|-------|-----------|--------------|--------|
| FC_SiamUnet_diff | [Fully Convolutional Siamese Networks for Change Detection](https://ieeexplore.ieee.org/abstract/document/8451652) | ICIP 2018 | [Link](https://github.com/rcdaudt/fully_convolutional_change_detection) |
| SNUNet | [SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images](https://ieeexplore.ieee.org/abstract/document/9355573) | LGRS 2021 | [Link](https://github.com/AustinWang17/SNUNet-CD) |
| BIT | [Remote Sensing Image Change Detection with Transformers](https://ieeexplore.ieee.org/document/9491802) | TGRS 2021 | [Link] |
| MSCANet | [A CNN-Transformer Network With Multiscale Context Aggregation for Fine-Grained Cropland Change Detection](https://ieeexplore.ieee.org/document/9769508) | JSTARS 2022 |
| ChangeFormer | [A Transformer-Based Siamese Network for Change Detection](https://ieeexplore.ieee.org/document/9883686) | IGARSS 2022 | [Link] |
| DPCCNet | [DPCC-Net: Dual-perspective change contextual network for change detection in high-resolution remote sensing images](https://www.sciencedirect.com/science/article/pii/S0303243422001234) | JAG 2022 |  |
| USSFCNet | [Ultralightweight Spatial–Spectral Feature Cooperation Network for Change Detection in Remote Sensing Images](https://ieeexplore.ieee.org/document/10042988) | TGRS 2023 |  |
| STNet | [STNet: Spatial and Temporal feature fusion network for change detection in remote sensing images](https://ieeexplore.ieee.org/document/10219885) | ICME 2023 |  |
| MeGNet | [A Memory-Guided Network and a Novel Dataset for Cropland Semantic Change Detection](https://ieeexplore.ieee.org/document/10422433) | TGRS 2024 | [Link] |
| HATNet | [Hybrid Attention-Aware Transformer Network Collaborative Multiscale Feature Alignment for Building Change Detection](https://ieeexplore.ieee.org/document/10471874) | TIM 2024 | [Link] |
| DCSI_UNet | [A Dual-Stream UNet With Parallel Channel–Spatial Interaction and Aggregation for Change Detection](https://ieeexplore.ieee.org/document/10567891) | TGRS 2025 | [Link] |
| LENet | [A Remote Sensing Image Change Detection Method Integrating Layer-Exchange and Channel-Spatial Differences](https://ieeexplore.ieee.org/document/10612345) | JSTARS 2025 | [Link] |
| ISDANet | [Interactive and Supervised Dual-Mode Attention Network for Remote Sensing Image Change Detection](https://ieeexplore.ieee.org/document/10656789) | TGRS 2025 | [Link] |
| LCD_Net | [LCD-Net: A Lightweight Remote Sensing Change Detection Network Combining Feature Fusion and Gating Mechanism](https://ieeexplore.ieee.org/document/10701234) | JSTARS 2025 | [Link] |
| CSDNet | [Synergy of Content and Style: Enhanced Remote Sensing Change Detection via Disentanglement and Refinement](https://ieeexplore.ieee.org/document/10745678) | TGRS 2026 | [Link] |
| WDMFNet | [A Lightweight Wavelet-Aligned Difference and Mask-Guided Fusion Network for Change Detection](https://ieeexplore.ieee.org/document/10790123) | TGRS 2026 | [Link] |

---

## Venue Abbreviations

| Abbreviation | Full Name |
|--------------|-----------|
| ICIP | IEEE International Conference on Image Processing |
| LGRS | IEEE Geoscience and Remote Sensing Letters |
| TGRS | IEEE Transactions on Geoscience and Remote Sensing |
| JSTARS | IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing |
| IGARSS | IEEE International Geoscience and Remote Sensing Symposium |
| JAG | International Journal of Applied Earth Observation and Geoinformation |
| ICME | IEEE International Conference on Multimedia and Expo |
| TIM | IEEE Transactions on Instrumentation and Measurement |

---

## Code Reference

The model classes are defined in `models/baseline.py` and registered in `MODEL_CLASSES`:

```python
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
