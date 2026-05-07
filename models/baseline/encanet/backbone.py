# EfficientNetV2-S Encoder for Remote Sensing Change Detection
# Adapted for FinalNet framework

import torch
import torch.nn as nn
import torchvision


class EfficientNetV2S_Encoder(nn.Module):
    """
    EfficientNetV2-S编码器，用于遥感变化检测

    该类从完整的EfficientNetV2-S模型中提取4个主要阶段作为特征编码器，
    并自动下载和加载ImageNet预训练权重。

    根据PyTorch官方架构：
    Stage 0: features[0] + features[1] (初始conv + stage1) → 24通道
    Stage 1: features[2] (stage2) → 48通道
    Stage 2: features[3] (stage3) → 64通道
    Stage 3: features[4] (stage4) → 128通道

    Args:
        freeze_backbone (bool): 是否冻结骨干网络参数的梯度
    """

    def __init__(self, freeze_backbone=False):
        super(EfficientNetV2S_Encoder, self).__init__()

        # 定义EfficientNetV2-S ImageNet预训练权重URL
        weights_url = "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth"

        try:
            # 创建空的EfficientNetV2-S模型结构（不加载预训练权重）
            full_model = torchvision.models.efficientnet_v2_s(weights=None)

            # 从URL加载预训练权重
            state_dict = torch.hub.load_state_dict_from_url(
                weights_url,
                map_location="cpu",
                progress=True
            )

            # 加载权重到完整模型
            full_model.load_state_dict(state_dict)
            # print("Successfully loaded EfficientNetV2-S ImageNet pretrained weights")

        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            # 如果下载失败，创建空模型
            full_model = torchvision.models.efficientnet_v2_s(weights=None)
            print("Using random initialization for EfficientNetV2-S")

        # 提取features的各个阶段
        # 根据PyTorch官方EfficientNetV2-S架构：
        # features[0]: 初始conv层 (3→24通道)
        # features[1]: FusedMBConvConfig(1, 3, 1, 24, 24, 2) → 24通道
        # features[2]: FusedMBConvConfig(4, 3, 2, 24, 48, 4) → 48通道
        # features[3]: FusedMBConvConfig(4, 3, 2, 48, 64, 4) → 64通道
        # features[4]: MBConvConfig(4, 3, 2, 64, 128, 6) → 128通道
        # features[5]: MBConvConfig(6, 3, 1, 128, 160, 9) → 160通道
        # features[6]: MBConvConfig(6, 3, 2, 160, 256, 15) → 256通道

        features = full_model.features

        # 提取4个主要阶段用于变化检测
        self.stage0 = nn.Sequential(features[0], features[1])  # 初始conv + stage1: → 24
        self.stage1 = features[2]  # stage2: → 48
        self.stage2 = features[3]  # stage3: → 64
        self.stage3 = features[4]  # stage4: → 128

        # 定义各阶段的输出通道数（与EfficientNetV2-S架构对应）
        self.skip_channels = [24, 48, 64, 128]  # 4个阶段的输出通道数
        self.out_channels = 128  # 第4个阶段的输出通道数

        # 应用冻结设置
        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False
            print("EfficientNetV2-S backbone parameters frozen")
        else:
            print("EfficientNetV2-S backbone parameters trainable")

    def forward(self, x):
        """
        前向传播，返回4个阶段的特征

        Args:
            x (torch.Tensor): 输入图像 (B, 3, H, W)

        Returns:
            list: 包含4个阶段输出特征的列表，从低分辨率到高分辨率
        """
        # 依次通过4个阶段
        x1 = self.stage0(x)  # Stage 0: → [B, 24, H/4, W/4]
        x2 = self.stage1(x1)  # Stage 1: → [B, 48, H/8, W/8]
        x3 = self.stage2(x2)  # Stage 2: → [B, 64, H/16, W/16]
        x4 = self.stage3(x3)  # Stage 3: → [B, 128, H/32, W/32]

        # 返回4个阶段的特征（从低分辨率到高分辨率）
        return [x4, x3, x2, x1]


def get_encoder(backbone_scale='efficientnetv2_s_22k', freeze_backbone=False):
    """
    获取编码器实例

    Args:
        backbone_scale (str): 骨干网络类型，目前只支持'efficientnetv2_s_22k'
        freeze_backbone (bool): 是否冻结骨干网络参数

    Returns:
        EfficientNetV2S_Encoder: 编码器实例

    Raises:
        NotImplementedError: 当backbone_scale不是'efficientnetv2_s_22k'时
    """
    if backbone_scale == "efficientnetv2_s_22k":
        return EfficientNetV2S_Encoder(freeze_backbone=freeze_backbone)
    else:
        raise NotImplementedError(
            f"Backbone '{backbone_scale}' is not supported. "
            f"Only 'efficientnetv2_s_22k' is available."
        )
