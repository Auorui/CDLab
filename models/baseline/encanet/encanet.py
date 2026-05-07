import torch
import torch.nn as nn
import torch.nn.functional as F
from models.baseline.encanet.backbone import get_encoder
from models.baseline.encanet.fusion_modules import TwoStageUniFiREFusion, UniFiREBlock


class EnFoCSAModule(nn.Module):
    """
    跨时相空间注意力模块 (Cross Spatial Attention Module, CSAM)

    从TinyCD迁移的CSAM模块，用于增强双时相特征图之间的空间关联性。
    通过计算两个时相特征图的空间统计信息（avg和max pooling），
    生成统一的注意力权重并同时应用到两个时相，实现跨时相的空间注意力增强。

    Args:
        kernel_size: 卷积核大小，默认为7
    """

    def __init__(self, kernel_size=7):
        super(EnFoCSAModule, self).__init__()

        # 跨时相注意力生成卷积：4通道输入 → 1通道输出
        # 4通道 = T1的(avg+max) + T2的(avg+max)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x, y):
        """
        前向传播

        Args:
            x: 第一时相特征图 (B, C, H, W)
            y: 第二时相特征图 (B, C, H, W)

        Returns:
            Tuple[Tensor, Tensor]: 注意力增强后的两个时相特征图
        """
        # 第一时相的空间统计特征
        x_avg = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        x_attn = torch.cat([x_avg, x_max], dim=1)  # (B, 2, H, W)

        # 第二时相的空间统计特征
        y_avg = torch.mean(y, dim=1, keepdim=True)  # (B, 1, H, W)
        y_max = torch.max(y, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        y_attn = torch.cat([y_avg, y_max], dim=1)  # (B, 2, H, W)

        # 跨时相特征拼接与注意力权重生成
        attn = torch.cat([x_attn, y_attn], dim=1)  # (B, 4, H, W)
        attn = self.conv(attn)  # (B, 4, H, W) → (B, 1, H, W)
        attn = torch.sigmoid(attn)  # 注意力权重归一化到[0,1]

        # 同一注意力权重同时应用到两个时相特征图
        x_enhanced = x * attn  # (B, C, H, W) * (B, 1, H, W) → (B, C, H, W)
        y_enhanced = y * attn  # (B, C, H, W) * (B, 1, H, W) → (B, C, H, W)

        return x_enhanced, y_enhanced


class EnCANet(nn.Module):
    def __init__(
            self,
            spatial_dims: int = 2,
            in_channels: int = 3,
            num_classes: int = 2,
            backbone_name: str = 'efficientnetv2_s_22k',
            pretrained: bool = True,
            backbone_trainable: bool = True,
            **kwargs
    ):
        super().__init__()

        if spatial_dims != 2:
            raise ValueError("`spatial_dims` must be 2 for EnCANet.")

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.backbone_trainable = backbone_trainable

        self.use_csam = True
        self.csam_kernel_size = 7
        self.use_entropy_weighting = True
        self.temporal_fusion_method = "channel_concat"
        self.spatial_fusion_method = "channel_concat"

        self.encoder = get_encoder(
            backbone_scale=backbone_name,
            freeze_backbone=not backbone_trainable
        )

        if not backbone_trainable:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.stageNumber = 4
        self.encoderNameScale = 2

        self.backbone_dims = {
            'tiny': [96, 192, 384, 768],
            'small': [96, 192, 384, 768],
            'base': [128, 256, 512, 1024],
            'large': [192, 384, 768, 1536],
            'xlarge': [256, 512, 1024, 2048],
            'efficientnetv2_s_22k': [128, 64, 48, 24]
        }

        self.size_dict = {
            'tiny': [24, 96, 192, 384, 768],
            'small': [24, 96, 192, 384, 768],
            'base': [32, 128, 256, 512, 1024],
            'large': [48, 192, 384, 768, 1536],
            'xlarge': [64, 256, 512, 1024, 2048],
            'efficientnetv2_s_22k': [32, 24, 48, 64, 128]
        }

        self.feature_dims = self.backbone_dims[backbone_name]
        self.size_change = list(reversed(self.size_dict[backbone_name]))

        self.use_attention = False
        self.use_cbam = False
        self.use_temporal_attention = False
        self.CSAMs = nn.ModuleList()
        self.FusionBlocks = nn.ModuleList()

        self._build_modules()

        self.ChangeFinalSqueezeConv = UniFiREBlock(
            in_channels=sum(self.size_change[:-1]),
            out_channels=self.size_change[-1] * self.encoderNameScale,
            use_se=False,
            use_residual=True
        )

        self.ChangeFinalConv = nn.Sequential(
            UniFiREBlock(
                in_channels=self.size_change[-1] * self.encoderNameScale,
                out_channels=self.size_change[-1],
                use_se=False,
                use_residual=True
            ),
            nn.Conv2d(self.size_change[-1], num_classes, kernel_size=1)
        )

        self.register_hook(self.encoder)
        self.backboneFeatures = []

    def _build_modules(self):

        for stage in range(self.stageNumber):

            csam_module = EnFoCSAModule(kernel_size=self.csam_kernel_size)
            self.CSAMs.append(csam_module)

            use_temporal_se = False
            use_spatial_se = False

            fusion_module = TwoStageUniFiREFusion(
                encoder_channels=self.feature_dims[stage],
                decoder_channels=None if stage == 0 else self.size_change[stage - 1],
                temporal_fusion_method=self.temporal_fusion_method,
                spatial_fusion_method=self.spatial_fusion_method,
                use_temporal_se=use_temporal_se,
                use_spatial_se=use_spatial_se
            )

            self.FusionBlocks.append(fusion_module)

    def register_hook(self, backbone):

        def hook(module, input, output):
            self.backboneFeatures.append(output)

        if self.backbone_name == 'efficientnetv2_s_22k':

            backbone.stage3.register_forward_hook(hook)
            backbone.stage2.register_forward_hook(hook)
            backbone.stage1.register_forward_hook(hook)
            backbone.stage0.register_forward_hook(hook)
        else:

            for index in range(len(self.feature_dims)):
                stage = backbone.stages[index]
                last_block = stage[-1]
                last_block.register_forward_hook(hook)

    def apply_entropy_weighting(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)

        x_flat = x_norm.view(batch_size, channels, -1)
        prob = x_flat * (x_flat > 1e-5).float()
        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=-1)  # 熵计算
        entropy = entropy / 100.0
        weight_adjustment = entropy.unsqueeze(-1).unsqueeze(-1)

        weight_adjustment = weight_adjustment.expand(-1, -1, x.size(2), x.size(3))

        return x * weight_adjustment

    def forward(self, x1, x2):
        self.backboneFeatures = []

        _ = self.encoder(x1)
        _ = self.encoder(x2)

        blocks1 = self.backboneFeatures[0:self.stageNumber]
        blocks2 = self.backboneFeatures[self.stageNumber:]

        self.backboneFeatures = []

        FusionFeatures = []
        change = None

        for stage in range(self.stageNumber):
            eff_last_1 = blocks1.pop()
            eff_last_2 = blocks2.pop()

            eff_last_1 = self.apply_entropy_weighting(eff_last_1)
            eff_last_2 = self.apply_entropy_weighting(eff_last_2)

            eff_last_1, eff_last_2 = self.CSAMs[stage](eff_last_1, eff_last_2)

            if stage == 0:
                change = self.FusionBlocks[stage](
                    x_decoder=None,
                    skip_T1=eff_last_1,
                    skip_T2=eff_last_2
                )
            else:
                change = self.FusionBlocks[stage](
                    x_decoder=change,
                    skip_T1=eff_last_1,
                    skip_T2=eff_last_2
                )

            FusionFeatures.append(change)

            if stage < self.stageNumber - 1:
                change = F.interpolate(change, scale_factor=2., mode='bilinear', align_corners=True)

        for index in range(len(FusionFeatures)):
            scale_factor = 2 ** (self.stageNumber - index - 1)
            FusionFeatures[index] = F.interpolate(
                FusionFeatures[index], scale_factor=scale_factor,
                mode='bilinear', align_corners=True
            )

        fusion = torch.cat(FusionFeatures, dim=1)

        change = self.ChangeFinalSqueezeConv(fusion)
        change = F.interpolate(change, scale_factor=self.encoderNameScale,
                               mode='bilinear', align_corners=True)
        change = self.ChangeFinalConv(change)

        return change

if __name__ == "__main__":
    x1 = torch.rand(1, 3, 256, 256).cuda()
    x2 = torch.rand(1, 3, 256, 256).cuda()
    Net = EnCANet().cuda()

    out = Net(x1, x2)
    print(out.shape)

    # Calculate GFLOPs & Parameters / 计算 FLOPs 与参数量
    from thop import profile

    flops, params = profile(Net, inputs=(x1, x2))
    print(f"Model FLOPs: {flops / 1e9:.4f} GFLOPs")
    print(f"Model Parameters: {params / 1e6:.2f} M")
