"""
AdaStyle-MAMB 融合模块
实现深度特征融合的高级融合机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoStageUniFiREFusion(nn.Module):
    """
    两阶段MAMB特征融合模块 (简化版，移除GFE)

    实现双阶段的深度特征融合：
    1. 第一阶段：时域MAMB比较 - 提取T1和T2之间的变化特征
    2. 第二阶段：域MAMB融合 - 将解码器语义特征与变化特征融合

    简化后直接使用原始编码器特征，无风格净化处理
    """

    def __init__(self, encoder_channels, decoder_channels=None,
                 temporal_fusion_method="channel_interleave",
                 spatial_fusion_method="channel_interleave",
                 use_temporal_se=False,
                 use_spatial_se=False):
        """
        初始化融合模块

        Args:
            encoder_channels: 编码器特征通道数
            decoder_channels: 解码器特征通道数 (None表示第一个stage无解码器输入)
            temporal_fusion_method: 时域融合方法 (T1+T2)，可选值:
                - "channel_interleave": 通道交错+分组卷积 (默认)
                - "channel_concat": 通道拼接+标准卷积
                - "hadamard": 哈达玛积
            spatial_fusion_method: 空间融合方法 (解码器+变化特征)，可选值:
                - "channel_interleave": 通道交错+分组卷积 (默认)
                - "channel_concat": 通道拼接+标准卷积
                - "hadamard": 哈达玛积
            use_temporal_se: 是否在时域融合中使用SE模块
            use_spatial_se: 是否在空间融合中使用SE模块
        """
        super(TwoStageUniFiREFusion, self).__init__()

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.temporal_fusion_method = temporal_fusion_method
        self.spatial_fusion_method = spatial_fusion_method
        self.use_temporal_se = use_temporal_se
        self.use_spatial_se = use_spatial_se

        # 第一阶段：时域融合 - 处理T1和T2编码器特征
        self.time_mixer = UniFiREBlockDual(
            in_channels=encoder_channels,
            fusion_method=temporal_fusion_method,
            use_se=use_temporal_se
        )

        # 第二阶段相关组件
        if decoder_channels is not None and decoder_channels != encoder_channels:
            # 解码器通道适配器：将解码器特征适配为编码器通道数
            self.decoder_adapter = nn.Sequential(
                nn.Conv2d(self.decoder_channels, self.encoder_channels, kernel_size=1, bias=True),
                nn.BatchNorm2d(self.encoder_channels),
                nn.SiLU()
            )
            # 输出适配器：将最终结果调整回编码器通道数
            self.output_adapter = nn.Sequential(
                nn.Conv2d(self.encoder_channels, self.encoder_channels, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(self.encoder_channels),
                nn.SiLU()
            )
        else:
            self.decoder_adapter = None
            self.output_adapter = None

        # 第二阶段：空间融合 - 统一使用编码器通道数
        self.domain_mixer = UniFiREBlockDual(
            in_channels=encoder_channels,
            fusion_method=spatial_fusion_method,
            use_se=use_spatial_se
        )

    def forward(self, x_decoder, skip_T1, skip_T2):
        """
        两阶段融合的前向传播 (简化版，无GFE)

        Args:
            x_decoder: 来自解码器深层的语义特征 (B, decoder_channels, H, W) [可为None，表示第一个stage]
            skip_T1: 第一时相的跳跃连接特征 (B, encoder_channels, H, W)
            skip_T2: 第二时相的跳跃连接特征 (B, encoder_channels, H, W)

        Returns:
            融合后的特征张量 (B, encoder_channels, H, W)
        """
        # 第一阶段：直接使用原始特征进行时域比较 (无GFE风格净化)
        change_feature = self.time_mixer(skip_T1, skip_T2)

        # 第二阶段：跨域MAMB融合
        if x_decoder is not None:
            # 如果有解码器特征，进行第二阶段融合
            if self.decoder_adapter is not None:
                # 将解码器特征通道数适配为编码器通道数
                x_decoder_adapted = self.decoder_adapter(x_decoder)
            else:
                x_decoder_adapted = x_decoder

            # 将语义特征和变化特征进行融合
            fused_output = self.domain_mixer(x_decoder_adapted, change_feature)

            # 如果需要，调整输出
            final_output = self.output_adapter(fused_output) if self.output_adapter is not None else fused_output
        else:
            # 第一个stage没有解码器特征，直接返回时相比较结果
            final_output = change_feature

        return final_output


# SEBlock 已移除（简化为固定不使用 SE）


class UniFiREBlock(nn.Module):
    """
    统一的融合精化模块

    用于替代UDD模块和复杂的MAMB融合，提供统一的特征处理架构
    结构：Conv3x3 -> BN -> SiLU -> (可选SE) -> Conv3x3 -> BN -> (残差连接) -> SiLU
    """

    def __init__(self, in_channels, out_channels, use_se=False, use_residual=True):
        """
        初始化融合精化模块

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            use_se: 是否使用SE注意力模块
            use_residual: 是否使用残差连接
        """
        super(UniFiREBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_se = use_se
        self.use_residual = use_residual

        # 第一层：Conv3x3 -> BatchNorm -> SiLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.silu1 = nn.SiLU()

        # 精简后不使用 SE
        self.se = None

        # 第二层：Conv3x3 -> BatchNorm
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接的skip路径
        if use_residual:
            if in_channels != out_channels:
                # 通道数不同时，使用1x1卷积调整
                self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            else:
                # 通道数相同时，直接连接
                self.skip_conv = None
        else:
            self.skip_conv = None

        # 最终激活函数
        self.silu_final = nn.SiLU()

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 (B, in_channels, H, W)

        Returns:
            处理后的特征 (B, out_channels, H, W)
        """
        # 保存残差连接的输入
        if self.use_residual:
            if self.skip_conv is not None:
                residual = self.skip_conv(x)
            else:
                residual = x

        # 主分支处理
        # 第一层：Conv3x3 -> BatchNorm -> SiLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu1(out)

        # 可选的SE注意力
        if self.se is not None:
            out = self.se(out)

        # 第二层：Conv3x3 -> BatchNorm
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        if self.use_residual:
            out = out + residual

        # 最终激活
        out = self.silu_final(out)

        return out


class UniFiREBlockDual(nn.Module):
    """
    双输入融合精化模块

    专门用于双特征融合（如T1+T2或解码器+变化特征），基于FusionRefineBlock架构
    支持三种融合方式：通道交错、通道拼接、哈达玛积
    """

    def __init__(self, in_channels, fusion_method="channel_concat", use_se=False):
        """
        初始化双输入融合精化模块

        Args:
            in_channels: 输入特征的通道数（每个输入）
            fusion_method: 融合方法，可选值:
                - "channel_interleave": 通道交错+分组卷积
                - "channel_concat": 通道拼接+标准卷积
                - "hadamard": 哈达玛积
            use_se: 是否使用SE注意力模块
        """
        super(UniFiREBlockDual, self).__init__()

        self.in_channels = in_channels
        self.fusion_method = fusion_method
        self.use_se = use_se

        if fusion_method in ["channel_interleave", "channel_concat"]:
            # 对于需要卷积的融合方法，使用FusionRefineBlock处理融合后的特征
            self.fusion_refine = UniFiREBlock(
                in_channels=2 * in_channels,  # 融合后的通道数
                out_channels=in_channels,  # 输出通道数
                use_se=use_se,
                use_residual=True  # 启用残差连接提升性能
            )
        elif fusion_method == "hadamard":
            # 哈达玛积后的精化处理
            self.fusion_refine = UniFiREBlock(
                in_channels=in_channels,  # 哈达玛积后通道数不变
                out_channels=in_channels,
                use_se=use_se,
                use_residual=True  # 哈达玛积使用残差连接
            )
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")

    def forward(self, x1, x2):
        """
        前向传播

        Args:
            x1: 第一个输入特征 (B, C, H, W)
            x2: 第二个输入特征 (B, C, H, W)

        Returns:
            融合后的特征 (B, C, H, W)
        """
        if self.fusion_method == "channel_interleave":
            # 通道交错排列
            mixed = torch.stack((x1, x2), dim=2)  # (B, C, 2, H, W)
            mixed = mixed.reshape(x1.shape[0], -1, x1.shape[2], x1.shape[3])  # (B, 2*C, H, W)

        elif self.fusion_method == "channel_concat":
            # 通道拼接
            mixed = torch.cat((x1, x2), dim=1)  # (B, 2*C, H, W)

        elif self.fusion_method == "hadamard":
            # 哈达玛积
            mixed = x1 * x2  # (B, C, H, W)

        # 通过FusionRefineBlock进行精化处理
        output = self.fusion_refine(mixed)

        return output 