# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


class FCNHead(nn.Module):
    """简化的 FCN Head，不依赖 MMSeg """
    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 num_convs=1,
                 dropout_ratio=0.1,
                 kernel_size=3,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.num_convs = num_convs
        convs = []
        convs.append(
            ConvModule(
                in_channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        for _ in range(num_convs - 1):
            convs.append(
                ConvModule(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x):
        output = self.convs(x)
        # Dropout
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.conv_seg(output)
        return output