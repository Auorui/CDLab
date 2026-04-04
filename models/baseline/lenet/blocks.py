import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torch.utils.checkpoint as checkpoint
from torchvision.models._utils import _make_divisible
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer, build_conv_layer
from mmengine.utils import is_tuple_of
from mmengine.model import BaseModule
from timm.layers import to_2tuple
from timm.models.swin_transformer_v2 import PatchMerging, SwinTransformerV2Block
from typing import Tuple, Union
_int_or_tuple_2_t = Union[int, Tuple[int, int]]

class BasicBlock(BaseModule):
    """Basic block for ResNet."""

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

class SwinTV2Block(nn.Module):
    def __init__(
            self,
            dim: int,
            out_dim: int,
            input_resolution: _int_or_tuple_2_t,
            depth: int = 2,
            num_heads: int = 8,
            window_size: _int_or_tuple_2_t = 8,
            downsample: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            pretrained_window_size: _int_or_tuple_2_t = 0,
            output_nchw: bool = False,
    ) -> None:
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            downsample: Use downsample layer at start of the block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
            pretrained_window_size: Local window size in pretraining.
            output_nchw: Output tensors on NCHW format instead of NHWC.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        self.output_nchw = output_nchw
        self.grad_checkpointing = False
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])

        # patch merging / downsample layer
        if downsample:
            self.downsample = PatchMerging(dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerV2Block(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
            )
            for i in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x.permute(0, 3, 1, 2)

    def _init_respostnorm(self) -> None:
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

class CDWeights(nn.Module):
    def __init__(self, channels=128, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(CDWeights, self).__init__()
        self.convA = BasicBlock(channels, planes=channels, norm_cfg=norm_cfg)
        self.convB = BasicBlock(channels, planes=channels, norm_cfg=norm_cfg)
        self.sigmoid = nn.Sigmoid()

    def spatial_difference(self, xA, xB):
        # (N, C, H, W) → 展平空间 → (N*H*W, C)
        xA_flat = xA.permute(0, 2, 3, 1).reshape(-1, xA.size(1))
        xB_flat = xB.permute(0, 2, 3, 1).reshape(-1, xB.size(1))
        cosine_sim = F.cosine_similarity(xA_flat, xB_flat, dim=1)
        cosine_sim = cosine_sim.view(xA.size(0), xA.size(2), xA.size(3))
        cosine_sim = cosine_sim.unsqueeze(1)
        c_weights = 1 - self.sigmoid(cosine_sim)
        return c_weights

    def channel_difference(self, xA, xB):
        N, C, H, W = xA.shape
        # 展平空间维度 → (N, C, H*W)
        xA_flat = xA.view(N, C, -1)
        xB_flat = xB.view(N, C, -1)
        cosine_sim = 1 - self.sigmoid(F.cosine_similarity(xA_flat, xB_flat, dim=2))
        hw_weights = cosine_sim.unsqueeze(-1).unsqueeze(-1)
        return hw_weights

    def forward(self, xA, xB):
        # 计算空间权重和通道权重
        c_weights = self.spatial_difference(xA, xB)  # (2, 1, 32, 32)
        hw_weights = self.channel_difference(xA, xB)  # (2, 128, 1, 1)

        # 将 c_weights 扩展到与 hw_weights 相同的形状
        c_weights_expanded = c_weights.expand(-1, hw_weights.size(1), -1, -1)  # (2, 128, 32, 32)

        # 合并权重 (比如可以选择相乘，也可以进行加权平均)
        combined_weights = c_weights_expanded * hw_weights  # (2, 128, 32, 32)

        # 对 xA 和 xB 进行加权处理
        xA_weighted = xA * combined_weights
        xB_weighted = xB * combined_weights

        # 通过卷积层处理
        xA_d = self.convA(xA_weighted)
        xB_d = self.convB(xB_weighted)

        # 得到最终输出
        outA = xA_d + xA
        outB = xB_d + xB

        return outA, outB

class SELayerIn2Out(nn.Module):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configured
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configured by the first dict and the
            second activation layer will be configured by the second dict.
            Default: (dict(type='ReLU'), dict(type='HSigmoid', bias=3.0,
            divisor=6.0)).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=None,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'),
                          dict(type='HSigmoid', bias=3.0, divisor=6.0))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=_make_divisible(in_channels // ratio, 8),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=_make_divisible(in_channels // ratio, 8),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])
        self.conv3 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1],
            norm_cfg=norm_cfg)

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        x = x * out
        x = self.conv3(x)
        return x