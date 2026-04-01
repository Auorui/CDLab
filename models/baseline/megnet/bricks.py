import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (Identity, GroupNorm, SyncBatchNorm, BatchNorm1d, BatchNorm2d,
    BatchNorm3d, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.act = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.act(x + 3) / 6


class HardSigmoid(nn.Module):
    def __init__(self, bias=1.0, divisor=2.0, min_value=0.0, max_value=1.0):
        super(HardSigmoid, self).__init__()
        assert divisor != 0, 'divisor is not allowed to be equal to zero'
        self.bias = bias
        self.divisor = divisor
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = (x + self.bias) / self.divisor
        return x.clamp_(self.min_value, self.max_value)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def build_normalization(norm_type='batchnorm2d', instanced_params=(0, {}), only_get_all_supported=False, **kwargs):
    supported_dict = {
        'identity': Identity,
        'layernorm': LayerNorm,
        'groupnorm': GroupNorm,
        'batchnorm1d': BatchNorm1d,
        'batchnorm2d': BatchNorm2d,
        'batchnorm3d': BatchNorm3d,
        'syncbatchnorm': SyncBatchNorm,
        'instancenorm1d': InstanceNorm1d,
        'instancenorm2d': InstanceNorm2d,
        'instancenorm3d': InstanceNorm3d,
    }
    if only_get_all_supported: return list(supported_dict.values())
    assert norm_type in supported_dict, 'unsupport norm_type %s...' % norm_type
    if norm_type == 'groupnorm':
        norm_layer = supported_dict[norm_type](instanced_params[0]//8, instanced_params[0], **instanced_params[1])
    else:
        norm_layer = supported_dict[norm_type](instanced_params[0], **instanced_params[1])
    return norm_layer

def build_activation(activation_type, **kwargs):
    supported_activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'relu6': nn.ReLU6,
        'prelu': nn.PReLU,
        'sigmoid': nn.Sigmoid,
        'hardswish': HardSwish,
        'identity': nn.Identity,
        'leakyrelu': nn.LeakyReLU,
        'hardsigmoid': HardSigmoid,
    }
    assert activation_type in supported_activations, 'unsupport activation type %s...' % activation_type
    return supported_activations[activation_type](**kwargs)

def build_dropout(dropout_type, **kwargs):
    supported_dropouts = {
        'droppath': DropPath,
        'dropout': nn.Dropout,
        'dropout2d': nn.Dropout2d,
        'dropout3d': nn.Dropout3d,
    }
    assert dropout_type in supported_dropouts, 'unsupport dropout type %s...' % dropout_type
    return supported_dropouts[dropout_type](**kwargs)

class FFN(nn.Module):
    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2, act_cfg=None, ffn_drop=0., dropout_cfg=None, add_identity=True, **kwargs):
        super(FFN, self).__init__()
        assert num_fcs >= 2, f'num_fcs should be no less than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation(act_cfg['type'], **act_cfg['opts'])
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Sequential(
                nn.Linear(in_channels, feedforward_channels),
                self.activate,
                nn.Dropout(ffn_drop)
            ))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        if dropout_cfg:
            self.dropout_layer = build_dropout(dropout_cfg['type'], **dropout_cfg['opts'])
        else:
            self.dropout_layer = torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

class AdaptivePadding(nn.Module):
    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super(AdaptivePadding, self).__init__()
        assert padding in ('same', 'corner')
        self.padding = padding
        self.kernel_size = self.totuple(kernel_size)
        self.stride = self.totuple(stride)
        self.dilation = self.totuple(dilation)

    def getpadshape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.getpadshape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x

    @staticmethod
    def totuple(x):
        if isinstance(x, int): return (x, x)
        assert isinstance(x, tuple) and (len(x) == 2)
        return x


class PatchEmbed(nn.Module):
    '''Image to Patch Embedding'''
    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 kernel_size=16,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None):
        super(PatchEmbed, self).__init__()
        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        stride = AdaptivePadding.totuple(stride)
        dilation = AdaptivePadding.totuple(dilation)
        kernel_size = AdaptivePadding.totuple(kernel_size)

        self.adap_padding = None
        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding
            )
            padding = 0

        padding = AdaptivePadding.totuple(padding)
        self.projection = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            dilation=dilation
        )

        self.norm = None
        if norm_cfg is not None:
            self.norm = build_normalization(norm_cfg['type'], (embed_dims, norm_cfg['opts']))

        self.init_input_size = None
        self.init_out_size = None
        if input_size:
            input_size = AdaptivePadding.totuple(input_size)
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.getpadshape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            h_out = (input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)

    def forward(self, x):
        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x, out_size

    def zerowdlayers(self):
        '''返回需要零权重衰减的层'''
        if self.norm is None:
            return {}
        return {'PatchEmbed.norm': self.norm}

    def nonzerowdlayers(self):
        '''返回需要非零权重衰减的层'''
        return {'PatchEmbed.projection': self.projection}


class PatchMerging(nn.Module):
    '''Merge patch feature map'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=None):
        super(PatchMerging, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride:
            self.stride = stride
        else:
            self.stride = kernel_size

        stride = AdaptivePadding.totuple(self.stride)
        dilation = AdaptivePadding.totuple(dilation)
        kernel_size = AdaptivePadding.totuple(kernel_size)

        self.adap_padding = None
        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding
            )
            padding = 0

        padding = AdaptivePadding.totuple(padding)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride
        )

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels
        self.norm = None
        if norm_cfg is not None:
            self.norm = build_normalization(norm_cfg['type'], (sample_dim, norm_cfg['opts']))

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, x, input_size):
        B, L, C = x.shape
        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])
        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]

        x = self.sampler(x)  # C -> (kernel_size[0]*kernel_size[1]*C)  H -> H/stride  W -> W/stride
        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1) - 1) // \
                self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1) - 1) // \
                self.sampler.stride[1] + 1
        output_size = (out_h, out_w)

        x = x.transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        x = self.reduction(x)
        return x, output_size

    def zerowdlayers(self):
        '''返回需要零权重衰减的层'''
        if self.norm is None:
            return {}
        return {'PatchMerging.norm': self.norm}

    def nonzerowdlayers(self):
        '''返回需要非零权重衰减的层'''
        return {'PatchMerging.reduction': self.reduction}