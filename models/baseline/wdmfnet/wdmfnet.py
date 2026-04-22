import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from pytorch_wavelets import DWTForward, DWTInverse
from models.baseline.wdmfnet.backbone import mobilenet_v2

# -------------------------------------------------------HFEA----------------------------------------------------------#
class EncoderFusion(nn.Module):
    def __init__(self, inc, midc=32, outc=64):
        super(EncoderFusion, self).__init__()

        if inc is None:
            inc = [16, 24, 32, 96, 320]
        self.inc = inc
        self.midc = midc
        self.outc = outc
        self.fusec = [midc * 3, midc * 3, midc * 3, midc * 2]


        # stage 1
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.inc[0], self.midc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.inc[1], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.inc[2], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # stage 2
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(self.inc[1], self.midc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(self.inc[2], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(self.inc[3], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # stage 3
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(self.inc[2], self.midc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(self.inc[3], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(self.inc[4], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # stage 4
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(self.inc[3], self.midc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True)
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(self.inc[4], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True)
        )

        # aggregation
        self.aggregation_s1 = AggregationModule(self.fusec[0], self.inc[1], self.outc)
        self.aggregation_s2 = AggregationModule(self.fusec[1], self.inc[2], self.outc)
        self.aggregation_s3 = AggregationModule(self.fusec[2], self.inc[3], self.outc)
        self.aggregation_s4 = AggregationModule(self.fusec[3], self.inc[4], self.outc)

    def forward(self, x1, x2, x3, x4, x5):
        s1_f1 = self.conv1_1(x1)
        s1_f2 = self.conv1_2(x2)
        s1_f3 = self.conv1_3(x3)
        s1 = self.aggregation_s1(torch.cat([s1_f1, s1_f2, s1_f3], dim=1), x2)

        s2_f1 = self.conv2_1(x2)
        s2_f2 = self.conv2_2(x3)
        s2_f3 = self.conv2_3(x4)
        s2 = self.aggregation_s2(torch.cat([s2_f1, s2_f2, s2_f3], dim=1), x3)

        s3_f1 = self.conv3_1(x3)
        s3_f2 = self.conv3_2(x4)
        s3_f3 = self.conv3_3(x5)
        s3 = self.aggregation_s3(torch.cat([s3_f1, s3_f2, s3_f3], dim=1), x4)

        s4_f1 = self.conv4_1(x4)
        s4_f2 = self.conv4_2(x5)
        s4 = self.aggregation_s4(torch.cat([s4_f1, s4_f2], dim=1), x5)
        return s1, s2, s3, s4


# feature aggregation
class AggregationModule(nn.Module):
    def __init__(self, fusec, inc, outc):
        super(AggregationModule, self).__init__()
        self.fusec = fusec
        self.inc = inc
        self.outc = outc

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fusec, self.outc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.outc, self.outc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.outc)
        )

        self.conv_identity = nn.Conv2d(self.inc, self.outc, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))
        return c_out


# -------------------------------------------------------WFAM----------------------------------------------------------#
class AlignedModule(nn.Module):
    def __init__(self, channels):
        super(AlignedModule, self).__init__()

        self.wavelet = WaveletAttention(channels)
        self.offset_conv = nn.Conv2d(channels * 2, 2, kernel_size=3, stride=1, padding=1)

    def flow_warp(self, x, flow):
        n, c, h, w = x.size()
        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(x, grid, align_corners=True)
        return output

    def forward(self, x, y):
        x = self.wavelet(x)   # 对 T1 做小波增强
        y = self.wavelet(y)   # 对 T2 做小波增强
        cat = torch.cat([x, y], 1)
        offset = self.offset_conv(cat)   # 预测光流
        warp_y = self.flow_warp(y, offset)  # 对齐 T2
        return x, warp_y


class ConvModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class WaveletAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(wave='haar')

        self.high_HL = ConvModule(channels)
        self.high_LH = ConvModule(channels)
        self.high_HH = ConvModule(channels)
        self.LL_attn = LLAttention(channels)

        self.fuse = nn.Conv2d(channels, channels, kernel_size=1)
        self.fc = nn.Linear(channels, channels, bias=True)

    def forward(self, x):
        x_L, x_H = self.dwt(x)
        x_HL = x_H[0][:, :, 0, :, :]
        x_LH = x_H[0][:, :, 1, :, :]
        x_HH = x_H[0][:, :, 2, :, :]

        x_HL_en = self.high_HL(x_HL)
        x_LH_en = self.high_LH(x_LH)
        x_HH_en = self.high_HH(x_HH)

        x_H_en = torch.stack([x_HL_en, x_LH_en, x_HH_en], dim=2)
        x_L_en = self.LL_attn(x_L)

        x_re = self.idwt((x_L_en, [x_H_en]))
        out = x_re + x
        return out


class LLAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape

        x_flat = x.reshape(B, C, H * W).permute(0, 2, 1)

        q = self.q(x_flat).reshape(B, H * W, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        kv = self.kv(x_flat).reshape(B, H * W, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


# ------------------------------------------------------CFDM-----------------------------------------------------------#
class DiffModule(nn.Module):
    def __init__(self, channels):
        super(DiffModule, self).__init__()

        self.align = AlignedModule(channels)
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1)
        self.attention = SC_Attention(channels)
        self.cbr = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x, y):
        x, y = self.align(x, y)
        diff = torch.abs(x - y)
        con = self.conv(torch.cat([x, y], dim=1))
        attn = self.attention(diff, con)
        diff_en = diff * attn
        con_en = con * attn
        output = torch.cat([diff_en, con_en], dim=1)
        output = self.cbr(output)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.conv3 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.conv7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, max], dim=1)

        attn3 = self.conv3(cat)
        attn5 = self.conv5(cat)
        attn7 = self.conv7(cat)

        attn = attn3 + attn5 + attn7
        attn = self.sigmoid(attn)
        return attn


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_mlp(self.avg_pool(x))
        maxout = self.shared_mlp(self.max_pool(x))
        attn = self.sigmoid(avgout + maxout)
        return attn


class SC_Attention(nn.Module):
    def __init__(self, channels):
        super(SC_Attention, self).__init__()
        self.spatial = SpatialAttention(channels)
        self.channel = ChannelAttention(channels)

    def forward(self, diff, con):
        attn1 = self.spatial(diff)
        attn2 = self.channel(con)
        attn = attn1 * attn2
        return attn


# -------------------------------------------------------MFFM----------------------------------------------------------#
class DecoderFusion(nn.Module):
    def __init__(self, channels):
        super(DecoderFusion, self).__init__()

        self.msa = MSA_Module(channels)

    def forward(self, f1, f2, f3, f4):
        f4 = self.msa(f4)
        f4_up = F.interpolate(f4, scale_factor=(2, 2), mode='bilinear')

        f3 = f3 + f4_up
        f3 = self.msa(f3)
        f3_up = F.interpolate(f3, scale_factor=(2, 2), mode='bilinear')

        f2 = f2 + f3_up
        f2 = self.msa(f2)
        f2_up = F.interpolate(f2, scale_factor=(2, 2), mode='bilinear')

        f1 = f1 + f2_up
        f1 = self.msa(f1)
        return f1, f2, f3, f4


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, channels):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(channels)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


class FeedForward(nn.Module):
    def __init__(self, channels, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden = int(channels * ffn_expansion_factor)

        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1,
                                groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    def __init__(self, channels, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

        self.qkv1conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias)
        self.qkv2conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias)
        self.qkv3conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias)

        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape

        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))

        if mask is not None:
            q = q * mask
            k = k * mask

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        return self.project_out(out)


class MSA_Head(nn.Module):
    def __init__(self, channels=64, num_heads=4, ffn_expansion_factor=4, bias=False):
        super(MSA_Head, self).__init__()
        self.norm1 = LayerNorm(channels)
        self.attn = Attention(channels, num_heads, bias)
        self.norm2 = LayerNorm(channels)
        self.ffn = FeedForward(channels, ffn_expansion_factor, bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.ffn(self.norm2(x))


class MSA_Module(nn.Module):
    def __init__(self, channels=64):
        super(MSA_Module, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.background = MSA_Head(channels)
        self.foreground = MSA_Head(channels)

        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip = x
        mask = self.conv(x)
        mask = torch.sigmoid(mask.detach())   # 前景掩码
        xf = self.foreground(x, mask)      # Foreground
        xb = self.background(x, 1 - mask)  # Background
        x = torch.cat([xb, xf], dim=1)
        x = self.fuse(x)
        out = self.out(skip * x)
        return out


# -----------------------------------------------------WDMFNet---------------------------------------------------------#
class WDMFNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(WDMFNet, self).__init__()
        self.encoder = mobilenet_v2(pretrained=pretrained)
        self.encoder_fusion = EncoderFusion(inc=[16, 24, 32, 96, 320], midc=32, outc=64)
        self.diff = DiffModule(64)
        self.decoder_fusion = DecoderFusion(64)
        self.decoder_out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x1, x2):
        # feature extraction
        x1_0, x1_1, x1_2, x1_3, x1_4 = self.encoder(x1)
        x2_0, x2_1, x2_2, x2_3, x2_4 = self.encoder(x2)

        # feature enhancement
        f1_1, f1_2, f1_3, f1_4 = self.encoder_fusion(x1_0, x1_1, x1_2, x1_3, x1_4)
        f2_1, f2_2, f2_3, f2_4 = self.encoder_fusion(x2_0, x2_1, x2_2, x2_3, x2_4)

        # feature difference
        diff1 = self.diff(f1_1, f2_1)
        diff2 = self.diff(f1_2, f2_2)
        diff3 = self.diff(f1_3, f2_3)
        diff4 = self.diff(f1_4, f2_4)

        # feature fusion
        f1, f2, f3, f4 = self.decoder_fusion(diff1, diff2, diff3, diff4)

        # change map
        f1 = self.decoder_out(f1)
        f2 = self.decoder_out(f2)
        f3 = self.decoder_out(f3)
        f4 = self.decoder_out(f4)

        f1_up = F.interpolate(f1, scale_factor=(4, 4), mode='bilinear')
        # f1_up = torch.sigmoid(f1_up)
        f2_up = F.interpolate(f2, scale_factor=(8, 8), mode='bilinear')
        # f2_up = torch.sigmoid(f2_up)
        f3_up = F.interpolate(f3, scale_factor=(16, 16), mode='bilinear')
        # f3_up = torch.sigmoid(f3_up)
        f4_up = F.interpolate(f4, scale_factor=(32, 32), mode='bilinear')
        # f4_up = torch.sigmoid(f4_up)
        return f1_up, f2_up, f3_up, f4_up



if __name__ == '__main__':
    model = WDMFNet()
    img = torch.randn(1, 3, 256, 256)
    img1 = torch.randn(1, 3, 256, 256)
    res = model(img, img1)
    print(res[0].shape, res[1].shape)
    from thop import profile
    # mmengine_flop_count(model, (3, 512, 512), show_table=True, show_arch=True)
    flops1, params1 = profile(model, inputs=(img, img1))
    print("flops=G", flops1)
    print("parms=M", params1)
