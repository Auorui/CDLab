import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.baseline.megnet.modules import TransformerDecoder, Transformer, TransformerDecoder2
from models.baseline.megnet.memory import Memory
from models.baseline.megnet.backbone import build_swin_backbone

class token_encoder(nn.Module):
    def __init__(self, in_chan = 32, token_len = 4, heads = 8):
        super(token_encoder, self).__init__()
        self.token_len = token_len
        self.conv_a = nn.Conv2d(in_chan, token_len, kernel_size=1, padding=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, token_len, in_chan))
        self.transformer = Transformer(dim=in_chan, depth=1, heads=heads, dim_head=64, mlp_dim=64, dropout=0)

    def forward(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()

        tokens = torch.einsum('bln, bcn->blc', spatial_attention, x)

        tokens += self.pos_embedding
        x = self.transformer(tokens)
        return x

class token_decoder(nn.Module):
    def __init__(self, in_chan = 32, heads = 8):
        super(token_decoder, self).__init__()
        self.transformer_decoder = TransformerDecoder(dim=in_chan, depth=1, heads=heads, dim_head=True, mlp_dim=in_chan*2, dropout=0,softmax=in_chan)

    def forward(self, x, m):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x


class context_aggregator(nn.Module):
    def __init__(self, in_chan=32):
        super(context_aggregator, self).__init__()
        self.token_encoder = token_encoder(in_chan=in_chan, token_len=8)
        self.token_decoder = token_decoder(in_chan = in_chan, heads = 8)

    def forward(self, feature):
        token = self.token_encoder(feature)
        out = self.token_decoder(feature, token)
        return out


class Classifier(nn.Module):
    def __init__(self, in_chan=64, n_class=2):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(
                            nn.Conv2d(in_chan * 2, in_chan, kernel_size=3, padding=1, stride=1, bias=False),
                            nn.BatchNorm2d(in_chan),
                            nn.ReLU(),
                            nn.Conv2d(in_chan, n_class, kernel_size=3, padding=1, stride=1))
    def forward(self, x):
        x = self.head(x)
        return x

class MeGNet(nn.Module):
    def __init__(self,  backbone='swin_tiny_p4w7', img_chan=3, num_classes=2, pretrained=True, pretrained_path=''):
        """
        swin_tiny_p4w7, swin_small_p4w7, swin_base_p4w7, swin_large_p4w12
        """
        super(MeGNet, self).__init__()
        self.backbone, self.channels_blocks, self.do_upsample = build_swin_backbone(
            backbone, pretrained, img_chan, pretrained_path=pretrained_path)
        self.CA_s32 = context_aggregator(in_chan=768)
        self.CA_s16 = context_aggregator(in_chan=512)
        self.CA_s8 = context_aggregator(in_chan=160)
        self.CA_s4 = context_aggregator(in_chan=80)

        self.conv_s16 = nn.Conv2d(1152, 512, kernel_size=3, padding=1)
        self.conv_s8 = nn.Conv2d(704, 160, kernel_size=3, padding=1)
        self.conv_s4 = nn.Conv2d(256, 80, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(160, 64, kernel_size=3, padding=1) #+memory

        self.m_items = F.normalize(torch.rand((num_classes, 64), dtype=torch.float), dim=1)
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)
        self.classifier = Classifier(in_chan=64, n_class=num_classes)
        self.memory = Memory(memory_size=num_classes, feature_dim=64, key_dim=64)

    def forward(self, img1, img2, memory_items=None):
        # CNN backbone, feature extractor
        out1_s4, out1_s8, out1_s16, out1_s32  = self.backbone(img1)
        out2_s4, out2_s8, out2_s16, out2_s32 = self.backbone(img2)

        x1_s32 = self.CA_s32(out1_s32)
        x2_s32 = self.CA_s32(out2_s32)

        x1_s32 = F.interpolate(x1_s32, size=out1_s16.shape[2:], mode='bicubic', align_corners=True)
        x2_s32 = F.interpolate(x2_s32, size=out2_s16.shape[2:], mode='bicubic', align_corners=True)

        out1_s16 = self.conv_s16(torch.cat([x1_s32, out1_s16], dim=1))
        out2_s16 = self.conv_s16(torch.cat([x2_s32, out2_s16], dim=1))

        # context aggregate (scale 16, scale 8, scale 4)
        x1_s16= self.CA_s16(out1_s16)
        x2_s16 = self.CA_s16(out2_s16)

        x1_s16 = F.interpolate(x1_s16, size=out1_s8.shape[2:], mode='bicubic', align_corners=True)
        x2_s16 = F.interpolate(x2_s16, size=out2_s8.shape[2:], mode='bicubic', align_corners=True)
        out1_s8 = self.conv_s8(torch.cat([x1_s16, out1_s8], dim=1))
        out2_s8 = self.conv_s8(torch.cat([x2_s16, out2_s8], dim=1))

        x1_s8 = self.CA_s8(out1_s8)
        x2_s8 = self.CA_s8(out2_s8)

        x1_s8 = F.interpolate(x1_s8, size=out1_s4.shape[2:], mode='bicubic', align_corners=True)
        x2_s8 = F.interpolate(x2_s8, size=out2_s4.shape[2:], mode='bicubic', align_corners=True)
        out1_s4 = self.conv_s4(torch.cat([x1_s8, out1_s4], dim=1))
        out2_s4 = self.conv_s4(torch.cat([x2_s8, out2_s4], dim=1))

        x1 = self.CA_s4(out1_s4)
        x2 = self.CA_s4(out2_s4)
        x2 = F.interpolate(x2, size=x1.shape[2:], mode='bicubic', align_corners=True)

        x = torch.cat([x1, x2], dim=1)
        x =self.conv(x)

        if self.training:
            # 训练, 更新记忆, 返回预测和更新的记忆
            updated_x, self.m_items = self.memory(x, self.m_items, train=True)
            x = self.classifier(updated_x)
            x = F.interpolate(x, size=img1.shape[2:], mode='bicubic', align_corners=True)
            return x, self.m_items

        else:
            # 测试, 只读取记忆, 不更新, 仅返回预测结果
            memory_to_use = memory_items if memory_items is not None else self.m_items
            updated_x = self.memory(x, memory_to_use, train=False)
            x = self.classifier(updated_x)
            x = F.interpolate(x, size=img1.shape[2:], mode='bicubic', align_corners=True)
            return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

# MeGNet Adapter
class MeGNetApt(nn.Module):
    """MeGNet适配器，使其符合标准CD模型接口"""
    def __init__(self, backbone='swin_tiny_p4w7', img_chan=3, num_classes=2, backbone_pretrained=True,
                 backbone_pretrained_path='', memory_path=None, **kwargs):
        """
        Args:
            backbone: 主干网络类型
            img_chan: 输入图像通道数
            num_classes: 分类数
            backbone_pretrained: 是否加载主干网络预训练权重
            backbone_pretrained_path: 主干网络预训练权重路径（可选）
        """
        super().__init__()
        self.megnet = MeGNet(
            backbone=backbone,
            img_chan=img_chan,
            num_classes=num_classes,
            pretrained=backbone_pretrained,
            pretrained_path=backbone_pretrained_path
        )

        # 存储记忆项（初始化为随机或从检查点加载）
        self.memory_items = None

    def forward(self, img1, img2):
        """
        统一的前向接口：输入两张图片，输出预测结果
        内部根据训练状态自动处理记忆项
        """
        if self.training:
            # 训练模式：直接调用，返回预测和更新后的记忆
            prob, new_memory = self.megnet(img1, img2)
            # 保存记忆项供后续使用
            self.memory_items = new_memory.detach().clone()
            return prob
        else:
            # 评估模式：传入保存的记忆项
            # MeGNet的forward已经支持memory_items参数
            output = self.megnet(img1, img2, self.memory_items)
            return output

    def get_memory_items(self):
        """获取当前的记忆项"""
        return self.memory_items

    def set_memory_items(self, memory_items):
        """设置记忆项"""
        self.memory_items = memory_items

    def state_dict(self, *args, **kwargs):
        """重写state_dict，保存记忆项"""
        state_dict = super().state_dict(*args, **kwargs)
        if self.memory_items is not None:
            # 将记忆项添加到state_dict中
            state_dict['memory_items'] = self.memory_items.cpu()
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        """重写load_state_dict，加载记忆项"""
        memory_items = state_dict.pop('memory_items', None)
        result = super().load_state_dict(state_dict, strict)

        if memory_items is not None:
            self.memory_items = memory_items
            print(f"Successfully loaded memory item, shape: {memory_items.shape}")
        else:
            print("Note: There are no memory items in the checkpoints; random initialization memory items will be used instead.")
        return result

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_train = MeGNetApt(
        backbone='swin_tiny_p4w7',
        num_classes=2
    )
    model_train.train()
    model_train = model_train.to(device)

    img1 = torch.randn(2, 3, 512, 512).to(device)
    img2 = torch.randn(2, 3, 512, 512).to(device)

    # 训练模式前向
    out_train = model_train(img1, img2)
    print(f"训练模式输出形状: {out_train.shape}")
    print(f"训练后记忆项形状: {model_train.get_memory_items().shape}")

    # 测试评估模式
    model_eval = MeGNetApt(
        backbone='swin_tiny_p4w7',
        num_classes=2
    )
    model_eval.eval()
    model_eval = model_eval.to(device)

    # 设置记忆项（模拟从训练保存的记忆）
    memory_items = torch.randn(2, 64).to(device)
    model_eval.set_memory_items(memory_items)
    print(f"设置记忆项形状: {memory_items.shape}")

    # 评估模式前向
    with torch.no_grad():
        out_eval = model_eval(img1, img2)
    print(f"评估模式输出形状: {out_eval.shape}")

    # 测试state_dict保存和加载
    print("\n" + "=" * 50)
    print("测试state_dict保存和加载")
    print("=" * 50)

    state_dict = model_train.state_dict()
    print(f"state_dict包含记忆项: {'memory_items' in state_dict}")

    model_new = MeGNetApt(
        backbone='swin_tiny_p4w7',
        num_classes=2
    )
    model_new.load_state_dict(state_dict)
    model_new.eval()
    model_new = model_new.to(device)

    loaded_memory = model_new.get_memory_items()
    print(f"加载后的记忆项形状: {loaded_memory.shape if loaded_memory is not None else 'None'}")

    with torch.no_grad():
        out_new = model_new(img1, img2)
    print(f"加载后模型输出形状: {out_new.shape}")