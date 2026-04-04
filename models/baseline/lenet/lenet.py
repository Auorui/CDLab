import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.baseline.lenet.fpn import FPN
from models.baseline.lenet.fcn_head import FCNHead
from models.baseline.lenet.blocks import CDWeights, SwinTV2Block, SELayerIn2Out


class LENet(nn.Module):
    def __init__(self, num_classes=2, dropout_ratio=0.1, model_name='swinv2_base_window8_256.ms_in1k'):
        super().__init__()
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        self.in_channels = [128, 256, 512, 1024]
        self.fpnA = FPN(in_channels=[128, 256, 512, 1024], out_channels=256, num_outs=4)
        self.fpnB = FPN(in_channels=[128, 256, 512, 1024], out_channels=256, num_outs=4)

        self.decode_layersA = nn.Sequential(
            nn.Identity(),
            SwinTV2Block(dim=256, out_dim=256, input_resolution=64,
                         num_heads=4),
            SwinTV2Block(dim=256, out_dim=256, input_resolution=32,
                         num_heads=8),
            SwinTV2Block(dim=256, out_dim=256, input_resolution=16,
                         num_heads=16),
            SwinTV2Block(dim=256, out_dim=256, input_resolution=8,
                         num_heads=32)
        )
        self.decode_layersB = nn.Sequential(
            nn.Identity(),
            SwinTV2Block(dim=256, out_dim=256, input_resolution=64,
                         num_heads=4),
            SwinTV2Block(dim=256, out_dim=256, input_resolution=32,
                         num_heads=8),
            SwinTV2Block(dim=256, out_dim=256, input_resolution=16,
                         num_heads=16),
            SwinTV2Block(dim=256, out_dim=256, input_resolution=8,
                         num_heads=32)
        )

        self.channelA = nn.Sequential(
            nn.Identity(),
            SELayerIn2Out(in_channels=256 * 2, out_channels=256,
                          norm_cfg=norm_cfg),
            SELayerIn2Out(in_channels=256 * 2, out_channels=256,
                          norm_cfg=norm_cfg),
            SELayerIn2Out(in_channels=256 * 2, out_channels=256,
                          norm_cfg=norm_cfg),
            SELayerIn2Out(in_channels=256 * 2, out_channels=256,
                          norm_cfg=norm_cfg)
        )
        self.channelB = nn.Sequential(
            nn.Identity(),
            SELayerIn2Out(in_channels=256 * 2, out_channels=256,
                          norm_cfg=norm_cfg),
            SELayerIn2Out(in_channels=256 * 2, out_channels=256,
                          norm_cfg=norm_cfg),
            SELayerIn2Out(in_channels=256 * 2, out_channels=256,
                          norm_cfg=norm_cfg),
            SELayerIn2Out(in_channels=256 * 2, out_channels=256,
                          norm_cfg=norm_cfg)
        )

        self.sigmoid = nn.Sigmoid()
        self.re_weight3 = CDWeights(256, norm_cfg=norm_cfg)
        self.re_weight2 = CDWeights(256, norm_cfg=norm_cfg)
        self.re_weight1 = CDWeights(256, norm_cfg=norm_cfg)

        self.rwe = nn.Sequential(
            CDWeights(128, norm_cfg=norm_cfg),
            CDWeights(128, norm_cfg=norm_cfg),
            CDWeights(256, norm_cfg=norm_cfg),
            CDWeights(512, norm_cfg=norm_cfg),
            CDWeights(1024, norm_cfg=norm_cfg)
        )

        # Decoder
        self.decode_head = FCNHead(
            in_channels=512,  # stage2
            channels=256,
            num_classes=num_classes,
            num_convs=1,
            dropout_ratio=dropout_ratio,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU')
        )

        # 辅助头
        self.aux_head1 = FCNHead(
            in_channels=512,
            channels=256,
            num_classes=num_classes,
            num_convs=1,
            dropout_ratio=dropout_ratio,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU')
        )

        self.aux_head2 = FCNHead(
            in_channels=512,
            channels=256,
            num_classes=num_classes,
            num_convs=1,
            dropout_ratio=dropout_ratio,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU')
        )

    def change_feature(self, x, y):
        i = 2
        for index in range(0, len(x), i):
            x[index], y[index] = y[index], x[index]
        return x, y

    def extract_feat(self, x_a, x_b):
        ii = 0
        x_a_list = []
        x_b_list = []
        for name, module in self.backbone.named_children():
            if name in ['blocks']:
                x_a = module(x_a)   # 64 x 64
                x_b = module(x_b)
            else:
                x_a = module(x_a)
                x_a = x_a.permute(0, 3, 1, 2)   # nhwc -> nchw
                x_b = module(x_b)
                x_b = x_b.permute(0, 3, 1, 2)
                x_a_nchw, x_b_nchw = self.rwe[ii](x_a, x_b)
                x_a_list.append(x_a_nchw)
                x_b_list.append(x_b_nchw)
                x_a = x_a_nchw.permute(0, 2, 3, 1)
                x_b = x_b_nchw.permute(0, 2, 3, 1)
                ii += 1

        return x_a_list, x_b_list

    def forward(self, img1, img2):
        feat1_list, feat2_list = self.extract_feat(img1, img2)
        feat1_list = feat1_list[1:]   # 去除第一个, 获取layer层特征
        feat2_list = feat2_list[1:]

        feat1_list, feat2_list = self.change_feature(feat1_list, feat2_list)
        feat1_list = self.fpnA(feat1_list)
        feat2_list = self.fpnB(feat2_list)
        feat1_list, feat2_list = self.change_feature(list(feat1_list), list(feat2_list))

        xA_list = [x.permute(0, 2, 3, 1) for x in feat1_list]
        xB_list = [x.permute(0, 2, 3, 1) for x in feat2_list]
        xA1, xA2, xA3, xA4 = xA_list
        xB1, xB2, xB3, xB4 = xB_list

        change_maps = []
        xA4_ = self.decode_layersA[4](xA4)
        xB4_ = self.decode_layersB[4](xB4)
        xA4_ = torch.cat([xA4_, xB4.permute(0, 3, 1, 2)], dim=1)
        xB4_ = torch.cat([xB4_, xA4.permute(0, 3, 1, 2)], dim=1)
        xA4 = self.channelA[4](xA4_)
        xB4 = self.channelB[4](xB4_)
        xA4 = F.interpolate(xA4, scale_factor=2, mode='bilinear', align_corners=False)
        xB4 = F.interpolate(xB4, scale_factor=2, mode='bilinear', align_corners=False)
        # change_maps.append(torch.cat([xA4, xB4], dim=1))

        # xA3 = torch.cat([xA3, xA4.permute(0, 2, 3, 1)], dim=-1)
        # xB3 = torch.cat([xB3, xB4.permute(0, 2, 3, 1)], dim=-1)
        xA3 = xA3 + xB4.permute(0, 2, 3, 1)
        xB3 = xB3 + xA4.permute(0, 2, 3, 1)
        xA3_ = self.decode_layersA[3](xA3)
        xB3_ = self.decode_layersB[3](xB3)
        xA3_ = torch.cat([xA3_, xB3.permute(0, 3, 1, 2)], dim=1)
        xB3_ = torch.cat([xB3_, xA3.permute(0, 3, 1, 2)], dim=1)
        xA3 = self.channelA[3](xA3_)
        xB3 = self.channelB[3](xB3_)
        xA3 = F.interpolate(xA3, scale_factor=2, mode='bilinear', align_corners=False)
        xB3 = F.interpolate(xB3, scale_factor=2, mode='bilinear', align_corners=False)
        xA3, xB3 = self.re_weight3(xA3, xB3)
        change_maps.append(torch.cat([xA3, xB3], dim=1))

        # xA2 = torch.cat([xA2, xA3.permute(0, 2, 3, 1)], dim=-1)
        # xB2 = torch.cat([xB2, xB3.permute(0, 2, 3, 1)], dim=-1)
        xA2 = xA2 + xB3.permute(0, 2, 3, 1)
        xB2 = xB2 + xA3.permute(0, 2, 3, 1)
        xA2_ = self.decode_layersA[2](xA2)
        xB2_ = self.decode_layersB[2](xB2)
        xA2_ = torch.cat([xA2_, xB2.permute(0, 3, 1, 2)], dim=1)
        xB2_ = torch.cat([xB2_, xA2.permute(0, 3, 1, 2)], dim=1)
        xA2 = self.channelA[2](xA2_)
        xB2 = self.channelB[2](xB2_)
        xA2 = F.interpolate(xA2, scale_factor=2, mode='bilinear', align_corners=False)
        xB2 = F.interpolate(xB2, scale_factor=2, mode='bilinear', align_corners=False)
        xA2, xB2 = self.re_weight2(xA2, xB2)
        change_maps.append(torch.cat([xA2, xB2], dim=1))

        # xA1 = torch.cat([xA1, xA2.permute(0, 2, 3, 1)], dim=-1)
        # xB1 = torch.cat([xA1, xB2.permute(0, 2, 3, 1)], dim=-1)
        xA1 = xA1 + xB2.permute(0, 2, 3, 1)
        xB1 = xB1 + xA2.permute(0, 2, 3, 1)
        xA1_ = self.decode_layersA[1](xA1)
        xB1_ = self.decode_layersB[1](xB1)
        xA1_ = torch.cat([xA1_, xB1.permute(0, 3, 1, 2)], dim=1)
        xB1_ = torch.cat([xB1_, xA1.permute(0, 3, 1, 2)], dim=1)
        xA1 = self.channelA[1](xA1_)
        xB1 = self.channelB[1](xB1_)
        xA1, xB1 = self.re_weight1(xA1, xB1)
        change_maps.append(torch.cat([xA1, xB1], dim=1))

        main_out = self.decode_head(change_maps[2])
        aux_out1 = self.aux_head1(change_maps[0])
        aux_out2 = self.aux_head2(change_maps[1])
        size = img1.shape[-2:]
        main_out = F.interpolate(main_out, size=size,
                                 mode='bilinear', align_corners=False)
        aux_out1 = F.interpolate(aux_out1, size=size,
                                 mode='bilinear', align_corners=False)
        aux_out2 = F.interpolate(aux_out2, size=size,
                                 mode='bilinear', align_corners=False)
        # 验证代码会只取第一个，因此要将其放在首位
        return main_out, aux_out1, aux_out2



if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 构造模型
    model = LENet().to(device)
    model.eval()

    xA = torch.randn(2, 3, 256, 256).to(device)
    xB = torch.randn(2, 3, 256, 256).to(device)

    # forward
    with torch.no_grad():
        outputs = model(xA, xB)

    print("输出层数:", len(outputs))
    for i, out in enumerate(outputs):
        print(f"输出{i} shape:", out.shape)

