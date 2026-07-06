# lgmmnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline.lgmmnet.encoder import Encoder
from models.baseline.lgmmnet.decoder import Decoder

# LGMM-Net：A Local-Global Encoder and Mask Mamba Decoder Hybrid Network for Remote Sensing Change Detection
class LGMMNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, depths=[3, 3, 4, 3], heads=[4, 4, 4, 4]):
        super().__init__()
        self.drop_path_rate = 0.1
        # 通道数调优
        base_chs = 64
        enc_channels = [base_chs, int(base_chs * 1.5), base_chs * 2, base_chs * 4]
        dec_embed_dim = base_chs * 4

        # shared encoder
        self.enc = Encoder(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                               mlp_ratios=[4, 4, 4, 4], drop_path_rate=self.drop_path_rate, depths=depths)

        # decoder
        self.dec = Decoder(in_channels=enc_channels, embedding_dim=dec_embed_dim, output_nc=num_classes)


    def forward(self, x1, x2):
        enc1_out = self.enc(x1)
        enc2_out = self.enc(x2)
        change_map = self.dec(enc1_out, enc2_out)
        if 'elgc_decoder' in self.setting['decoder_module']:
            return change_map
        output, out_128, out_64, out_32, out_16 = change_map
        mask1 = F.interpolate(out_128, scale_factor=2, mode='bilinear', align_corners=False).clamp(min=-20, max=20)
        mask2 = F.interpolate(out_64, scale_factor=4, mode='bilinear', align_corners=False).clamp(min=-20, max=20)
        mask3 = F.interpolate(out_32, scale_factor=8, mode='bilinear', align_corners=False).clamp(min=-20, max=20)
        mask4 = F.interpolate(out_16, scale_factor=16, mode='bilinear', align_corners=False).clamp(min=-20, max=20)
        return [output, mask1, mask2, mask3, mask4]

