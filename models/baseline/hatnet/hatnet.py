import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline.hatnet.hafe import HAFE
from models.baseline.hatnet.block import ChannelChecker
from models.baseline.hatnet.cffi import CFFI

class HATNet(nn.Module):
    def __init__(self, inplanes=224, input_size=224, num_classes=2):
        super().__init__()
        self.inplanes = inplanes
        self.hafe = HAFE()
        self.check_channels = ChannelChecker(self.hafe, self.inplanes, input_size)
        self.cffi = CFFI(self.inplanes, num_classes)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xa, xb):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."
        fa1, fa2, fa3, fa4 = self.hafe(xa)
        fa1, fa2, fa3, fa4 = self.check_channels(fa1, fa2, fa3, fa4)
        fb1, fb2, fb3, fb4 = self.hafe(xb)
        fb1, fb2, fb3, fb4 = self.check_channels(fb1, fb2, fb3, fb4)
        ms_feats = fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4
        change = self.cffi(ms_feats)
        out_size = (h_input, w_input)
        out = F.interpolate(change, size=out_size, mode='bilinear', align_corners=True)
        return out


if __name__ == "__main__":
    x1 = torch.rand(1, 3, 256, 256).cuda()
    x2 = torch.rand(1, 3, 256, 256).cuda()
    Net = HATNet().cuda()

    # Calculate GFLOPs & Parameters / 计算 FLOPs 与参数量
    from thop import profile

    flops, params = profile(Net, inputs=(x1, x2))
    print(f"Model FLOPs: {flops / 1e9:.4f} GFLOPs")
    print(f"Model Parameters: {params / 1e6:.2f} M")