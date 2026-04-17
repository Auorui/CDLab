import torch
import torch.nn as nn

class Conv1Relu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv1Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (1, 1), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x

class ChannelChecker(nn.Module):
    def __init__(self, backbone, inplanes, input_size):
        super(ChannelChecker, self).__init__()
        input_sample = torch.randn(1, 3, input_size, input_size)
        f1, f2, f3, f4 = backbone(input_sample)

        channels1 = f1.size(1)
        channels2 = f2.size(1)
        channels3 = f3.size(1)
        channels4 = f4.size(1)

        self.conv1 = Conv1Relu(channels1, inplanes) if (channels1 != inplanes) else None
        self.conv2 = Conv1Relu(channels2, inplanes*2) if (channels2 != inplanes*2) else None
        self.conv3 = Conv1Relu(channels3, inplanes*4) if (channels3 != inplanes*4) else None
        self.conv4 = Conv1Relu(channels4, inplanes*8) if (channels4 != inplanes*8) else None

    def forward(self, f1, f2, f3, f4):
        f1 = self.conv1(f1) if (self.conv1 is not None) else f1
        f2 = self.conv2(f2) if (self.conv2 is not None) else f2
        f3 = self.conv3(f3) if (self.conv3 is not None) else f3
        f4 = self.conv4(f4) if (self.conv4 is not None) else f4

        return f1, f2, f3, f4