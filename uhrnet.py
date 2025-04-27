import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.backbone import ConvBnReLU

BN_MOMENTUM = 0.1


class UHRnet(nn.Module):
    def __init__(self, num_classes=16, backbone='UHRNet_W48'):
        super(UHRnet, self).__init__()
        if backbone == 'UHRNet_W18_Small':
            from net.backbone import UHRNet_W18_Small
            self.backbone = UHRNet_W18_Small()
            last_inp_channels = int(279)

        if backbone == 'UHRNet_W18':
            from net.backbone import UHRNet_W18
            self.backbone = UHRNet_W18()
            last_inp_channels = int(279)

        if backbone == 'UHRNet_W48':
            from net.backbone import UHRNet_W48
            self.backbone = UHRNet_W48()
            last_inp_channels = int(48)

        self.head = nn.Sequential()
        self.head.add_module("conv_1",
                             ConvBnReLU(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1,
                                        stride=1, padding=0, bias=True),
                             )
        self.head.add_module("drop_out",nn.Dropout2d(0., False))
        self.head.add_module("cls",
                             nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1,
                                       padding=0)
                             )
        self.softmax = nn.Softmax(dim=1)  # 添加Softmax激活函数

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone(inputs)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.head(x)
        # x = self.softmax(x)
        return x


if __name__ == '__main__':
    images = torch.rand(1, 3, 1024, 1024)
    # print(images)
    model = UHRnet()
    # model = model.to("cuda")
    print(model(images).size())
