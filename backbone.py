# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch.nn as nn
import math

import torch
import torch.nn.functional as F

__all__ = ["UHRNet_W18_Small", "UHRNet_W18", "UHRNet_W48"]

from torch.nn import init

from net.bra_nchw import nchwBRA
from net.do_conv_pytorch_1_10 import DOConv2d


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        if padding == 'same':
            pad = (kernel_size - 1) // 2
        else:
            pad = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        return self.ReLU(self.bn(self.conv(x)))


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        if padding == 'same':
            pad = (kernel_size - 1) // 2
        else:
            pad = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)

    def forward(self, x):
        return self.bn(self.conv(x))


class UHRNet(nn.Module):
    """
    The UHRNet implementation based on PaddlePaddle.
    The original article refers to
    Jian Wang, et, al. "U-HRNet: Delving into Improving Semantic Representation of High Resolution Network for Dense Prediction"
    (https://arxiv.org/pdf/2210.07140.pdf).
    Args:
        in_channels (int, optional): The channels of input image. Default: 3.
        pretrained (str): The path of pretrained model.
        stage1_num_modules (int): Number of modules for stage1. Default 1.
        stage1_num_blocks (list): Number of blocks per module for stage1. Default [4].
        stage1_num_channels (list): Number of channels per branch for stage1. Default [64].
        stage2_num_modules (int): Number of modules for stage2. Default 1.
        stage2_num_blocks (list): Number of blocks per module for stage2. Default [4, 4]
        stage2_num_channels (list): Number of channels per branch for stage2. Default [18, 36].
        stage3_num_modules (int): Number of modules for stage3. Default 5.
        stage3_num_blocks (list): Number of blocks per module for stage3. Default [4, 4]
        stage3_num_channels (list): Number of channels per branch for stage3. Default [36, 72].
        stage4_num_modules (int): Number of modules for stage4. Default 2.
        stage4_num_blocks (list): Number of blocks per module for stage4. Default [4, 4]
        stage4_num_channels (list): Number of channels per branch for stage4. Default [72. 144].
        stage5_num_modules (int): Number of modules for stage5. Default 2.
        stage5_num_blocks (list): Number of blocks per module for stage5. Default [4, 4]
        stage5_num_channels (list): Number of channels per branch for stage5. Default [144, 288].
        stage6_num_modules (int): Number of modules for stage6. Default 1.
        stage6_num_blocks (list): Number of blocks per module for stage6. Default [4, 4]
        stage6_num_channels (list): Number of channels per branch for stage6. Default [72. 144].
        stage7_num_modules (int): Number of modules for stage7. Default 1.
        stage7_num_blocks (list): Number of blocks per module for stage7. Default [4, 4]
        stage7_num_channels (list): Number of channels per branch for stage7. Default [36, 72].
        stage8_num_modules (int): Number of modules for stage8. Default 1.
        stage8_num_blocks (list): Number of blocks per module for stage8. Default [4, 4]
        stage8_num_channels (list): Number of channels per branch for stage8. Default [18, 36].
        stage9_num_modules (int): Number of modules for stage9. Default 1.
        stage9_num_blocks (list): Number of blocks per module for stage9. Default [4]
        stage9_num_channels (list): Number of channels per branch for stage9. Default [18].
        has_se (bool): Whether to use Squeeze-and-Excitation module. Default False.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 pretrained=None,
                 stage1_num_modules=1,
                 stage1_num_blocks=(4,),
                 stage1_num_channels=(64,),
                 stage2_num_modules=1,
                 stage2_num_blocks=(4, 4),
                 stage2_num_channels=(18, 36),
                 stage3_num_modules=5,
                 stage3_num_blocks=(4, 4),
                 stage3_num_channels=(36, 72),
                 stage4_num_modules=2,
                 stage4_num_blocks=(4, 4),
                 stage4_num_channels=(72, 144),
                 stage5_num_modules=2,
                 stage5_num_blocks=(4, 4),
                 stage5_num_channels=(144, 288),
                 stage6_num_modules=1,
                 stage6_num_blocks=(4, 4),
                 stage6_num_channels=(72, 144),
                 stage7_num_modules=1,
                 stage7_num_blocks=(4, 4),
                 stage7_num_channels=(36, 72),
                 stage8_num_modules=1,
                 stage8_num_blocks=(4, 4),
                 stage8_num_channels=(18, 36),
                 stage9_num_modules=1,
                 stage9_num_blocks=(4,),
                 stage9_num_channels=(18,),
                 has_se=False,
                 align_corners=False):
        super(UHRNet, self).__init__()
        self.has_se = has_se
        self.align_corners = align_corners
        state1 = 'encoder'
        state2 = 'decoder'
        self.feat_channels = [
            sum([
                stage5_num_channels[-1], stage6_num_channels[-1],
                stage7_num_channels[-1], stage8_num_channels[-1],
                stage9_num_channels[-1]
            ]) // 2
        ]

        cur_stride = 1
        # stem net
        self.conv_layer1_1 = ConvBnReLU(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding='same')
        cur_stride *= 2

        self.conv_layer1_2 = ConvBnReLU(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding='same')
        cur_stride *= 2

        self.la1 = Layer1(
            num_channels=64,
            num_blocks=stage1_num_blocks[0],
            num_filters=stage1_num_channels[0],
            has_se=has_se,
            name="layer2")

        self.tr1 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage1_num_channels[0] * 4,
            stride_cur=[
                cur_stride * (2 ** i) for i in range(len(stage2_num_channels))
            ],
            out_channels=stage2_num_channels,
            align_corners=self.align_corners,
            name="tr1")
        self.st2 = Stage(
            num_channels=stage2_num_channels,
            num_modules=stage2_num_modules,
            num_blocks=stage2_num_blocks,
            num_filters=stage2_num_channels,
            has_se=self.has_se,
            name="st2",
            align_corners=align_corners,
            state=state1
            )
        cur_stride *= 2

        self.tr2 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage2_num_channels[-1],
            stride_cur=[
                cur_stride * (2 ** i) for i in range(len(stage3_num_channels))
            ],
            out_channels=stage3_num_channels,
            align_corners=self.align_corners,
            name="tr2")
        self.st3 = Stage(
            num_channels=stage3_num_channels,
            num_modules=stage3_num_modules,
            num_blocks=stage3_num_blocks,
            num_filters=stage3_num_channels,
            has_se=self.has_se,
            name="st3",
            align_corners=align_corners,
            state=state1
           )
        cur_stride *= 2

        self.tr3 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage3_num_channels[-1],
            stride_cur=[
                cur_stride * (2 ** i) for i in range(len(stage4_num_channels))
            ],
            out_channels=stage4_num_channels,
            align_corners=self.align_corners,
            name="tr3")
        self.st4 = Stage(
            num_channels=stage4_num_channels,
            num_modules=stage4_num_modules,
            num_blocks=stage4_num_blocks,
            num_filters=stage4_num_channels,
            has_se=self.has_se,
            name="st4",
            align_corners=align_corners,
            state=state1
            )
        cur_stride *= 2

        self.tr4 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage4_num_channels[-1],
            stride_cur=[
                cur_stride * (2 ** i) for i in range(len(stage5_num_channels))
            ],
            out_channels=stage5_num_channels,
            align_corners=self.align_corners,
            name="tr4")
        self.st5 = Stage(
            num_channels=stage5_num_channels,
            num_modules=stage5_num_modules,
            num_blocks=stage5_num_blocks,
            num_filters=stage5_num_channels,
            has_se=self.has_se,
            name="st5",
            align_corners=align_corners,
            state=state1
            )

        self.tr5 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage5_num_channels[0],
            stride_cur=[
                cur_stride // (2 ** (len(stage6_num_channels) - i - 1))
                for i in range(len(stage6_num_channels))
            ],
            out_channels=stage6_num_channels,
            align_corners=self.align_corners,
            name="tr5")
        self.st6 = Stage(
            num_channels=stage6_num_channels,
            num_modules=stage6_num_modules,
            num_blocks=stage6_num_blocks,
            num_filters=stage6_num_channels,
            has_se=self.has_se,
            name="st6",
            align_corners=align_corners,
            state=state2
            )
        cur_stride = cur_stride // 2

        self.tr6 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage6_num_channels[0],
            stride_cur=[
                cur_stride // (2 ** (len(stage7_num_channels) - i - 1))
                for i in range(len(stage7_num_channels))
            ],
            out_channels=stage7_num_channels,
            align_corners=self.align_corners,
            name="tr6")
        self.st7 = Stage(
            num_channels=stage7_num_channels,
            num_modules=stage7_num_modules,
            num_blocks=stage7_num_blocks,
            num_filters=stage7_num_channels,
            has_se=self.has_se,
            name="st7",
            align_corners=align_corners,
            state=state2
            )
        cur_stride = cur_stride // 2

        self.tr7 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage7_num_channels[0],
            stride_cur=[
                cur_stride // (2 ** (len(stage8_num_channels) - i - 1))
                for i in range(len(stage8_num_channels))
            ],
            out_channels=stage8_num_channels,
            align_corners=self.align_corners,
            name="tr7")
        self.st8 = Stage(
            num_channels=stage8_num_channels,
            num_modules=stage8_num_modules,
            num_blocks=stage8_num_blocks,
            num_filters=stage8_num_channels,
            has_se=self.has_se,
            name="st8",
            align_corners=align_corners,
            state=state2
            )
        cur_stride = cur_stride // 2

        self.tr8 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage8_num_channels[0],
            stride_cur=[
                cur_stride // (2 ** (len(stage9_num_channels) - i - 1))
                for i in range(len(stage9_num_channels))
            ],
            out_channels=stage9_num_channels,
            align_corners=self.align_corners,
            name="tr8")
        self.st9 = Stage(
            num_channels=stage9_num_channels,
            num_modules=stage9_num_modules,
            num_blocks=stage9_num_blocks,
            num_filters=stage9_num_channels,
            has_se=self.has_se,
            name="st9",
            align_corners=align_corners,
            state=state2
            )

        self.last_layer = nn.Sequential(
            ConvBnReLU(
                in_channels=self.feat_channels[0],
                out_channels=self.feat_channels[0],
                kernel_size=1,
                padding='same',
                stride=1,
                bias=True),
            nn.Conv2d(
                in_channels=self.feat_channels[0],
                out_channels=19,
                kernel_size=1,
                stride=1,
                padding=0))

        self.deconvlist = nn.ModuleList()
        self.conv1list = nn.ModuleList()
        self.CSAlist = nn.ModuleList()
        self.brulist = nn.ModuleList()
        deconvchannels = [stage5_num_channels[-1], stage6_num_channels[-1], stage7_num_channels[-1],
                          stage8_num_channels[-1]]
        bruchannels = [stage6_num_channels[-1], stage7_num_channels[-1], stage8_num_channels[-1],
                       stage9_num_channels[-1]]
        catchannels = [stage7_num_channels[-1], stage8_num_channels[-1], stage9_num_channels[-1]]
        for i in range(len(deconvchannels)):
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=deconvchannels[i], out_channels=deconvchannels[i],
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(deconvchannels[i]),
                nn.ReLU(),
                nn.Conv2d(deconvchannels[i], deconvchannels[i] // 2, kernel_size=1),
                nn.BatchNorm2d(deconvchannels[i] // 2),
                nn.ReLU(), )
            self.deconvlist.append(self.deconv)
            self.csa = CSA_Block(num_channels=deconvchannels[i])
            self.CSAlist.append(self.csa)

        for i in range(len(bruchannels)):
            self.bru = BRU_Block(bruchannels[i])
            self.brulist.append(self.bru)

        for i in catchannels:
            self.conv = nn.Conv2d(in_channels=i * 2, out_channels=i, kernel_size=1)
            self.conv1list.append(self.conv)

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=stage9_num_channels[-1] + 64, out_channels=stage9_num_channels[-1], kernel_size=1),
            nn.BatchNorm2d(stage9_num_channels[-1]),
            nn.ReLU()
        )

        self.doconv_1 = DOConv2d(stage3_num_channels[0], stage3_num_channels[0], kernel_size=3, stride=1, padding=1)
        self.doconv_1_bn = nn.BatchNorm2d(stage3_num_channels[0])
        self.doconv_1_relu = nn.ReLU()
        self.lk_aspp = LK_ASPPBlock(stage4_num_channels[0], stage4_num_channels[0], [6, 12, 18])
        self.lk_aspp_1 = LK_ASPPBlock(stage5_num_channels[-1], stage5_num_channels[-1], [6, 12, 18])
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(stage3_num_channels[0] + stage4_num_channels[0], stage3_num_channels[0], kernel_size=1),
            nn.BatchNorm2d(stage3_num_channels[0]),
            nn.ReLU()
        )
        self.coordatt = CoordAtt(stage3_num_channels[0],stage3_num_channels[0])
        self.mlf = MLF_block(stage2_num_channels[0], stage2_num_channels[0], [6, 12, 18])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _concat(self, x1, x2):
        x1_1 = F.avg_pool3d(
            x1.unsqueeze(1), kernel_size=(2, 1, 1), stride=(2, 1, 1)).squeeze(1)
        x1_2 = F.max_pool3d(
            x1.unsqueeze(1), kernel_size=(2, 1, 1), stride=(2, 1, 1)).squeeze(1)

        x11 = x1_1 + x1_2

        x2_1 = F.avg_pool3d(
            x2.unsqueeze(1), kernel_size=(2, 1, 1), stride=(2, 1, 1)).squeeze(1)
        x2_2 = F.max_pool3d(
            x2.unsqueeze(1), kernel_size=(2, 1, 1), stride=(2, 1, 1)).squeeze(1)
        x22 = x2_1 + x2_2

        return torch.cat([x11, x22], dim=1)

    def forward(self, x):
        conv1 = self.conv_layer1_1(x)
        conv2 = self.conv_layer1_2(conv1)

        la1 = self.la1(conv2)

        tr1 = self.tr1(la1)
        st2 = self.st2(tr1)
        skip21 = st2[0]

        tr2 = self.tr2(st2[-1])
        st3 = self.st3(tr2)
        skip31 = st3[0]

        tr3 = self.tr3(st3[-1])
        st4 = self.st4(tr3)
        skip41 = st4[0]

        tr4 = self.tr4(st4[-1])
        st5 = self.st5(tr4)
        x5 = st5[-1]

        tr5 = self.tr5(st5[0], shape=skip41.shape[-2:])
        tr5[0] = self._concat(tr5[0], skip41)
        st6 = self.st6(tr5)
        x4 = st6[-1]

        tr6 = self.tr6(st6[0], shape=skip31.shape[-2:])
        tr6[0] = self._concat(tr6[0], skip31)
        st7 = self.st7(tr6)
        x3 = st7[-1]

        tr7 = self.tr7(st7[0], shape=skip21.shape[-2:])
        tr7[0] = self._concat(tr7[0], skip21)
        st8 = self.st8(tr7)
        x2 = st8[-1]

        tr8 = self.tr8(st8[0])
        st9 = self.st9(tr8)
        x1 = st9[-1]

        # x = [x1, x2, x3, x4, x5]
        # for i in range(len(x)):
        #     x[i] = F.avg_pool3d(
        #         x[i].unsqueeze(1), kernel_size=(2, 1, 1), stride=(2, 1,
        #                                                           1)).squeeze(1)
        #
        # # upsampling
        # x0_h, x0_w = (x[0]).shape[-2:]
        # for i in range(1, len(x)):
        #     x[i] = F.interpolate(
        #         x[i],
        #         size=[x0_h, x0_w],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        # x = torch.concat(x, axis=1)
        #
        # return x

        x5 = self.lk_aspp_1(x5)
        out = [x5, x4, x3, x2, x1]

        # out1 = self.CSAlist[0](out[0])
        out1 = self.deconvlist[0](out[0])
        out1 = out1 + out[1]
        out1 = self.brulist[0](out1)

        csa_out1 = self.CSAlist[2](skip41)
        out2 = torch.cat([csa_out1, out[2]], dim=1)
        out2 = self.conv1list[0](out2)
        out3 = self.deconvlist[1](out1)
        out3 = out3 + out2
        out3 = self.brulist[1](out3)

        # doconv_out = self.doconv_1_relu(self.doconv_1_bn(self.doconv_1(skip31)))
        # lk_aspp_out = self.lk_aspp(skip41)
        # lk_aspp_out = F.interpolate(lk_aspp_out, size=skip31.size()[2:], mode="bilinear", align_corners=False)
        # doconv_out = torch.cat([lk_aspp_out, doconv_out], dim=1)
        # doconv_out = self.conv1_1(doconv_out)
        # doconv_out = self.coordatt(doconv_out)
        # doconv_out = self.bra(doconv_out)

        # out4 = torch.cat([doconv_out, out[3]], dim=1)
        csa_out2 = self.CSAlist[3](skip31)
        out4 = torch.cat([csa_out2, out[3]], dim=1)
        out4 = self.conv1list[1](out4)

        out5 = self.deconvlist[2](out3)
        out5 = out5 + out4
        out5 = self.brulist[2](out5)

        bra_out_2 = self.mlf(skip21, skip31, skip41)
        out6 = torch.cat([bra_out_2, out[4]], dim=1)
        # out6 = torch.cat([skip21, out[4]], dim=1)
        out6 = self.conv1list[2](out6)
        # out7 = self.CSAlist[3](out5)
        out7 = self.deconvlist[3](out5)
        out7 = out7 + out6
        out7 = self.brulist[3](out7)

        # out7 = self.final_conv(torch.cat([out7, conv2], dim=1))

        return out7


class Layer1(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 num_blocks,
                 has_se=False,
                 name=None):
        super(Layer1, self).__init__()

        self.bottleneck_block_list = nn.Sequential()

        for i in range(num_blocks):
            self.bottleneck_block_list.add_module(
                "bb_{}_{}".format(name, i + 1),
                BottleneckBlock(
                    num_channels=num_channels if i == 0 else num_filters * 4,
                    num_filters=num_filters,
                    has_se=has_se,
                    stride=1,
                    downsample=True if i == 0 else False,
                    name=name + '_' + str(i + 1)))

    def forward(self, x):
        conv = x
        for block_func in self.bottleneck_block_list:
            conv = block_func(conv)
        return conv


class TransitionLayer(nn.Module):
    def __init__(self,
                 stride_pre,
                 in_channel,
                 stride_cur,
                 out_channels,
                 align_corners=False,
                 name=None):
        super(TransitionLayer, self).__init__()
        self.align_corners = align_corners
        num_out = len(out_channels)
        if num_out != len(stride_cur):
            raise ValueError(
                'The length of `out_channels` does not equal to the length of `stride_cur`'
                    .format(num_out, len(stride_cur)))
        self.conv_bn_func_list = nn.ModuleList()
        for i in range(num_out):
            residual = nn.Sequential()
            if stride_cur[i] == stride_pre:
                if in_channel != out_channels[i]:
                    residual.add_module(
                        "transition_{}_layer_{}".format(name, i + 1),
                        ConvBnReLU(
                            in_channels=in_channel,
                            out_channels=out_channels[i],
                            kernel_size=3,
                            padding='same',
                        ))
                else:
                    residual = None
            elif stride_cur[i] > stride_pre:
                residual.add_module(
                    "transition_{}_layer_{}".format(name, i + 1),
                    ConvBnReLU(
                        in_channels=in_channel,
                        out_channels=out_channels[i],
                        kernel_size=3,
                        stride=2,
                        padding='same',
                    ))
            else:
                residual.add_module(
                    "transition_{}_layer_{}".format(name, i + 1),
                    ConvBnReLU(
                        in_channels=in_channel,
                        out_channels=out_channels[i],
                        kernel_size=1,
                        stride=1,
                        padding='same',
                    ))
            self.conv_bn_func_list.append(residual)

    def forward(self, x, shape=None):
        outs = []
        for conv_bn_func in self.conv_bn_func_list:
            if conv_bn_func is None:
                outs.append(x)
            else:
                out = conv_bn_func(x)
                if shape is not None:
                    out = F.interpolate(
                        out,
                        shape,
                        mode='bilinear',
                        align_corners=self.align_corners)
                outs.append(out)
        return outs


class Branches(nn.Module):
    def __init__(self,
                 num_blocks,
                 in_channels,
                 out_channels,
                 has_se=False,
                 name=None,
                 state = 'encoder'):
        super(Branches, self).__init__()

        self.basic_block_list = nn.ModuleList()
        self.state = state

        for i in range(len(out_channels)):
            basic_block_func = nn.ModuleList()
            for j in range(num_blocks[i]):
                in_ch = in_channels[i] if j == 0 else out_channels[i]
                if self.state == 'encoder':
                    basic_block_func.add_module(
                        "bb_{}_branch_layer_{}_{}".format(name, i + 1, j + 1),
                        Bottle2neck(
                            num_channels=in_ch,
                            num_filters=out_channels[i],
                            has_se=has_se,
                            name=name + '_branch_layer_' + str(i + 1) + '_' +
                                 str(j + 1)))
                else:
                    basic_block_func.add_module(
                        "bb_{}_branch_layer_{}_{}".format(name, i + 1, j + 1),
                        BasicBlock(
                            num_channels=in_ch,
                            num_filters=out_channels[i],
                            has_se=has_se,
                            name=name + '_branch_layer_' + str(i + 1) + '_' +
                                 str(j + 1)))
            self.basic_block_list.append(basic_block_func)

    def forward(self, x):
        outs = []
        for idx, input in enumerate(x):
            conv = input
            for basic_block_func in self.basic_block_list[idx]:
                conv = basic_block_func(conv)
            outs.append(conv)
        return outs


class BottleneckBlock(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 has_se,
                 stride=1,
                 downsample=False,
                 name=None):
        super(BottleneckBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBnReLU(in_channels=num_channels, out_channels=num_filters, kernel_size=1, padding='same')

        self.conv2 = ConvBnReLU(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=stride,
                                padding='same')

        self.conv3 = ConvBn(in_channels=num_filters, out_channels=num_filters * 4, kernel_size=1, padding='same')

        if self.downsample:
            self.conv_down = ConvBn(in_channels=num_channels, out_channels=num_filters * 4, kernel_size=1,
                                    padding='same')

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters * 4,
                num_filters=num_filters * 4,
                reduction_ratio=16,
                name=name + '_fc')

    def forward(self, x):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        if self.downsample:
            residual = self.conv_down(x)

        if self.has_se:
            conv3 = self.se(conv3)

        y = conv3 + residual
        y = F.relu(y)
        return y


class BasicBlock(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride=1,
                 has_se=False,
                 downsample=False,
                 name=None):
        super(BasicBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBnReLU(in_channels=num_channels,out_channels=num_filters,kernel_size=3,stride=stride,padding='same',)
        self.conv2 = ConvBn(in_channels=num_filters,out_channels=num_filters,kernel_size=3,padding='same',)

        if self.downsample:
            self.conv_down = ConvBnReLU(
                in_channels=num_channels,
                out_channels=num_filters,
                kernel_size=1,
                padding='same',
            )

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters,
                num_filters=num_filters,
                reduction_ratio=16,
                name=name + '_fc')

    def forward(self, x):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        if self.downsample:
            residual = self.conv_down(x)

        if self.has_se:
            conv2 = self.se(conv2)

        y = conv2 + residual
        y = F.relu(y)
        return y


class Bottle2neck(nn.Module):

    def __init__(self, num_channels,
                 num_filters,
                 stride=1,
                 baseWidth=26,
                 has_se=False,
                 scale=4,
                 name=None):
        """ Constructor
        Args:
            num_channels: input channel dimensionality
            num_filters: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        self.has_se = has_se
        width = int(math.floor(num_filters * (baseWidth / 48.0)))
        self.conv1 = nn.Conv2d(num_channels, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.nums = scale - 1
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, num_filters, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        self.scale = scale
        self.width = width

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters,
                num_filters=num_filters,
                reduction_ratio=16,
                name=name + '_fc')

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual

        if self.has_se:
            out = self.se(out)
        out = self.relu(out)
        return out

class SELayer(nn.Module):
    def __init__(self, num_channels, num_filters, reduction_ratio, name=None):
        super(SELayer, self).__init__()

        self.pool2d_gap = nn.AdaptiveAvgPool2d(1)

        self._num_channels = num_channels

        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = nn.Linear(
            num_channels,
            med_ch,
            act="relu",
            param_attr=torch.ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv)))

        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = nn.Linear(
            med_ch,
            num_filters,
            act="sigmoid",
            param_attr=torch.ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv)))

    def forward(self, x):
        pool = self.pool2d_gap(x)
        pool = torch.reshape(pool, shape=[-1, self._num_channels])
        squeeze = self.squeeze(pool)
        excitation = self.excitation(squeeze)
        excitation = torch.reshape(
            excitation, shape=[-1, self._num_channels, 1, 1])
        out = x * excitation
        return out


class Stage(nn.Module):
    def __init__(self,
                 num_channels,
                 num_modules,
                 num_blocks,
                 num_filters,
                 has_se=True,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False,state='encoder'):
        super(Stage, self).__init__()

        self._num_modules = num_modules

        self.stage_func_list = nn.Sequential()
        for i in range(num_modules):
            if i == num_modules - 1 and not multi_scale_output:
                self.stage_func_list.add_module(
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                        has_se=has_se,
                        multi_scale_output=False,
                        name=name + '_' + str(i + 1),
                        align_corners=align_corners,state=state))
            else:
                self.stage_func_list.add_module(
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                        has_se=has_se,
                        name=name + '_' + str(i + 1),
                        align_corners=align_corners,state=state))

    def forward(self, x):
        out = x
        for idx in range(self._num_modules):
            out = self.stage_func_list[idx](out)
        return out


class HighResolutionModule(nn.Module):
    def __init__(self,
                 num_channels,
                 num_blocks,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False,state='encoder'):
        super(HighResolutionModule, self).__init__()

        self.branches_func = Branches(
            num_blocks=num_blocks,
            in_channels=num_channels,
            out_channels=num_filters,
            has_se=has_se,
            name=name,
            state=state)

        self.fuse_func = FuseLayers(
            in_channels=num_filters,
            out_channels=num_filters,
            multi_scale_output=multi_scale_output,
            name=name,
            align_corners=align_corners)

    def forward(self, x):
        out = self.branches_func(x)
        out = self.fuse_func(out)
        return out


class FuseLayers(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False):
        super(FuseLayers, self).__init__()

        self._actual_ch = len(in_channels) if multi_scale_output else 1
        self._in_channels = in_channels
        self.align_corners = align_corners

        self.residual_func_list = nn.Sequential()
        for i in range(self._actual_ch):
            for j in range(len(in_channels)):
                if j > i:
                    self.residual_func_list.add_module(
                        "residual_{}_layer_{}_{}".format(name, i + 1, j + 1),
                        ConvBn(
                            in_channels=in_channels[j],
                            out_channels=out_channels[i],
                            kernel_size=1,
                            padding='same',
                        ))
                elif j < i:
                    pre_num_filters = in_channels[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            self.residual_func_list.add_module(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvBn(
                                    in_channels=pre_num_filters,
                                    out_channels=out_channels[i],
                                    kernel_size=3,
                                    stride=2,
                                    padding='same',
                                ))
                            pre_num_filters = out_channels[i]
                        else:
                            self.residual_func_list.add_module(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvBnReLU(
                                    in_channels=pre_num_filters,
                                    out_channels=out_channels[j],
                                    kernel_size=3,
                                    stride=2,
                                    padding='same',
                                ))
                            pre_num_filters = out_channels[j]

        if len(self.residual_func_list) == 0:
            self.residual_func_list.add_module("identity",
                                               nn.Identity())  # for flops calculation

    def forward(self, x):
        outs = []
        residual_func_idx = 0
        for i in range(self._actual_ch):
            residual = x[i]
            residual_shape = residual.shape[-2:]

            for j in range(len(self._in_channels)):
                if j > i:
                    y = self.residual_func_list[residual_func_idx](x[j])
                    residual_func_idx += 1

                    y = F.interpolate(
                        y,
                        residual_shape,
                        mode='bilinear',
                        align_corners=self.align_corners)
                    residual = residual + y
                elif j < i:
                    y = x[j]
                    for k in range(i - j):
                        y = self.residual_func_list[residual_func_idx](y)
                        residual_func_idx += 1

                    residual = residual + y

            residual = F.relu(residual)
            outs.append(residual)

        return outs


class LK_ASPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(LK_ASPPBlock, self).__init__()

        # Atrous convolutions with different rates
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 3), padding=(2 * rates[0], rates[0]),
                                     dilation=rates[0])
        self.conv3x3_1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 5), padding=(rates[0], 2 * rates[0]),
                                     dilation=rates[0])
        self.conv3x3_2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 3), padding=(2 * rates[1], rates[1]),
                                     dilation=rates[1])
        self.conv3x3_2_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 5), padding=(rates[1], 2 * rates[1]),
                                     dilation=rates[1])
        self.conv3x3_3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 3), padding=(2 * rates[2], rates[2]),
                                     dilation=rates[2])
        self.conv3x3_3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 5), padding=(rates[2], 2 * rates[2]),
                                     dilation=rates[2])

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(5 * out_channels, out_channels, kernel_size=1)

        # Batch normalization and ReLU activation
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Atrous convolutions
        conv1x1_1 = self.relu(self.batch_norm(self.conv1x1_1(x)))
        conv3x3_1_1 = self.relu(self.batch_norm(self.conv3x3_1_1(x)))
        conv3x3_1_2 = self.relu(self.batch_norm(self.conv3x3_1_2(x)))
        conv3x3_2_1 = self.relu(self.batch_norm(self.conv3x3_2_1(x)))
        conv3x3_2_2 = self.relu(self.batch_norm(self.conv3x3_1_2(x)))
        conv3x3_3_1 = self.relu(self.batch_norm(self.conv3x3_3_1(x)))
        conv3x3_3_2 = self.relu(self.batch_norm(self.conv3x3_1_2(x)))
        conv3x3_1 = conv3x3_1_1 + conv3x3_1_2
        conv3x3_2 = conv3x3_2_1 + conv3x3_2_2
        conv3x3_3 = conv3x3_3_1 + conv3x3_3_2

        # Global average pooling
        global_avg_pool = self.global_avg_pool(x)
        conv1x1_2 = self.conv1x1_2(global_avg_pool)
        upsampled = F.interpolate(conv1x1_2, size=x.size()[2:], mode='bilinear', align_corners=False)

        # Concatenate all branches
        out = torch.cat([conv1x1_1, conv3x3_1, conv3x3_2, conv3x3_3, upsampled], dim=1)
        out = self.relu(self.batch_norm(self.conv1x1_3(out)))

        return out


class MLF_block(nn.Module):
    def __init__(self, inchannel, out_channel, rates):
        super(MLF_block, self).__init__()
        self.do_conv_1 = DOConv2d(inchannel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.do_conv_2 = DOConv2d(2 * inchannel, 2 * out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * out_channel)
        self.conv1_1 = nn.Conv2d(2 * out_channel, out_channel, kernel_size=1)

        self.lk_aspp_1 = LK_ASPPBlock(2 * inchannel, 2 * out_channel, rates=rates)
        self.lk_aspp_2 = LK_ASPPBlock(4 * inchannel, 4 * out_channel, rates=rates)
        self.relu = nn.ReLU()

        self.conv1_2 = nn.Conv2d(4 * out_channel, 2 * out_channel, kernel_size=1)
        self.conv1_3 = nn.Conv2d(3 * out_channel, out_channel, kernel_size=1)
        self.bra = nchwBRA(dim=out_channel, n_win=16, topk=4)
        self.coordatt = CoordAtt(out_channel,out_channel)

    def forward(self, x1, x2, x3):
        x1_1 = self.relu(self.bn1(self.do_conv_1(x1)))
        x2_1 = self.relu(self.bn2(self.do_conv_2(x2)))
        x2_2 = self.lk_aspp_1(x2)
        x2_2_up = F.interpolate(x2_2, size=x1.size()[2:], mode='bilinear', align_corners=False)
        x2_2_up = self.conv1_1(x2_2_up)

        out_1 = x1_1 * x2_2_up

        x3_1 = self.lk_aspp_2(x3)
        x3_1_up = F.interpolate(x3_1, size=x2.size()[2:], mode='bilinear', align_corners=False)
        x3_1_up = self.conv1_2(x3_1_up)

        out_2 = x2_1 * x3_1_up

        out_2 = F.interpolate(out_2, size=x1.size()[2:], mode='bilinear', align_corners=False)
        out_2 = torch.cat([out_2, out_1], dim=1)
        out = self.conv1_3(out_2)
        # out = self.relu(self.bn1(self.do_conv_1(out)))
        out = self.coordatt(out)
        # out = self.bra(out)

        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=16):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

# 定义通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 使用全连接层实现多层感知机
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 通道数不变，H*W变为1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  #

        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction, num_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # print(avg_out.shape)
        # 两层神经网络共享
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


# 定义空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 使用卷积层实现空间注意力
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # b,c,h,w=x.size()
        maxpool_out, _ = torch.max(x, dim=1, keepdim=True)
        avgpool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([maxpool_out, avgpool_out], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return out * x


class CSA_Block(nn.Module):
    def __init__(self, num_channels):
        super(CSA_Block, self).__init__()
        self.num_channels = num_channels
        self.channel_attention = ChannelAttention(self.num_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        channelattention_out = self.channel_attention(x)
        spatialattention_out = self.spatial_attention(x)

        out = channelattention_out + spatialattention_out + x

        return out


class BRU_Block(nn.Module):
    def __init__(self, num_channels):
        super(BRU_Block, self).__init__()

        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        n, c, h, w = x.size()
        x_res = x
        x_res = torch.mul(
            nn.Parameter(torch.randn((n, c, h, w), dtype=torch.float32, requires_grad=True, device=x.device)),
            self.conv1(x_res))
        # x_res = self.bn(x_res)
        x_res = self.relu(x_res)
        x_res = torch.mul(
            nn.Parameter(torch.randn((n, c, h, w), dtype=torch.float32, requires_grad=True, device=x.device)),
            self.conv2(x_res))
        # x_res = self.bn(x_res)
        # x = self.relu(x + x_res)
        x = x + x_res

        return x


def UHRNet_W18_Small(**kwargs):
    model = UHRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[2],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[2, 2],
        stage2_num_channels=[18, 36],
        stage3_num_modules=2,
        stage3_num_blocks=[2, 2],
        stage3_num_channels=[36, 72],
        stage4_num_modules=2,
        stage4_num_blocks=[2, 2],
        stage4_num_channels=[72, 144],
        stage5_num_modules=2,
        stage5_num_blocks=[2, 2],
        stage5_num_channels=[144, 288],
        stage6_num_modules=1,
        stage6_num_blocks=[2, 2],
        stage6_num_channels=[72, 144],
        stage7_num_modules=1,
        stage7_num_blocks=[2, 2],
        stage7_num_channels=[36, 72],
        stage8_num_modules=1,
        stage8_num_blocks=[2, 2],
        stage8_num_channels=[18, 36],
        stage9_num_modules=1,
        stage9_num_blocks=[2],
        stage9_num_channels=[18],
        **kwargs)
    return model


def UHRNet_W18(**kwargs):
    model = UHRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4, ],
        stage1_num_channels=[64, ],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[18, 36],
        stage3_num_modules=5,
        stage3_num_blocks=[4, 4],
        stage3_num_channels=[36, 72],
        stage4_num_modules=2,
        stage4_num_blocks=[4, 4],
        stage4_num_channels=[72, 144],
        stage5_num_modules=2,
        stage5_num_blocks=[4, 4],
        stage5_num_channels=[144, 288],
        stage6_num_modules=1,
        stage6_num_blocks=[4, 4],
        stage6_num_channels=[72, 144],
        stage7_num_modules=1,
        stage7_num_blocks=[4, 4],
        stage7_num_channels=[36, 72],
        stage8_num_modules=1,
        stage8_num_blocks=[4, 4],
        stage8_num_channels=[18, 36],
        stage9_num_modules=1,
        stage9_num_blocks=[4, ],
        stage9_num_channels=[18, ],
        **kwargs)
    return model


def UHRNet_W48(**kwargs):
    model = UHRNet(
        stage1_num_modules=1,
        stage1_num_blocks=(4,),
        stage1_num_channels=[64, ],
        stage2_num_modules=1,
        stage2_num_blocks=(4, 4),
        stage2_num_channels=[48, 96],
        stage3_num_modules=5,
        stage3_num_blocks=(4, 4),
        stage3_num_channels=[96, 192],
        stage4_num_modules=2,
        stage4_num_blocks=(4, 4),
        stage4_num_channels=[192, 384],
        stage5_num_modules=2,
        stage5_num_blocks=(4, 4),
        stage5_num_channels=[384, 768],
        stage6_num_modules=1,
        stage6_num_blocks=(4, 4),
        stage6_num_channels=[192, 384],
        stage7_num_modules=1,
        stage7_num_blocks=(4, 4),
        stage7_num_channels=[96, 192],
        stage8_num_modules=1,
        stage8_num_blocks=(4, 4),
        stage8_num_channels=[48, 96],
        stage9_num_modules=1,
        stage9_num_blocks=(4,),
        stage9_num_channels=[48, ],
        **kwargs)
    return model


if __name__ == '__main__':
    images = torch.rand(1, 3, 512, 512)
    model = UHRNet_W48()
    print(model(images).size())
    # print(model)
    # from tensorboardX import SummaryWriter
    # writer = SummaryWriter('model_logs')
    # writer.add_graph(model,images)
    # writer.close()