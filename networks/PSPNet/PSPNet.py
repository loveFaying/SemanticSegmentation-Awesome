'''
[description]
PSPNet
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from .resnet101 import get_resnet101
from collections import OrderedDict

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class PyramidPooling(nn.Module):
    def __init__(self, name, out_planes, fc_dim=4096, pool_scales=[1, 2, 3, 6],
                 norm_layer=nn.BatchNorm2d):
        super(PyramidPooling, self).__init__()
        # self.psp_layer = PyramidPooling('psp', class_num, 2048, norm_layer=nn.BatchNorm2d)
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(OrderedDict([
                ('{}/pool_1'.format(name), nn.AdaptiveAvgPool2d(scale)),
                ('{}/cbr'.format(name),
                 ConvBnRelu(fc_dim, 512, 1, 1, 0, has_bn=True,
                            has_relu=True, has_bias=False,
                            norm_layer=norm_layer))
            ])))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv6 = nn.Sequential(
            ConvBnRelu(fc_dim + len(pool_scales) * 512, 512, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer),
            nn.Dropout2d(0.1, inplace=False),
            nn.Conv2d(512, out_planes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()
        ppm_out = [x]
        for pooling in self.ppm:
            ppm_out.append(
                F.interpolate(pooling(x), size=(input_size[2], input_size[3]),
                              mode='bilinear', align_corners=True))
        ppm_out = torch.cat(ppm_out, 1)

        ppm_out = self.conv6(ppm_out)
        return ppm_out

class PSPNet(nn.Module):
    def __init__(self, class_num, bn_momentum=0.01):
        super(PSPNet, self).__init__()
        self.Resnet101 = get_resnet101(dilation=[1, 1, 1, 2], bn_momentum=bn_momentum, is_fpn=False)
        self.psp_layer = PyramidPooling('psp', class_num, 2048, norm_layer=nn.BatchNorm2d)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        b, c, h, w = input.shape
        x = self.Resnet101(input)
        psp_fm = self.psp_layer(x)
        pred = F.interpolate(psp_fm, size=input.size()[2:4], mode='bilinear', align_corners=True)
        pred = self.sigmoid(pred)

        return pred

def get_pspnet_model(n_class=1):
    model = PSPNet(class_num=n_class)
    return model

def test():
    device = torch.device('cuda')  # cuda:0
    inputs = torch.rand(2, 3, 256, 256).to(device)   # PSPNet要有batch
    print(inputs.shape)

    net = get_pspnet_model().to(device)
    res = net(inputs)  # res是一个tuple类型
    print('res shape:', res.shape)

    # 计算模型参数量
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print("The number of parameters : %.3f M" % (num_params/1e6))


if __name__ == '__main__':
    test()



