import torch.nn as nn
from mmdet.models.backbones import ResNet
from mmrotate.models.builder import ROTATED_BACKBONES
import torch
from torch import nn
import numpy as np
import cv2
import torch.nn.functional as F



@ROTATED_BACKBONES.register_module()
class Diff_ResNet(ResNet):
    def get_sobel_kernel(self, k, channels):
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator  # [[-0.5  0.   0.5],[-1.   0.   1. ],[-0.5  0.   0.5]]，为什么权重不是常见的1,2,1
        #分组卷积，每个channel分一组
        sobel_filter_x = nn.Conv2d(in_channels=channels,
                                   out_channels=channels,
                                   kernel_size=3,
                                   groups=channels,
                                   padding=1,
                                   bias=False).to(device='cuda:0')
        sobel_filter_x.weight.data = nn.Parameter(torch.from_numpy(sobel_2D).float().cuda().unsqueeze(0).unsqueeze(0).repeat(channels,1,1,1), requires_grad=False)
        #卷积核的数据维度为(channels，1,3,3)
        sobel_filter_y = nn.Conv2d(in_channels=channels,
                                   out_channels=channels,
                                   kernel_size=3,
                                   groups=channels,
                                   padding=1,
                                   bias=False).to(device='cuda:0')

        sobel_filter_y.weight.data = nn.Parameter(torch.from_numpy(sobel_2D.T).float().cuda().unsqueeze(0).unsqueeze(0).repeat(channels,1,1,1), requires_grad=False)
        return sobel_filter_x, sobel_filter_y

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []


        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        #求差分
        outs_xy = []
        channels = [256, 512, 1024, 2048]  # 原resnet每个stage的输出维度
        for i, out in enumerate(outs):
            sobel_filter_x, sobel_filter_y = self.get_sobel_kernel(3, channels[i])
            x_x = sobel_filter_x(out)
            x_y = sobel_filter_y(out)
            x_x_y = torch.cat((x_x, x_y), dim=1)
            outs_xy.append(x_x_y)

        #return tuple(outs)
        return tuple(outs_xy)