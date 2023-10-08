# 方案一：只使用注意力后的FPN
# 方案二：把使用注意力后的FPN和之前的FNP cat后通过1*1卷积进行融合
# 方案三：把使用注意力后的FPN和之前的FNP按元素相乘 #不可行，注意力机制本身就是按元素相乘，没必要再乘一次
# 方案四：使用注意力前要不要对梯度图进行3*3卷积，以及各层卷积核是否统一使用3*3

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
import cv2

from mmdet.models import NECKS

@NECKS.register_module()
class Grad_FPN_2(BaseModule): #方案而
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs, #在fasterRCNN上默认5，FPN本只有4层，在最顶层下采样形成5层
                 start_level=0,
                 end_level=-1,
                 low_threshold=40,
                 high_threshold=80,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(Grad_FPN_2, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins  #默认=4
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv) #默认4个1*1卷积
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.grad_conv = ConvModule(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.fuse_conv = ConvModule(in_channels=self.out_channels*2, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0)  #把注意力机制前后的FPN 1*1融合

    def get_gaussian_kernel(slef, k=3, mu=0, sigma=1, normalize=True):
        # compute 1 dimension gaussian
        gaussian_1D = np.linspace(-1, 1, k)
        # compute a grid distance from center
        x, y = np.meshgrid(gaussian_1D, gaussian_1D)
        distance = (x ** 2 + y ** 2) ** 0.5

        # compute the 2 dimension gaussian
        gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
        gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

        # normalize part (mathematically)
        if normalize:
            gaussian_2D = gaussian_2D / np.sum(gaussian_2D)

        gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=3,
                                         padding=1,
                                         bias=False).to(device = 'cuda:0')
        gaussian_filter.weight.data[:, :] = nn.Parameter(torch.from_numpy(gaussian_2D),
                                                              requires_grad=False)  # 填充固定权重
        return gaussian_filter

    def get_sobel_kernel(self, k=3):
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator  # [[-0.5  0.   0.5],[-1.   0.   1. ],[-0.5  0.   0.5]]，为什么权重不是常见的1,2,1

        sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=3,
                                        padding=1,
                                        bias=False).to(device = 'cuda:0')
        sobel_filter_x.weight.data[:, :] = nn.Parameter(torch.from_numpy(sobel_2D), requires_grad=False)

        sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=3,
                                        padding=1,
                                        bias=False).to(device = 'cuda:0')
        sobel_filter_y.weight.data[:, :] = nn.Parameter(torch.from_numpy(sobel_2D.T), requires_grad=False)
        return sobel_filter_x, sobel_filter_y

    def get_thin_kernels(self, start=0, end=360, step=45):
        k_thin = 3  # actual size of the directional kernel
        # increase for a while to avoid interpolation when rotating
        k_increased = k_thin + 2

        # get 0° angle directional kernel
        thin_kernel_0 = np.zeros((k_increased, k_increased))
        thin_kernel_0[k_increased // 2, k_increased // 2] = 1
        thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

        # rotate the 0° angle directional kernel to get the other ones
        thin_kernels = []
        for angle in range(start, end, step):
            (h, w) = thin_kernel_0.shape
            # get the center to not rotate around the (0, 0) coord point
            center = (w // 2, h // 2)
            # apply rotation
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

            # get the k=3 kerne
            kernel_angle = kernel_angle_increased[1:-1, 1:-1]
            is_diag = (abs(kernel_angle) == 1)  # because of the interpolation
            kernel_angle = kernel_angle * is_diag  # because of the interpolation
            thin_kernels.append(kernel_angle)  # 8个3*3矩阵，中心元素为1，-1一次依次在八邻域转一圈，其他元素为0

        directional_kernels = np.stack(thin_kernels)  # 8*3*3的numpy

        directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False).to(device = 'cuda:0')  # 8通道的3*3卷积核
        directional_filter.weight.data[:, 0] = nn.Parameter(torch.from_numpy(directional_kernels),
                                                                 requires_grad=False)
        return directional_filter

    def get_hysteresis_kernels(self):
        return nn.MaxPool2d(kernel_size=3, stride=1, padding=1).to(device = 'cuda:0')
    def get_dilate_kernel(self):
        return nn.MaxPool2d(kernel_size=3, stride=1, padding=1).to(device = 'cuda:0')

    def get_canny_grad(self, img, low_threshold, high_threshold, is_dilate=True):
        std = torch.Tensor([58.395, 57.12, 57.375]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        mean = torch.Tensor([123.675, 116.28, 103.53]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        img = img * std + mean  # 转回normalize前的值
        sum_core = torch.Tensor([[[[0.299]], [[0.587]], [[0.114]]]]).cuda()  # RGB
        img = F.conv2d(img, weight=sum_core, stride=1)
        # gaussian
        gaussian_filter = self.get_gaussian_kernel()
        sobel_filter_x, sobel_filter_y = self.get_sobel_kernel()
        blurred = gaussian_filter(img)
        grad_x = sobel_filter_x(blurred)
        grad_y = sobel_filter_y(blurred)

        # thick edges
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_orientation = torch.atan2(grad_y, grad_x)  # 值域范围(-pi, pi)
        grad_orientation = grad_orientation * (180 / torch.pi) + 180  # 9点钟方向为0°，逆时针转360°
        # grad_orientation = torch.round(grad_orientation / 45) * 45
        # get indices of positive and negative directions
        positive_idx = torch.round(
            grad_orientation / 45) % 8  # (B,1,H,W),归一化为8个整数方向，[337.5-22.5]归为0，[22.5-67.5]归为1，以此类推
        directional_filter = self.get_thin_kernels()
        directional = directional_filter(grad_magnitude)  # (B,8,H,W)

        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):  # 以pos_i=0为例
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = ((positive_idx == pos_i) + (positive_idx == neg_i)) * 1  # (B,1,H,W)对边缘图生成梯度方向为0和4的mask
            pos_directional = directional[:, pos_i]  # directional中取出第0张图 (B,H,W)
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])  # (2,B,H,W)，第0维度是梯度和0和4两方向邻域的差值

            # get the local maximum pixels for the angle
            # selected_direction.min(dim=0)返回一个列表[0]中包含两者中的小的，[1]包含了小值的索引
            is_max = selected_direction.min(dim=0)[0] > 0.0  # (B,H,W),梯度正反两方向邻域的差值是否都大于0
            is_max = torch.unsqueeze(is_max, dim=1)  # (B,1,H,W),每个元素表示在0和4方向是不是最大值

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (
                is_oriented_i) > 0  # is_max == 0取出非最大值的部分（直接取反则包括了非梯度和非0.4方向的位置），因此和is_oriented_i再与一下
            thin_edges[to_remove] = 0.0  # (B,1,H,W)，虽然to_remove是(B,1,H,W)，但a[b]=0这种写法会自动将a的b为True的值对应坐标的值置0

        # thresholds
        hysteresis = self.get_hysteresis_kernels()
        if low_threshold is not None:
            low = thin_edges > low_threshold  # nms后的梯度图大于low_threshold的mask

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5  # 为0.5的部分表示仅大于low_threshold，为1的部分大于high_threshold
                # get weaks and check if they are high or not
                weak = (thin_edges == 0.5) * 1  # 高低阈值之间的mask
                weak_is_high = (hysteresis(
                        thin_edges) == 1.0) * weak  # thin_edges中原先的0也可能变成1，因此再*weak，则只剩下0.5变1的pixel
                thin_edges = high * 1. + weak_is_high * 1.
            else:
                thin_edges = low * 1.
                print('请输出canny上阈值')
        else:
            print('请输出canny下阈值')
        #name = str(np.random.randint(1,100))
        #cv2.imwrite('images/train_gray'+name+'.jpg', thin_edges.clone().cpu().numpy()[0][0] * 255)
        if is_dilate:
            dilate = self.get_dilate_kernel()
            thin_edges = dilate(thin_edges)
        #cv2.imwrite('images/train_gray'+name+'dilate.jpg', thin_edges.clone().cpu().numpy()[0][0] * 255)
        return thin_edges, grad_magnitude*thin_edges,grad_x*thin_edges,grad_y*thin_edges  # thin_edges即为mask

    @auto_fp16()
    def forward(self, inputs, imgs):
        '''
        imputs:list[tensor(B,C,W,H)]
        '''
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # 进行边缘注意力机制
        grad_ls = []
        for img in imgs:
            l, h = self.low_threshold, self.high_threshold
            grad_mask, grad_l, grad_x, grad_y = self.get_canny_grad(img.unsqueeze(0), l, h)
            while torch.sum(grad_mask / (1024 ** 2)) < 0.05:
                l = l * 0.7
                h = h * 0.7
                grad_mask, grad_l, grad_x, grad_y = self.get_canny_grad(img.unsqueeze(0), l, h)
            grad_ls.append(grad_l)

            # name = str(np.random.randint(1, 100))
            # cv2.imwrite('images/train_gray' + name + 'dilate.jpg', grad_mask.clone().cpu().numpy()[0][0] * 255)
        grad_ls = torch.cat(grad_ls, dim=0)  #(B,1,W,H)
        grad_ls_4 = F.sigmoid(self.grad_conv(F.max_pool2d(grad_ls, kernel_size=5, stride=4, padding=2)))  # 降采样4倍
        grad_ls_8 = F.sigmoid(self.grad_conv(F.max_pool2d(grad_ls_4, kernel_size=3, stride=2, padding=1)))  # 降采样8倍
        grad_ls_16 = F.sigmoid(self.grad_conv(F.max_pool2d(grad_ls_8, kernel_size=3, stride=2, padding=1)))  # 降采样16倍
        grad_ls_32 = F.sigmoid(self.grad_conv(F.max_pool2d(grad_ls_16, kernel_size=3, stride=2, padding=1)))  # 降采样32倍
        grad_ls_64 = F.sigmoid(self.grad_conv(F.max_pool2d(grad_ls_32, kernel_size=3, stride=2, padding=1)))  # 降采样32倍
        grad_ls_downsapmle = [grad_ls_4, grad_ls_8, grad_ls_16, grad_ls_32, grad_ls_64]  #list[(B,1,W,H)]

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]  #backbone4个stage的输出经1*1卷积后的结果

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1): #3->2->1->0
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:] #待融合下层的w,h
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg) #待融合的上层上采样后加上下层，采样方式：最近邻
        #由于是在原laterals上融合度，FPN最上一层即backbone最后一层输出，FPN其余三层是自上而下融合的结果
        #融合后laterals内各层，分辨率依次减小
        # build outputs
        # part 1: from original levels
        #原FPN输出
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels) #对融合后的各层分别进行3*3卷积
        ]
        #先进行注意力机制，然后在c上拼接，然后1*1回复原来维度
        #方案2
        #outs = [self.fuse_conv(torch.cat([out*grad_ls_downsapmle[i],out],dim=1)) for i, out in enumerate(outs)]
        #方案1
        #outs = [out * grad_ls_downsapmle[i] for i, out in enumerate(outs)]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    '''
                    #方案1
                    #outs.append(F.max_pool2d(outs[-1], 1, stride=2)*grad_ls_downsapmle[-1]) #对最上层进行1*1进行下采样stride=2，形成第五层,换成3*3是不是更好
                    #方案2
                    #out5 = F.max_pool2d(outs[-1], 1, stride=2) #原始最顶层
                    #outs.append(self.fuse_conv(torch.cat([out5 * grad_ls_downsapmle[-1], out5],dim=1)))
                    '''
                    #outs中第四层本来就已经进了自注意机制，因此只需要下采样生成第5层即可，不需要再次自注意
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)
