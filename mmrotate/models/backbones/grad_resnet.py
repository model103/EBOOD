from mmdet.models.backbones import ResNet
from mmrotate.models.builder import ROTATED_BACKBONES
import torch
from torch import nn
import numpy as np
import cv2
import torch.nn.functional as F



@ROTATED_BACKBONES.register_module()
class Grad_ResNet(ResNet):
    def __init__(self,
                 depth,
                 low_threshold=None,
                 high_threshold=None,
                 in_channels=4,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None,):
        super().__init__(
                 depth,
                 in_channels,
                 stem_channels,
                 base_channels,
                 num_stages,
                 strides,
                 dilations,
                 out_indices,
                 style,
                 deep_stem,
                 avg_down,
                 frozen_stages,
                 conv_cfg,
                 norm_cfg,
                 norm_eval,
                 dcn,
                 stage_with_dcn,
                 plugins,
                 with_cp,
                 zero_init_residual,
                 pretrained,
                 init_cfg)

        self.low_threshold = low_threshold #nn.Parameter(torch.Tensor([low_threshold]).cuda(), requires_grad = True) #确实是叶子结点，但grad一直为None
        self.high_threshold = high_threshold #nn.Parameter(torch.Tensor([high_threshold]).cuda(), requires_grad = True)
        #self.low_threshold = low_threshold
        #self.high_threshold = high_threshold

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

    def forward(self, x):  # should return a tuple
        imgs = x.clone()
        grad_masks = []
        for img in imgs:
            l, h = self.low_threshold, self.high_threshold
            grad_mask, grad_l, grad_x, grad_y = self.get_canny_grad(img.unsqueeze(0), l, h)
            while torch.sum(grad_mask/(1024**2))<0.05:
                l = l*0.7
                h = h*0.7
                grad_mask, grad_l, grad_x, grad_y = self.get_canny_grad(img.unsqueeze(0), l, h)
            grad_masks.append(grad_mask)
            #name = str(np.random.randint(1, 100))
            #cv2.imwrite('images/train_gray' + name + 'dilate.jpg', grad_mask.clone().cpu().numpy()[0][0] * 255)
        grad_masks = torch.cat(grad_masks, dim=0)

        #print(grad_l.max(),grad_x.max(),grad_y.max())  #数字在450左右
        #x = torch.cat((x, grad_mask*2.5), dim=1) #×2.5是为了和RGB三个维度的数值在一个尺度上。另一种方法：也进行normalization
        x = torch.cat((x, grad_masks*2.5), dim=1)
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
        return tuple(outs)









