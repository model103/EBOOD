import torch
from torch import nn
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision.io import image
import time


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
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
    return gaussian_2D


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D   #[[-0.5  0.   0.5],[-1.   0.   1. ],[-0.5  0.   0.5]]，为什么权重不是常见的1,2,1


def get_thin_kernels(start=0, end=360, step=45):
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
        thin_kernels.append(kernel_angle)
    return thin_kernels  #8个3*3矩阵，中心元素为1，-1一次依次在八邻域转一圈，其他元素为0

def get_dilate_kernel():
    return nn.MaxPool2d(kernel_size=3, stride=1, padding=1).to(device = 'cuda:0')


class CannyFilter(nn.Module):
    def __init__(self,
                k_gaussian=3,
                mu=0,
                sigma=1,
                k_sobel=3,
                device = 'cuda:0'):
        super(CannyFilter, self).__init__()
        # device
        self.device = device
        # gaussian
        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_gaussian,
                                        padding=k_gaussian // 2,
                                        bias=False)
        self.gaussian_filter.weight.data[:,:] = nn.Parameter(torch.from_numpy(gaussian_2D), requires_grad=False)  #填充固定权重
        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=k_sobel,
                                       padding=k_sobel // 2,
                                       bias=False)
        self.sobel_filter_x.weight.data[:,:] = nn.Parameter(torch.from_numpy(sobel_2D), requires_grad=False)

        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=k_sobel,
                                       padding=k_sobel // 2,
                                       bias=False)
        self.sobel_filter_y.weight.data[:,:] = nn.Parameter(torch.from_numpy(sobel_2D.T), requires_grad=False)

        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels) #8*3*3的numpy

        self.directional_filter = nn.Conv2d(in_channels=1,
                                           out_channels=8,
                                           kernel_size=thin_kernels[0].shape,
                                           padding=thin_kernels[0].shape[-1] // 2,
                                           padding_mode='replicate',
                                           bias=False)  #8通道的3*3卷积核
        self.directional_filter.weight.data[:, 0] = nn.Parameter(torch.from_numpy(directional_kernels), requires_grad=False)

        # hysteresis，处理高低阈值之间的梯度用
        self.hysteresis = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=True):
        #to gray img
        sum_core = torch.Tensor([[[[0.299]], [[0.587]], [[0.114]]]]).cuda()  #RGB
        img = F.conv2d(img, weight=sum_core, stride=1)  # (B,1,H,W) 灰度图


        # gaussian
        blurred = self.gaussian_filter(img)
        grad_x = self.sobel_filter_x(blurred)
        grad_y = self.sobel_filter_y(blurred)

        # thick edges
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_orientation = torch.atan2(grad_y, grad_x) #值域范围(-pi, pi)
        grad_orientation = grad_orientation * (180 / torch.pi) + 180  # 9点钟方向为0°，逆时针转360°
        #grad_orientation = torch.round(grad_orientation / 45) * 45
        # get indices of positive and negative directions
        positive_idx = torch.round(grad_orientation / 45) % 8  #(B,1,H,W),归一化为8个整数方向，[337.5-22.5]归为0，[22.5-67.5]归为1，以此类推
        directional = self.directional_filter(grad_magnitude) #(B,8,H,W)

        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4): #以pos_i=0为例
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = ((positive_idx == pos_i) + (positive_idx == neg_i)) * 1 #(B,1,H,W)对边缘图生成梯度方向为0和4的mask
            pos_directional = directional[:, pos_i] #directional中取出第0张图 (B,H,W)
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])  #(2,B,H,W)，第0维度是梯度和0和4两方向邻域的差值

            # get the local maximum pixels for the angle
            # selected_direction.min(dim=0)返回一个列表[0]中包含两者中的小的，[1]包含了小值的索引
            is_max = selected_direction.min(dim=0)[0] > 0.0  #(B,H,W),梯度正反两方向邻域的差值是否都大于0
            is_max = torch.unsqueeze(is_max, dim=1) #(B,1,H,W),每个元素表示在0和4方向是不是最大值

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0 #is_max == 0取出非最大值的部分（直接取反则包括了非梯度和非0.4方向的位置），因此和is_oriented_i再与一下
            thin_edges[to_remove] = 0.0 #(B,1,H,W)，虽然to_remove是(B,1,H,W)，但a[b]=0这种写法会自动将a的b为True的值对应坐标的值置0

        # thresholds
        if low_threshold is not None:
            low = thin_edges > low_threshold #nms后的梯度图大于low_threshold的mask

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5 #为0.5的部分表示仅大于low_threshold，为1的部分大于high_threshold
                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1 #高低阈值之间的mask
                    weak_is_high = (self.hysteresis(thin_edges) == 1.0) * weak  #thin_edges中原先的0也可能变成1，因此再*weak，则只剩下0.5变1的pixel
                    thin_edges = high * 1. + weak_is_high * 1.
            else:
                thin_edges = low * 1
        dilate = get_dilate_kernel()
        thin_edges = dilate(thin_edges)

        return thin_edges, grad_magnitude*thin_edges, grad_x*thin_edges, grad_y*thin_edges  #thin_edges即为mask

if __name__ == "__main__":
    #边缘比opencv的canny更稀
    #img = image.read_image('P0004.png',mode = image.ImageReadMode.GRAY).float()
    img3 = image.read_image('image/P0003.png').float()
    img3 = img3.to('cuda:0')
    img4 = image.read_image('image/P0003.png').float()
    img4 = img4.to('cuda:0')
    img = torch.stack((img3,img4))
    img = img.repeat(64,1,1,1)
    print(img.shape)
    model = CannyFilter()
    model = model.to('cuda:0')
    l,h = 40,80


    count_time = []
    for i in range(10):
        with torch.no_grad():
            start = time.time()
            for i in range(2):
                mask, img_l,img_x,img_y = model(img, 40, 80)
            end = time.time()
            count_time.append(end-start)
            print('canny_pytorch:', end-start)
    print('平均时间:',sum(count_time[1:])/(len(count_time)-1))

    '''
    img_l_4 = F.max_pool2d(img_l, kernel_size=5, stride=4, padding=2) #降采样4倍
    img_l_8 = F.max_pool2d(img_l_4, kernel_size=3, stride=2, padding=1) #降采样8倍
    img_l_16 = F.max_pool2d(img_l_8, kernel_size=3, stride=2, padding=1) #降采样16倍
    img_l_32 = F.max_pool2d(img_l_16, kernel_size=3, stride=2, padding=1) #降采样32倍
    print('梯度百分比:',torch.sum(mask.cpu()[0][0])/(1440*1920))
    print('梯度百分比:',torch.sum(mask.cpu()[1][0]) / (1440 * 1920))

    cv2.imwrite('image/P0003_gray.jpg', img_l.cpu().numpy()[0][0])
    cv2.imwrite('image/P0003_gray_4.jpg', img_l_4.cpu().numpy()[0][0])
    cv2.imwrite('image/P0003_gray_8.jpg', img_l_8.cpu().numpy()[0][0])
    cv2.imwrite('image/P0003_gray_16.jpg', img_l_16.cpu().numpy()[0][0])
    cv2.imwrite('image/P0003_gray_32.jpg', img_l_32.cpu().numpy()[0][0])
    '''
    '''
    #opencv_canny
    img = image.read_image('image/P0003.png').float()
    sum_core = torch.Tensor([[[[0.114]], [[0.587]], [[0.299]]]])
    img = F.conv2d(img, weight=sum_core, stride=1)  # (B,1,H,W)
    img = img.squeeze(0).squeeze(0).numpy()
    img = np.uint8(img)
    time_count = []
    for i in range(50):
        start = time.time()
        for i in range(1024):
            img_ = cv2.Canny(img,40,80)  #opencv本身就会使用自适应阈值，比自己设计的还好
        end = time.time()
        time_count.append(end-start)
        print('opencv_canny',end-start)
    print('平均时间:', sum(time_count[1:])/(len(time_count)-1))
    #cv2.imwrite('canny_cv2.jpg', img_)
    '''

