import cv2
import torch
import torchvision.transforms as transforms
from  torchvision import utils as vutils
import torch.nn.functional as F
import numpy as np
from mmrotate.core import obb2poly, poly2obb
from torchvision.io import image
import time
from mmrotate.utils.Canny import CannyFilter
import random
import mmcv

def gradient_extractor(img):  #(B 3,H,W)
    sum_core = torch.Tensor([[[[0.114]],[[0.587]],[[0.299]]]]).cuda()
    img_gray = F.conv2d(img, weight=sum_core, stride=1)  #(B,1,H,W)
    filter_x = torch.Tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).cuda()
    gradient_x = F.conv2d(img_gray, weight=filter_x, padding='same', stride=1)
    filter_y = torch.Tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).cuda()
    gradient_y = F.conv2d(img_gray, weight=filter_y, padding='same', stride=1)
    gradient_length = gradient_x * gradient_x + gradient_y * gradient_y #(B,1,H,W)

    '''
    flatten = torch.flatten(gradient_length,2) #将灰度图拉成一条直线 (B,1,H*W)
    #mean_g= torch.mean(flatten,dim=-1)  #max:(B,1) ，可不可以用均值代替max
    max_g,_ = torch.max(flatten,dim=-1)
    #print(max_g)
    th = max_g/4

    #阈值处理方法一
    th = th.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)  # (B,1,H,W),将个图片的阈值处理成和图片一样的shape
    gradient_length[gradient_length < th] = 0
    mask = torch.where(gradient_length > 0, 1, gradient_length)  # 将gradient_length中大于0的用1替换，否则不变
    gradient_x_th = gradient_x * mask
    gradient_y_th = gradient_y * mask
    #vutils.save_image(gradient_length, './gradient_mean.jpg', normalize=True)  #保存图片


    #求mask方法二,目前还有bug，也很慢
    gradient_length, gradient_x_th, gradient_y_th = [],[],[]
    for i in range(len(gradient_length)):
        gradient_length[i] = torch.nn.Threshold(th, 0)(gradient_length[i].squeeze(0))
        mask = -torch.nn.Threshold(-0.00001, -1)(-gradient_length[i])  #非零梯度的位置mask
        gradient_x_th[i] = gradient_x[i].squeeze(0) * mask
        gradient_y_th[i] = gradient_y[i].squeeze(0) * mask
        gradient_length.append(gradient_length[i].unsqueeze(0))
        gradient_x_th.append(gradient_x_th[i].unsqueeze(0))
        gradient_y_th.append(gradient_y_th[i].unsqueeze(0))
    gradient_length = torch.tensor(gradient_length).cuda()
    gradient_x_th = torch.tensor(gradient_x_th).cuda()
    gradient_y_th = torch.tensor(gradient_y_th).cuda()
    '''
    return torch.sqrt(gradient_length), gradient_x, gradient_y# [B,1,H,W] #先不进行阈值处理

def th_process(gradient_length, gradient_x, gradient_y):  #把mask处理后的梯度进行阈值处理  [h,w]
    H,W = gradient_length.shape
    flatten = torch.flatten(gradient_length)
    max_g, _ = torch.max(flatten, dim=-1)
    th = max_g / 2

    th = th.repeat(H, W)  # 将图片的阈值处理成和图片一样的shape
    gradient_length[gradient_length < th] = 0
    mask = torch.where(gradient_length > 0, 1, gradient_length)  # 将gradient_length中大于0的用1替换，否则不变
    gradient_x_th = gradient_x * mask
    gradient_y_th = gradient_y * mask
    return gradient_length, gradient_x_th, gradient_y_th  #[H,W]


def get_nonzero_tensor(gradient_length_th, gradient_x, gradient_y): #提取非零梯度（x或y分量）,以及他们的坐标 #一次只处理一张图片
    #mask = []
    nonzerolist_x = []
    nonzerolist_y = []

    gradient_length_th_i = gradient_length_th ** 0.5
    gradient_x_i = gradient_x / gradient_length_th_i  # 单位化后x方向梯度大小
    gradient_y_i = gradient_y / gradient_length_th_i

    nozero_mask = []
    nozero_x =[]
    nozero_y = []
    for i in range(len(gradient_length_th_i)):
        mask_i = torch.flip(torch.nonzero(gradient_length_th_i[i].squeeze(0)),[1]) #[num,2](y_coor,x_coor),再翻转为(x_coor,y_coor)
        nozero_mask.append(mask_i)
        one_img_x = gradient_x_i[i].squeeze(0)
        one_img_y = gradient_y_i[i].squeeze(0)
        #print('one_img_x.shape',one_img_x.shape)

        height, width = one_img_x.shape
        #print('小图里的非零梯度个数：', len(onemask))
        if len(mask_i.shape) == 1:  #即只有一个非零梯度时，mask没有维度1,下文onemask[:,1]会有bug
            mask_i = mask_i.unsqueeze(0)
            print('发现仅一个梯度的小图')
        #print("梯度个数:",onemask)
        #dex = onemask[:,1]*width+onemask[:,2]  #非零梯度的顺序索引
        dex = mask_i[:, 1] * width + mask_i[:, 0]  # 非零梯度的顺序索引
        nozero_x.append(torch.take(one_img_x, dex))
        nozero_y.append(torch.take(one_img_y, dex))
    return nozero_x, nozero_y, nozero_mask  #[B,X非零分类],[B,Y非零分量],[B,x_coor,y_coor], B维度为list

def get_nonzero_tensor_inbox(gradient_x_inbox,gradient_y_inbox,gradient_l_inbox):
    mask_i = torch.flip(torch.nonzero(gradient_l_inbox),[1])
    height, width = gradient_l_inbox.shape
    if len(mask_i.shape) == 1:  # 即只有一个非零梯度时，mask没有维度1,下文onemask[:,1]会有bug
        mask_i = mask_i.unsqueeze(0)
        print('发现仅一个梯度的小图')
    dex = mask_i[:, 1] * width + mask_i[:, 0]
    torch.take(gradient_l_inbox, dex)
    #按索引找出非零梯度并归一化
    length = torch.sqrt(torch.take(gradient_l_inbox, dex))
    gradient_inbox = torch.stack((torch.take(gradient_x_inbox, dex)/length,torch.take(gradient_y_inbox, dex)/length),1)
    return gradient_inbox, mask_i  #[num,2](x分类,y分量), [num](x_coor,y_coor)#全局坐标

def get_most_length_inbox(gradient_x_inbox,gradient_y_inbox,gradient_l_inbox, num = 100):
    length,indices = torch.topk(torch.flatten(gradient_l_inbox), num, -1, True, False)
    length = length + 1e-8
    #length = torch.sqrt(length)
    gradient_most = torch.stack((torch.take(gradient_x_inbox, indices) / length, torch.take(gradient_y_inbox, indices) / length), 1)
    #求最长的num个梯度的位置
    _, width = gradient_l_inbox.shape
    mask_y = torch.div(indices, width, rounding_mode='trunc')
    mask_x = indices - mask_y * width
    mask = torch.stack((mask_x, mask_y),1)
    return gradient_most, mask




'''
def in_box(x, y, box): #判断位置(x,y)是否在某个box内 #也可用ploy的叉乘方法判断
    flag = False
    k_h = torch.tan(box[4]) #h方向的斜率
    k_w = torch.tan(box[4] + torch.pi/2)
    if torch.abs(y-k_h*x + k_h*box[0] - box[1])/torch.sqrt(1+k_h**2) < box[3]/2 and \
            torch.abs(y-k_w*x + k_w*box[0] - box[1])/torch.sqrt(1+k_w**2) < box[2]/2:
        flag = True
    return flag
'''
def getcross(p1,p2,p): # 计算 (p1 p) x (p1 p2)
    x1 = p[0] - p1[0]
    y1 = p[1] - p1[1]
    x2 = p2[0] - p1[0]
    y2 = p2[1] - p1[1]
    return x1*y2 -x2*y1

def in_box(corners, p): #判断点p是否在4个角构成的矩形内
    p1 = corners[0:2]
    p2 = corners[2:4]
    p3 = corners[4:6]
    p4 = corners[6:]
    #p = [p[1],p[0]]  #nozero_maskd的坐标书写方式是[y,x]
    return (getcross(p1,p2,p) >=0 and getcross(p2,p3,p) >=0 and getcross(p3,p4,p) >=0 and getcross(p4,p1,p) >=0) or \
        (getcross(p1, p2, p) <= 0 and getcross(p2, p3, p) <= 0 and getcross(p3, p4, p) <= 0 and getcross(p4, p1,p) <= 0)  #即四项同号

    #return GetCross(p1,p2,p) * GetCross(p3,p4,p) >=0 and GetCross(p2,p3,p) * GetCross(p4,p1,p) >=0

#该无法同时处理一个batch，因为出了pb，gt
def get_gradients_inbox(nozero_x, nozero_y, nozero_mask,box): #nozero_x, nozero_y,nozero_mask为一张图片的数据取出在box内的梯度[X非零分类][Y非零分类][x_coor,y_coor]，box一个框为poly模式
    up = max(box[1],box[3],box[5],box[7])  #box的上下左右边界
    down = min(box[1],box[3],box[5],box[7])
    left = min(box[0],box[2],box[4],box[6])
    right = max(box[0], box[2], box[4], box[6])
    gradients_inbox = []
    inbox_mask = torch.Tensor().cuda()
    time1 = time.time()
    n = 0
    for i in range(len(nozero_mask)):
        if left<=nozero_mask[i][0]<=right and down<=nozero_mask[i][1]<=up:
            if in_box(box,nozero_mask[i]):
                n+=1
                gradients_inbox.append([nozero_x[i],nozero_y[i]])
                inbox_mask = torch.cat((inbox_mask,nozero_mask[i].unsqueeze(0)),0)

    time2 = time.time()
    print('取梯度时间：', time2 - time1)
    print('框内梯度个数', n)
    return torch.Tensor(gradients_inbox).cuda(),inbox_mask #[num,2](两列分别为x分量，y分量),[num,2](两列分别为x,y坐标)


def get_ralative_pos(p, gt, poly_gt, poly_pb):  # p[num,2](x_coor,y_coor),pb,poly_gt,poly_pb都是一个框
    r = torch.sqrt((p[:, 0] - poly_gt[0]) ** 2 + (p[:, 1] - poly_gt[1]) ** 2 + 1e-8)  # [num] ，与左后点的距离r,
    #开方后求导出现在分母，可以加一个很小的余项1e-8，或者不开放（后续代码作相应调整）
    theta = torch.arctan((p[:, 1] - poly_gt[1]) / (p[:, 0] - poly_gt[0] + 1e-8))  # 弧度制, [num] ，r的与水平方向的夹角,90°时分母为0，添加1e-8
    relative_w = torch.abs(r * torch.cos(theta - gt[4]) / gt[2])  #
    relative_h = torch.abs(r * torch.sin(theta - gt[4]) / gt[3])
    #print(relative_w)
    #print(relative_h)
    p1 = poly_pb[0:2]  # 顺时针排列的
    p2 = poly_pb[2:4]
    p4 = poly_pb[6:]
    p5 = (p2 - p1) * relative_w.unsqueeze(0).t() + p1
    p6 = (p4 - p1) * relative_h.unsqueeze(0).t() + p1
    return p5 + p6 - p1  #[num,2](x_coor,y_coor)

def euclid_dis(a,b): #(m,2),(n,2),a的行向量依次与b的行向量求欧氏距离平方，返回（m,n）tensor
    sq_a = a ** 2
    sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b ** 2
    sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
    return sum_sq_a + sum_sq_b - 2 * torch.mm(a,b.t())  #[m,n]



'''
def get_loss(img, gt,pb, sigma = 0.1): #计算hot_map_loss,gt,pb为obb形式,nozero_x, nozero_y为一张图片的梯度
    poly_pb = obb2poly(pb, version='le90')  # [num,8]
    poly_gt = obb2poly(gt, version='le90')
    h,w = img.shape[2:]
    gradient_length_th, gradient_x_th, gradient_y_th = gradient_extractor(img)
    nozero_x, nozero_y, nozero_mask = get_nonzero_tensor(gradient_length_th, gradient_x_th, gradient_y_th)
    loss = torch.Tensor().cuda()
    #zero_gradient_num = 0
    for i in range(len(pb)):
        gradients_in_pb,coor_in_pb = get_gradients_inbox(nozero_x[0], nozero_y[0], nozero_mask[0],poly_pb[i]) #[m,2]
        gradients_in_gt, coor_in_gt = get_gradients_inbox(nozero_x[0], nozero_y[0], nozero_mask[0], poly_gt[i])#[n,2]
        time1 = time.time()
        #if gradients_in_gt.numel() and gradients_in_pb.numel():  #可能某些框内没梯度
        w = torch.mm(gradients_in_gt,gradients_in_pb.t())  #[n,m]
        ralative_pos_in_pb = get_ralative_pos(coor_in_gt,gt[i],poly_gt[i], poly_pb[i])
        diagn = pb[i][2]**2 +pb[i][3]**2
        heat_map = torch.exp(-euclid_dis(ralative_pos_in_pb,coor_in_pb)/(2 *diagn* sigma ** 2))#/(2*torch.pi * sigma) #(m,n)
        loss = torch.cat((loss,-torch.mean(w*heat_map).unsqueeze(0)),0)
        time2 = time.time()
        print('算loss时间：', time2 - time1)
        #else:
        #    loss = torch.cat((loss,torch.Tensor([0]).cuda()),0)
        #    zero_gradient_num += 1
    return loss
'''
def get_loss(img, gt,pb, sigma = 0.1): #计算hot_map_loss,gt,pb为obb形式,nozero_x, nozero_y为一张图片的梯度
    #提梯度》》取每个框内的梯度》》阈值处理》》取框内非零梯度》》算每个框的loss
    poly_pb = obb2poly(pb, version='le90')  # [num,8]
    poly_gt = obb2poly(gt, version='le90') #等下转np只能是cpu类型
    #detach_poly_pb = poly_pb.detach()  #用于计算w，不需要反向求导
    hight,width = img.shape[2:]
    low_th = 40
    high_th = 80
    with torch.no_grad():
        canny_model = CannyFilter().to('cuda:0')
        gradient_length, gradient_x, gradient_y = canny_model(img, (low_th-117.037)/57.53, (high_th-117.037)/57.53)
    print('gradient_x',gradient_x.requires_grad)
        #norm后的图像对应高低阈值也要对应处理
    #gradient_length, gradient_x, gradient_y = gradient_extractor(img)
    #gradient_length_th = gradient_length ** 0.5
    #gradient_x_th = gradient_x_th / gradient_length_th  # 单位化后x方向梯度大小
    #gradient_y_th = gradient_y_th / gradient_length_th
    loss = torch.Tensor().cuda()
    num = 100
    poly_gt_ = poly_gt.detach()
    poly_pb_ = poly_pb.detach()
    #zero_gradient_num = 0
    for i in range(len(gt)):
        #with torch.no_grad():
        gt_mask = torch.Tensor(cv2.fillConvexPoly(np.zeros((hight, width, 1)), poly_gt_[i].view(4, 2).long().cpu().numpy(), (1))).cuda().squeeze(-1)  #(h,w)的mask，框内全为1
        pb_mask = torch.Tensor(cv2.fillConvexPoly(np.zeros((hight, width, 1)), poly_pb_[i].view(4, 2).long().cpu().numpy(), (1))).cuda().squeeze(-1) #不求梯度
        gradient_x_in_gtbox = gradient_x.squeeze(0).squeeze(0) * gt_mask   #gt框内的x梯度，框外全为0
        gradient_y_in_gtbox = gradient_y.squeeze(0).squeeze(0) * gt_mask  #不求梯度
        gradient_l_in_gtbox = gradient_length.squeeze(0).squeeze(0) * gt_mask
        gradient_x_in_pbbox = gradient_x.squeeze(0).squeeze(0) * pb_mask #不求梯度
        gradient_y_in_pbbox = gradient_y.squeeze(0).squeeze(0) * pb_mask #不求梯度
        gradient_l_in_pbbox = gradient_length.squeeze(0).squeeze(0) * pb_mask #不求梯度
            #l_th_ingtbox,x_th_ingtbox,y_th_ingtbox = th_process(gradient_l_in_gtbox,gradient_x_in_gtbox,gradient_y_in_gtbox)  #阈值处理
            #l_th_inpbbox, x_th_inpbbox, y_th_inpbbox = th_process(gradient_l_in_pbbox, gradient_x_in_pbbox,gradient_y_in_pbbox)  #未归一化
        #取出gt框内的梯度及其坐标

        
            #nonzero_g_in_gtbox, coor_in_gt = get_nonzero_tensor_inbox(x_th_ingtbox,y_th_ingtbox,l_th_ingtbox)
            #nonzero_g_in_pbbox, coor_in_pb = get_nonzero_tensor_inbox(x_th_inpbbox, y_th_inpbbox,l_th_inpbbox)
        most_g_in_gtbox, coor_in_gt = get_most_length_inbox(gradient_x_in_gtbox, gradient_y_in_gtbox, gradient_l_in_gtbox, num)
        most_g_in_pbbox, coor_in_pb = get_most_length_inbox(gradient_x_in_pbbox, gradient_y_in_pbbox, gradient_l_in_pbbox, num)
        w = 1/(torch.mm(most_g_in_gtbox, most_g_in_pbbox.t())+1.5) #(m,n)  #不是w的问题,w的形状没问题

        ralative_pos_in_pb = get_ralative_pos(coor_in_gt, gt[i], poly_gt[i], poly_pb[i]) #会求梯度
        diagn = pb[i][2] ** 2 + pb[i][3] ** 2 + 1e-8 #会求梯度
        heat_map = torch.exp(-euclid_dis(ralative_pos_in_pb, coor_in_pb.float()) / (2 * diagn * sigma ** 2))  # /(2*torch.pi * sigma) #(m,n)，会求梯度
        heat_map.retain_grad()
        abox_loss = torch.mean(w * heat_map)
        loss = torch.cat((loss, abox_loss.unsqueeze(0)), 0)  #不应该是求平均，而是求和除m(有待商议) 会求梯度
        #temp = (nonzero_g_in_gtbox.numel()+1)/( nonzero_g_in_pbbox.numel()+1) * (coor_in_gt.numel()+1/coor_in_pb.numel()+1)
        #temp = -(w).mean()
        #，ralative_pos_in_pb,不可以,
        # ，nonzero_g_in_gtbox，nonzero_g_in_pbbox，l_th_inpbbox 可以
        #w，nonzero_g_in_pbbox前面几轮迭代可以，后面全是nan，学习率太高
        #temp = torch.rand(1).cuda().squeeze(0) #随机数可以显示loss
        #temp = w.shape[0]/w.shape[1]  #w的shape可以
        #print(-(w).mean())
        #print(torch.rand(1).squeeze(0))
        #print(w.shape[0]/w.shape[1])
        #print('sss')


        #diff = torch.abs(pb[i] - gt[i])
        #smooth = torch.where(diff < 1, 0.5 * diff * diff / 1, diff - 0.5 * 1).mean().unsqueeze(0)
        #loss = torch.cat((loss, temp*smooth), 0)
    #torch.cuda.empty_cache()  #会拖慢速度，可以高几个iter再情况一次（得找到在哪里迭代的）

    return loss





def split_gradient_img(pred_ploy, target_ploy, gradient, gradient_x, gradient_y):
    shape = gradient_x.shape
    height, width = shape[2], shape[3]
    split_gradient_x_th = []
    split_gradient_y_th = []
    split_gradient_mask = []
    offset = []
    # print("梯度shape：",gradient.shape)   #没问题
    # print("预测框shape：",pred_ploy.shape)
    # print('预测框：',pred_ploy)
    # print("GT shape：",target_ploy.shape)
    # print('GT框：',target_ploy)
    for i in range(len(pred_ploy)):
        one_pred_ploy = pred_ploy[i]
        one_target_ploy = target_ploy[i]
        max_x = int(max(one_pred_ploy[0], one_pred_ploy[2], one_pred_ploy[4], one_pred_ploy[6], one_target_ploy[0],
                        one_target_ploy[2], one_target_ploy[4], one_target_ploy[6]))
        min_x = int(min(one_pred_ploy[0], one_pred_ploy[2], one_pred_ploy[4], one_pred_ploy[6], one_target_ploy[0],
                        one_target_ploy[2], one_target_ploy[4], one_target_ploy[6]))

        max_y = int(max(one_pred_ploy[1], one_pred_ploy[3], one_pred_ploy[5], one_pred_ploy[7], one_target_ploy[1],
                        one_target_ploy[3],
                        one_target_ploy[5], one_target_ploy[7]))
        min_y = int(min(one_pred_ploy[1], one_pred_ploy[3], one_pred_ploy[5], one_pred_ploy[7], one_target_ploy[1],
                        one_target_ploy[3],
                        one_target_ploy[5], one_target_ploy[7]))
        if max_y > height - 1:
            max_y = height - 1
        if max_x > width - 1:
            max_x = width - 1
        if min_y < 0:
            min_y = 0
        if min_x < 0:
            min_x = 0
        offset.append([min_x, min_y])  # 小图的左上角坐标
        # print('小图左上角坐标:',offset)
        split_gradient_x_ = torch.split(gradient_x, [min_y, max_y - min_y + 1, height - max_y - 1], 2)[1]
        split_gradient_x = torch.split(split_gradient_x_, [min_x, max_x - min_x + 1, width - max_x - 1], 3)[1]
        # split_gradient_x_img.append(split_gradient_x)
        # vutils.save_image(split_gradient_x, './test/'+str(i)+'x.jpg', normalize=True)

        split_gradient_y_ = torch.split(gradient_y, [min_y, max_y - min_y + 1, height - max_y - 1], 2)[1]
        split_gradient_y = torch.split(split_gradient_y_, [min_x, max_x - min_x + 1, width - max_x - 1], 3)[1]
        # split_gradient_y_img.append(split_gradient_y)
        # vutils.save_image(split_gradient_y, './test/' + str(i) + 'y.jpg', normalize=True)

        split_gradient_ = torch.split(gradient, [min_y, max_y - min_y + 1, height - max_y - 1], 2)[1]
        split_gradients = torch.split(split_gradient_, [min_x, max_x - min_x + 1, width - max_x - 1], 3)[1]
        # split_gradient_imgs.append(split_gradients)
        # print("小图个数：",split_gradients.shape)
        nonzero_split_gradient_x, nonzero_split_gradient_y, mask = get_nonzero_tensor(split_gradients, split_gradient_x,
                                                                                      split_gradient_y)  # 分割出的梯度小图提取非零部分
        split_gradient_x_th.append(nonzero_split_gradient_x)
        split_gradient_y_th.append(nonzero_split_gradient_y)  # [小图数目，y方向梯度]
        split_gradient_mask.append(mask)

        # if (len(nonzero_split_gradient_x.squeeze(0)) is 0):  #判断tensor是否为空
        # print('nan对应过滤前的小图split_gradient_x：', split_gradient_x)
        # save_img_with_poly(gradient, one_pred_ploy,'nan_img_pred_ploy.png')
        # save_img_with_poly(gradient, one_target_ploy, 'nan_img_target_ploy.png')
    return split_gradient_x_th, split_gradient_y_th, split_gradient_mask, offset  # 返回所有小图的非零梯度，非零梯度坐标，以及小图左上角位置


def draw_rectangle(image, point):  #numpy, list
    p1_new = [int(point[0]), int(point[1])]
    p2_new = [int(point[2]), int(point[3])]
    p3_new = [int(point[4]), int(point[5])]
    p4_new = [int(point[6]), int(point[7])]

    img = cv2.line(image, p1_new, p2_new, (0, 255, 0), 1)
    img = cv2.line(image, p2_new, p3_new, (0, 255, 0), 1)
    img = cv2.line(image, p3_new, p4_new, (0, 255, 0), 1)
    img = cv2.line(image, p1_new, p3_new, (0, 255, 0), 1)
    img = cv2.line(image, p2_new, p4_new, (0, 255, 0), 1)

    return img







if __name__ == '__main__':
    from torch.autograd import gradcheck
    #img = image.read_image('screws_002.png').float()
    img = cv2.imread('image/screws_002.png')  #BGR
    #cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

    #归一化
    mean=np.array([123.675, 116.28, 103.53])
    std=np.array([58.395, 57.12, 57.375])
    img = mmcv.imnormalize(img,mean,std)  #RGB

    img = transforms.ToTensor()(img).cuda().unsqueeze(0)
    print('img', img.requires_grad)
    boxs = []
    with open('image/screws_002.txt') as f:
        lines = f.readlines()
        for line in lines:
            boxs.append(list(map(float,line.split(' ')[:8])))
    boxs = torch.Tensor(boxs).cuda()
    gt = poly2obb(boxs, version='le90')
    pb = poly2obb(boxs, version='le90')
    #pb[:,0:2] = pb[:,0:2]+15  #偏移10像素
    #pb[:,4] = pb[:,4] + torch.pi/12  #偏移10°
    pb[:,2:4] = pb[:,2:4]*1.2  #拓宽1.2倍
    pb.requires_grad_()
    print('pb',pb.requires_grad)
    #偏移后loss确实变大了
    loss = get_loss(img,gt,pb)
    print('loss',loss)
    loss.sum().backward()
    print(pb.grad)
    print('是不是叶子结点：',pb.is_leaf)
    #test = gradcheck(lambda x: get_loss(img,gt,x), pb)
    #print("Are the gradients correct: ",test)
    '''

    x,y,l = gradient_extractor(img)
    cv2.imwrite('sobel.jpg', l.cpu().numpy()[0][0])
    '''



    '''
    img = gradient_length_th[0].squeeze(0)
    img = img.repeat(3, 1, 1)
    img = img.cpu().numpy()
    print('图片shaped：', img.shape)
    for i in img:
        for j in i:
            for k in j:
                if k > 255:
                    print('异常值：', k)
    '''

    '''
    f=open('P0003.txt',"r")
    point = []
    with open('P0003.txt',"r") as f:
        for line in f:
            point.append(list(line.strip('\n').split(' ')))
    print(img.numpy().shape)
    img = np.transpose(img.numpy(), (1, 2, 0))  #通道排序不一样
    print('point:', point[0])
    img = draw_rectangle(img, point[0])

    cv2.imwrite('rectangle_img.png', img)
    '''
