import cv2
import numpy as np
import torch
import torch.nn as nn
from  torchvision import utils as vutils
from mmrotate.utils import gradient_extractor, get_nonzero_tensor, draw_rectangle
from mmrotate.core import obb2poly
import math




from mmrotate.models.builder import ROTATED_LOSSES
from mmdet.models.losses.utils import weighted_loss

@weighted_loss
def my_loss(pred, target,**kwargs): #**kwargs为[b,3,h,w]

    assert pred.size() == target.size() and target.numel() > 0
    gradient, gradient_x, gradient_y = gradient_extractor(kwargs['oringin_img'])  #梯度向量的requires_grad为False
    pred_ploy = obb2poly(pred, version='le90')  #[num,8]
    target_ploy = obb2poly(target, version='le90')
    #print('target_ploy:', target_ploy)


    '''
    #在原图上画上矩形框
    #print('原图shape', kwargs['oringin_img'].cpu().numpy().shape)
    img = np.transpose(kwargs['oringin_img'].cpu().numpy(), (0, 2, 3, 1))  #交换通道次序
    img = np.ascontiguousarray(img, dtype=np.uint8)
    print('交换通道后shape:', img.shape)
    point = list(target_ploy[0].cpu().numpy())
    print('point[0]:', point)
    draw_img0 = draw_rectangle(img[0], list(point))
    cv2.imwrite('/opt/data/private/shen/mmrotate-0.3.3/work_dir/myloss/draw_img0.png',draw_img0)
    #尽管图片被随机翻转和缩放，但gt也做了对应改变
    '''



    split_gradient_x_th, split_gradient_y_th, split_gradient_mask, offset = \
        split_gradient_img(pred_ploy, target_ploy, gradient, gradient_x, gradient_y)  #已经归一化的非零梯度向量，及其位置mask
    loss_list = []
    for i in range(len(pred_ploy)):  #计算每个小图（每个预测框）的loss  ,可不可以整体计算？
        row_x = split_gradient_x_th[i]
        col_x = torch.unsqueeze(row_x, dim=1)

        row_y = split_gradient_y_th[i]
        col_y = torch.unsqueeze(row_y, dim=1)
        w = col_x * row_x + col_y * row_y  #w的requires_grad为False

        approxi_1, approxi_2 = get_approxi(split_gradient_mask[i], pred[i], target[i], offset[i])
        loss_ = torch.mean(w * approxi_2 * approxi_1)
        loss_list.append(loss_)

    loss = torch.Tensor(loss_list).cuda()
    return loss




    '''
    
    #smooth l1
    beta=1.0
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    print('pred:', pred)
    print('target:',target)
    return loss
    '''

def split_gradient_img(pred_ploy, target_ploy, gradient, gradient_x, gradient_y):
    shape = gradient_x.shape
    height, width = shape[2], shape[3]
    split_gradient_x_th = []
    split_gradient_y_th = []
    split_gradient_mask = []
    offset = []
    #print("梯度shape：",gradient.shape)   #没问题
    #print("预测框shape：",pred_ploy.shape)
    #print('预测框：',pred_ploy)
    #print("GT shape：",target_ploy.shape)
    #print('GT框：',target_ploy)
    for i in range(len(pred_ploy)):
        one_pred_ploy = pred_ploy[i]
        one_target_ploy = target_ploy[i]
        max_x = int(max(one_pred_ploy[0],one_pred_ploy[2],one_pred_ploy[4],one_pred_ploy[6],one_target_ploy[0],one_target_ploy[2],one_target_ploy[4],one_target_ploy[6]))
        min_x = int(min(one_pred_ploy[0],one_pred_ploy[2],one_pred_ploy[4],one_pred_ploy[6],one_target_ploy[0],one_target_ploy[2],one_target_ploy[4],one_target_ploy[6]))

        max_y = int(max(one_pred_ploy[1], one_pred_ploy[3], one_pred_ploy[5], one_pred_ploy[7], one_target_ploy[1], one_target_ploy[3],
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
        offset.append([min_x, min_y])   #小图的左上角坐标
        #print('小图左上角坐标:',offset)
        split_gradient_x_ = torch.split(gradient_x,[min_y,max_y-min_y+1,height-max_y-1],2)[1]
        split_gradient_x = torch.split(split_gradient_x_,[min_x,max_x-min_x+1,width-max_x-1],3)[1]
        #split_gradient_x_img.append(split_gradient_x)
        #vutils.save_image(split_gradient_x, './test/'+str(i)+'x.jpg', normalize=True)

        split_gradient_y_ = torch.split(gradient_y, [min_y, max_y - min_y + 1, height - max_y - 1], 2)[1]
        split_gradient_y = torch.split(split_gradient_y_, [min_x, max_x - min_x + 1, width - max_x - 1], 3)[1]
        #split_gradient_y_img.append(split_gradient_y)
        #vutils.save_image(split_gradient_y, './test/' + str(i) + 'y.jpg', normalize=True)

        split_gradient_ = torch.split(gradient, [min_y, max_y - min_y + 1, height - max_y - 1], 2)[1]
        split_gradients = torch.split(split_gradient_, [min_x, max_x - min_x + 1, width - max_x - 1], 3)[1]
        #split_gradient_imgs.append(split_gradients)
        #print("小图个数：",split_gradients.shape)
        nonzero_split_gradient_x, nonzero_split_gradient_y, mask = get_nonzero_tensor(split_gradients, split_gradient_x, split_gradient_y)  #分割出的梯度小图提取非零部分
        split_gradient_x_th.append(nonzero_split_gradient_x[0])
        split_gradient_y_th.append(nonzero_split_gradient_y[0]) #[小图数目，y方向梯度]
        split_gradient_mask.append(mask[0])
        
        #if (len(nonzero_split_gradient_x.squeeze(0)) is 0):  #判断tensor是否为空
            #print('nan对应过滤前的小图split_gradient_x：', split_gradient_x)
            #save_img_with_poly(gradient, one_pred_ploy,'nan_img_pred_ploy.png')
            #save_img_with_poly(gradient, one_target_ploy, 'nan_img_target_ploy.png')
    return split_gradient_x_th, split_gradient_y_th, split_gradient_mask, offset  #返回所有小图的非零梯度，非零梯度坐标，以及小图左上角位置


def get_approxi(mask, pred,target,offset, k = 15, theta2 = 400): #计算某一个小图的某梯度是否在GT内的近似，以及Predbox相同位置附近的近似
    pred_x = pred[0] - offset[0]
    pred_y = pred[1] - offset[1]
    pred_w = pred[2]
    pred_h = pred[3]
    pred_a = pred[4]
    gt_x = target[0] - offset[0]
    gt_y = target[1] - offset[1]
    gt_w = target[2]
    gt_h = target[3]
    gt_a = target[4]
    approxi_1 = []
    approxi_2 = []
    for i in range(len(mask)):  #计算每个梯度点i的近似
        d = pow((mask[i][0]-gt_x)**2 + (mask[i][1]-gt_y)**2, 0.5)
        if gt_y >= mask[i][1]:
            beta = gt_a + math.acos((gt_x - mask[i][0])/d)  #mask[i][1]
        else:
            beta = gt_a - math.acos((gt_x - mask[i][0])/d)
        d_w = abs(d * math.cos(beta))
        d_h = abs(d * math.sin(beta))
        approxi_ = (1-1/(1+math.exp(-k*(d_w-gt_w)/gt_w))) * (1-1/(1+math.exp(-k*(d_h-gt_h)/gt_h)))
        approxi_1.append(approxi_)

        d_w_p = pred_w * d_w/gt_w  #按相对位置计算预测框取提梯度的地方
        d_h_p = pred_h * d_h / gt_h
        points = obb2poly(torch.tensor([0,0,d_w_p*2,d_h_p*2,pred_a]).cuda().unsqueeze(0), version='le90').squeeze(0) #潜在的4个相对位置点，还要根据beta来确定
        if 0<beta<=torch.pi/2:
            guass_x = pred_x+points[0]
            guass_y = pred_y + points[1]
        if -torch.pi/2<beta<=0:
            guass_x = pred_x + points[2]
            guass_y = pred_y + points[3]
        if -math.pi<beta<=-math.pi/2:
            guass_x = pred_x + points[4]
            guass_y = pred_y + points[5]
        if math.pi/2<beta<=math.pi:
            guass_x = pred_x + points[6]
            guass_y = pred_y + points[7]
        approxi_i = []
        for j in range(len(mask)):
            approxi_ij = math.exp(((guass_x-mask[j][0])**2 + (guass_y-mask[j][1])**2)/(2*theta2**2))/(2*math.pi*theta2)
            approxi_i.append(approxi_ij)
        approxi_2.append(approxi_i)
    return approxi_1, approxi_2

'''
def get_approxi(mask, pred,target,offset, k = 15, theta2 = 400):
    pred_x = pred[0] - offset[0]  #pred_等的requires_grad都为True
    pred_y = pred[1] - offset[1]
    pred_w = pred[2]
    pred_h = pred[3]
    pred_a = pred[4]
    gt_x = target[0] - offset[0]  #gt_等的requires_grad都为False
    gt_y = target[1] - offset[1]
    gt_w = target[2]
    gt_h = target[3]
    gt_a = target[4]

    d = torch.pow(torch.pow((mask[:,1] - gt_x), 2) + torch.pow((mask[:,0] - gt_y), 2), 0.5)
    beta_mask__ = mask[:,0].clone()
    #下面两行代码生成一个beta_mask，某梯度向量是在gt_y上方，则对应值为1，下方为-1
    #beta_mask_ = torch.nn.Threshold(gt_y, 1)(beta_mask__)  #比gt_y小的置成1
    #beta_mask = -torch.nn.Threshold(-gt_y, 1)(-beta_mask_)  #比gt_y大的置成-1, 1不可能比gt_y大

    #beta = gt_a - beta_mask * torch.atan((gt_y - mask[:, 0])/(gt_x - mask[:, 1]))
    if (gt_x - mask[:, 1])>0:
        if (gt_y - mask[:, 0])>0:
            beta = gt_a - (torch.atan((gt_y - mask[:, 0]) / (gt_x - mask[:, 1]))-180)
        else: beta = gt_a - (torch.atan((gt_y - mask[:, 0]) / (gt_x - mask[:, 1]))+180)
    else:
        beta = gt_a - (torch.atan((gt_y - mask[:, 0]) / (gt_x - mask[:, 1])))


    beta = gt_a - torch.atan((gt_y - mask[:, 0]) / (gt_x - mask[:, 1]))
    d_w = torch.abs(d * torch.cos(beta))
    d_h = torch.abs(d * torch.sin(beta))
    approxi_1 = (1 - torch.sigmoid(k * (d_w - gt_w) )) * (1 - torch.sigmoid(k * (d_h - gt_h) ))  #要不要(d_w - gt_w)再除以gt_w?

    beta_p = torch.atan(torch.tan(beta)*(pred_h*gt_w)/(gt_h*pred_w))  #预测框相同相对位置的β角
    for i in range(len(beta_p)):
        if beta[i] >= torch.pi/2:
            beta_p[i] += torch.pi/2
        if beta[i] <= -torch.pi / 2:
            beta_p[i] += -torch.pi / 2
    d_p = torch.pow(torch.pow(d * torch.cos(beta) * pred_w /gt_w, 2) + torch.pow(d * torch.sin(beta) * pred_h /gt_h, 2), 0.5)
    guass_x = pred_x + d_p * torch.cos(beta_p+pred_a)  #行向量
    guass_y = pred_y + d_p * torch.sin(beta_p + pred_a)
    approxi_2 = torch.exp(((guass_x.unsqueeze(1) - mask[:, 0]) ** 2 + (guass_y.unsqueeze(1) - mask[:,1]) ** 2) / (2 * theta2)) / (
                2 * torch.pi * theta2)
    return approxi_1.unsqueeze(1), approxi_2
'''


def my_threshold(x,th1, th2): #把tensor中在th1与th2之间的置1，其余置0
    y1 = torch.nn.Threshold(th1,0)(x)
    y2 = -torch.nn.Threshold(-0.0001,-1)(-y1)
    return y2

def save_img_with_poly(img):
    img = img.squeeze(0)
    img = img.repeat(3,1,1)
    img = img.cpu().numpy()
    #poly = poly.detach().cpu().numpy()
    #contours = np.array([[poly[0],poly[1]],[poly[2],poly[3]],[poly[4],poly[5]],[poly[6],poly[7]]],dtype=np.int)
    #cv2.polylines(img, [contours], isClosed=True, color=[0, 0, 255], thickness=5)
    #print(img.shape)
    #print(img)
    print('图片shaped：', img.shape)
    for i in img:
        for j in i:
            for k in j:
                if k >255:
                    print('异常值：', k)

    #cv2.imwrite(save_name, img)
















@ROTATED_LOSSES.register_module()
class MyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * my_loss(
            pred, target, **kwargs)
        return loss_bbox


