import torch.nn as nn
import torch
from mmrotate.utils import get_loss
from mmrotate.core import obb2poly, poly2obb


from mmrotate.models.builder import ROTATED_LOSSES
from mmdet.models.losses.utils import weighted_loss
from torch.autograd import gradcheck

@weighted_loss
#速度还可以
def my_loss_v2(pred, target, **kwargs):  # **kwargs为[b,3,h,w]

    assert pred.size() == target.size() #and target.numel() > 0
    img = kwargs['oringin_img']  # 梯度向量的requires_grad为False
    loss = get_loss(img,target,pred) #log显示的loss全是nan,因为除了0
    #loss.sum().backward()
    print('pred梯度', pred.grad) #第一次为None，后续全报错
    #print(gradcheck(lambda x: get_loss(img,target,x), pred))
    return loss


    '''
    #for mean smooth l1
    #可以完成训练，不会出现nan
    poly_pb = obb2poly(pred, version='le90')  # [num,8]
    poly_gt = obb2poly(target, version='le90')
    pred = poly2obb(poly_pb,version = 'le90')
    target = poly2obb(poly_gt, version='le90')
    beta=1.0
    img = kwargs['oringin_img']
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.Tensor().cuda()
    for i in range(len(pred)):
        diff = torch.abs(pred[i] - target[i])
        loss = torch.cat((loss,torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta).mean().unsqueeze(0)), 0)
    return loss
    '''
    '''
    #standard smooth l1
    beta = 1.0
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                                            diff - 0.5 * beta)
    return loss
    '''



@ROTATED_LOSSES.register_module()
class MyLoss_v2(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MyLoss_v2, self).__init__()
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
        loss_bbox = self.loss_weight * my_loss_v2(
            pred, target, **kwargs)
        return loss_bbox


