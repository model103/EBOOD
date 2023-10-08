import json
import shutil
from mmrotate.core import obb2poly, poly2obb
import torch
screw_annotaion_file = '/opt/data/private/shen/data/screw/mvtec_screws_train.json'  #标注文件
screw_imgs_path = '/opt/data/private/shen/data/screw/images/'  #图片路径
dota_save_path = '/opt/data/private/shen/data/screw_dota/train/'

with open(screw_annotaion_file,'r',encoding='utf-8') as f:
    coco_annotaion = json.load(f)  #load为dict类型

#获取标注数据
categories = [dic.get('name') for dic in coco_annotaion['categories']]  #取所有类别
categories_id = [dic.get('id')-1 for dic in coco_annotaion['categories']]  #取所有类别id

imgs_name = [dic.get('file_name') for dic in coco_annotaion['images']]
imgs_id = [dic.get('id')-1 for dic in coco_annotaion['images']]

obbs = [dic.get('bbox') for dic in coco_annotaion['annotations']]  #(row,col,w,h,-theta) #theta范围（-pi,pi），且方向与le90相反

obbs_category_id = [dic.get('category_id')-1 for dic in coco_annotaion['annotations']]
obbs_imgs_id = [dic.get('image_id')-1 for dic in coco_annotaion['annotations']]

img_obbs_num = []  #每张图片的框数目
id = 0
p = 0
for i in range(len(obbs_imgs_id)):
    if obbs_imgs_id[i] != id:
        num = i-p
        p = i
        id = obbs_imgs_id[i]
        img_obbs_num.append(num)
    if i == len(obbs_imgs_id) - 1:  #最后一个框
        img_obbs_num.append(i+1-p)

'''
#移动图片
for img_name in imgs_name:
    shutil.copy(screw_imgs_path+img_name,dota_save_path+'images/')
'''

#进行转换
sum_obb = 0  #已经处理的图片的所有框的数目
for img_id in imgs_id:
    txt_name = imgs_name[img_id].split('.')[0]
    img_annotations = open(dota_save_path + 'annfiles/'+txt_name+'.txt','w')
    polys = torch.Tensor(obbs[sum_obb:sum_obb + img_obbs_num[img_id]])  #该图片id的几个框
    polys[:,[0,1]] = polys[:,[1,0]]  #交换两列 (x,y,w,h,-theta), (-pi,pi)
    polys[:, 4] = -polys[:,4]  #(x,y,w,h,theta),(-pi,pi)
    for i in range(len(polys)): #(x,y,w,h,theta),(-pi/2,pi/2)
        if polys[i][4] > torch.pi/2:
            polys[i][4] = polys[i][4] - torch.pi
        if polys[i][4] < -torch.pi/2:
            polys[i][4] = polys[i][4] + torch.pi
    polys = obb2poly(polys, 'le90')

    for i in range(len(polys)):  #某张图片的几个box
        s = ''
        for _ in polys[i]:
            s += (str(round(float(_),1)) + ' ')
        s += categories[obbs_category_id[sum_obb+i]] + ' ' + '0\n'
        img_annotations.write(s)

    sum_obb += img_obbs_num[img_id]
