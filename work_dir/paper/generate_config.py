#DOTA数据集默认png格式图片，暂时把/root/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmrotate/datasets/dota.py里的png全改成bmp

from mmdet.apis import set_random_seed

from mmcv import Config
cfg = Config.fromfile('../../configs/redet/redet_re50_refpn_1x_dota_le90.py')
cfg.work_dir = 'E:\OneDrive - stu.hit.edu.cn\E盘\python\mmrotate-0.3.3\work_dir\paper/redet/'

# Modify dataset type and path


cfg.load_from = '../../../checkpoints/r3det_r50_fpn_1x_dota_oc-b1fb045c.pth'

cfg.data.samples_per_gpu=2
cfg.data.workers_per_gpu=2



cfg.data_root = '/opt/data/private/shen/data/DOTA'

cfg.data.test.ann_file = 'E:/data/split_ss_dota/val/annfiles/'
cfg.data.test.img_prefix = 'E:/data/split_ss_dota/val/images/'

cfg.data.train.ann_file = 'E:/data/split_ss_dota/train/annfiles/'
cfg.data.train.img_prefix = 'E:/data/split_ss_dota/train/iamges/'

cfg.data.val.ann_file = 'E:/data/split_ss_dota/val/annfiles/'
cfg.data.val.img_prefix = 'E:/data/split_ss_dota/val/images/'

#cfg.model.roi_head.bbox_head[0].num_classes=1
#cfg.model.roi_head.bbox_head[1].num_classes=1



#cfg.optimizer.lr = 0.005
#cfg.lr_config.warmup = None
cfg.runner.max_epochs = 12
cfg.log_config.interval = 100   #每下降10次展示一次loss

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 1  #每迭代1次评估一次mAP
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 1  #每迭代3次存储一次训练模型

# Set seed thus the results are more reproducible
cfg.seed = 518
#set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)


# We can also use tensorboard to log the training process
#cfg.log_config.hooks = [
#    dict(type='TextLoggerHook'),
#    dict(type='TensorboardLoggerHook')]

# We can initialize the logger for training and have a look
# at the final config used for training
#cfg.model.roi_head.bbox_head.loss_bbox=dict(type='RotatedIoULoss')#, loss_weight=1.0)
print(f'Config:\n{cfg.pretty_text}')
cfg.dump(F'{cfg.work_dir}/redet_config.py')  #保存配置文件