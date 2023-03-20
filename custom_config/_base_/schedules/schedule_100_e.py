# -*- coding: utf-8 -*-
# @Time    : 2023/2/21 14:26
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : schedule_100_e.py
# @Software: PyCharm
# optimizer
init_lr = 0.001
optimizer = dict(type='AdamW', lr=init_lr)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(type='LinearLR',
         start_factor=init_lr,
         by_epoch=True,
         begin=0,
         end=5),
    dict(type='CosineAnnealingLR',
         T_max=100,
         by_epoch=True,
         begin=5,
         end=100)]
# training schedule for 100 epoch

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, save_best='mIoU', max_keep_ckpts=5,
                    rule='greater', out_dir='CrossEntropy'),
    sampler_seed=dict(type='DistSamplerSeedHook'))

custom_hooks = [dict(type='EMAHook')]
