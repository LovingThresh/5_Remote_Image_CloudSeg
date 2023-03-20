# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 7:58
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : early_verification.py
# @Software: PyCharm
import os

import mmseg
from mmengine.runner import Runner
from mmengine.config import Config
from mmseg.utils import register_all_modules

from runner import *

mmseg.utils.register_all_modules()


def EarlyValidation(cfg_path):
    cfg = Config.fromfile(cfg_path)

    cfg['work_dir'] = ''
    cfg['train_dataloader']['dataset']['indices'] = 500

    print('------------------测试配置文件是否能够构建正确的Runner------------------')
    runner = Runner.from_cfg(cfg)
    print('---------------------------Runner构建成功---------------------------')
    # indices = 5000
    print('------------------测试训练数据是否正常生成与尺寸是否匹配------------------')
    print('-----------------------------请耐心等待-----------------------------')
    train_data = next(runner.train_dataloader)
    assert all(data['inputs'].shape == data['data_samples'].img_shape for data in runner.train_dataloader)
    print('---------------------------训练数据测试通过---------------------------')

    print('------------------测试验证数据是否正常生成与尺寸是否匹配------------------')
    val_data = next(runner.val_dataloader)
    assert all(data['inputs'].shape == data['data_samples'].img_shape for data in runner.val_dataloader)
    print('---------------------------验证数据测试通过---------------------------')

    print('------------------测试训练模型是否正常运行且尺寸是否匹配------------------')
    pred, label = runner.model.predict(inputs=train_data['inputs'], data_samples=train_data['data_samples'])
    assert (pred.shape == label.shape)
    print('---------------------------训练模型预测通过---------------------------')

    print('------------------测试训练模型是否正常运行且损失是否正常------------------')
    loss_dict = runner.model.loss(inputs=train_data['inputs'], data_samples=train_data['data_samples'])
    print('---------------------------训练模型损失通过---------------------------')

    print('---------------------检查runner的Hook是否符合预设---------------------')
    print(runner.get_hooks_info())
    print('----------------------------Hook测试通过----------------------------')

    print('-----------------------删除构建Runner产生的文件-----------------------')
    os.remove(os.path.join(runner.work_dir, runner.timestamp))

    print('----------------------------所有测试通过----------------------------')
