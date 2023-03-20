# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 14:25
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : ConvNext_tiny.py
# @Software: PyCharm
_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/Cloud_Seg_Dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_100_e.py'
]
