# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 14:04
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : Cloud_Segmentation.py
# @Software: PyCharm
from typing import List

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class CloudSegDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('background', 'cloud'),
        palette=[[250, 237, 205], [212, 163, 115]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs
                 ):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

    def get_cat_ids(self, idx: int) -> List[int]:
        pass
