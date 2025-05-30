# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import os.path as osp

from typing import List

from aLab.register import DATASET
from .base_dataset_det2d import BaseDatasetDet2D


__all__ = ['DemoDatasetDet2D']


@DATASET.register_module()
class DemoDatasetDet2D(BaseDatasetDet2D):
    def __init__(self, demo: str, preprocesses: List[dict] = None) -> None:
        super(DemoDatasetDet2D, self).__init__(test_mode=True, preprocesses=preprocesses)
        self.samples = self._collect_samples(demo)
        self.total_samples = len(self.samples)
        assert self.total_samples > 0, ValueError(f'No valid samples were found in {demo}.')
    
    def _collect_samples(self, demo: str) -> List[str]:
        if osp.isfile(demo):  # 单张图片
            samples = [demo]

        elif osp.isdir(demo): # 文件夹
            samples = []
            for root, _, files in os.walk(demo):
                for file in files:
                    extname = osp.splitext(file)[-1]
                    if extname not in ['.jpg', '.png', '.bmp', '.jpeg']:
                        continue

                    samples.append(osp.join(root, file))
        else:
            raise ValueError('Currently, pictures and folders containing pictures are supported.')
        
        return samples

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index) -> dict:
        data = dict(img_id=index,
                    filename=self.samples[index])
        data = self.preprocesses(data)
        return data
