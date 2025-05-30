# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import numpy as np

from torch.utils.data import Dataset
from typing import (List, Tuple, Sequence)

from aLab.preprocess import ComposePreprocess
from aLab.structures import (InstanceData, DetDataSample)


class BaseDatasetDet2D(Dataset):
    def __init__(self,
                 test_mode: bool = False,
                 preprocesses: List[dict] = None,
                 max_fetch: int = 100) -> None:
        super(BaseDatasetDet2D, self).__init__()
        self.test_mode = test_mode
        self.max_fetch = max_fetch
        self.preprocesses = ComposePreprocess(preprocesses)
    
    def _rand_another(self) -> int:
        return np.random.randint(0, len(self))
    
    @staticmethod
    def collate_fn(batch: List[dict]) -> Tuple[dict, dict, list]:
        batch_samples = dict(data_samples=[], inputs=dict(imgs=[]))

        for data in batch:
            metainfo = {}
            instance_data = InstanceData()

            for k, v in data.items():
                if k == 'img':
                    if isinstance(v, Sequence):
                        imgs = [torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))) for img in v]
                        imgs = torch.stack(imgs, dim=0)
                    else:
                        imgs = torch.from_numpy(np.ascontiguousarray(v.transpose(2, 0, 1)))
                    
                    batch_samples['inputs']['imgs'].append(imgs)

                elif k.startswith('gt_'):
                    instance_data[k[3:]] = torch.from_numpy(v)
                else:
                    metainfo[k] = v

            data_sample = DetDataSample(metainfo=metainfo)
            data_sample.gt_instances = instance_data

            batch_samples['data_samples'].append(data_sample)

        batch_samples['inputs']['imgs'] = torch.stack(batch_samples['inputs']['imgs'], dim=0)
        
        return batch_samples