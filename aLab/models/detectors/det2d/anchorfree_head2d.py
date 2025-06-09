# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch.nn as nn

from torch import Tensor
from typing import (Tuple, List, Optional)

from aLab.register import LOSS
from aLab.structures import DetDataSample
from aLab.models.utils import unpack_gt_instances


__all__ = ['AnchorFreeHead2D']


class AnchorFreeHead2D(nn.Module):
    def __init__(self,
                 loss_cfgs: dict,
                 in_channels: int,
                 num_classes: int,
                 initial_cfgs: Optional[dict] = dict(use_bg_cls=False, score_thrs=0.01)) -> None:
        super(AnchorFreeHead2D, self).__init__()
        
        self.loss_cfgs = loss_cfgs
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.initial_cfgs = initial_cfgs
        self.cls_out_channels = self.num_classes

        self.score_thrs = 0.01
        self.use_bg_cls = False
        
        if self.initial_cfgs is not None:
            self.score_thrs = self.initial_cfgs.get('score_thrs', 0.01)
            self.use_bg_cls = self.initial_cfgs.get('use_bg_cls', False)
        
        # 是否使用bg类
        if self.use_bg_cls:
            self.cls_out_channels += 1
            
        self._init_heads()
        self._init_losses()
        self._init_weights()
        
    def _init_heads(self) -> None:
        NotImplementedError
    
    def _init_weights(self) -> None:
        NotImplementedError
    
    def _init_losses(self) -> None:
        if self.loss_cfgs is not None:
            for loss_key, loss_cfg in self.loss_cfgs.items():
                loss = LOSS.build(loss_cfg)
                setattr(self, f'loss_{loss_key}', loss)

    def forward(self, *args, **kwargs) -> Tuple[Tensor]:
        NotImplementedError

    def loss(self, img_feats: Tensor, batch_data_samples: List[DetDataSample]) -> dict:
        # step1. forward
        batch_preds = self(img_feats, training=True)

        # step2. losses
        (batch_gt_instances, batch_metainfos) = unpack_gt_instances(batch_data_samples)
        losses = self.get_losses(batch_preds=batch_preds, batch_gt_instances=batch_gt_instances, batch_metainfos=batch_metainfos)
        return losses
    
    def get_losses(self, *args, **kwargs) -> dict:
        NotImplementedError

    def predict(self, img_feat: Tensor, batch_data_samples: List[DetDataSample], rescale=True, to_numpy=True) -> List[DetDataSample]:
        # step1. forward
        batch_preds = self(img_feat, training=False)

        # step2. postprocess
        predictions = self.postprocesser.postprocess(batch_preds, batch_data_samples=batch_data_samples, rescale=rescale, to_numpy=to_numpy)

        return predictions
