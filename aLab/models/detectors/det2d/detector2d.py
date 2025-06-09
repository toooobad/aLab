# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch.nn as nn

from torch import Tensor
from typing import (Tuple, List, Optional, Union)

from aLab.register import (BACKBONE, NECK, HEAD)
from aLab.structures import DetDataSample


__all__ = ['Detector2D']


class Detector2D(nn.Module):
    def __init__(self, 
                 backbone: dict,
                 head: Optional[dict] = None,
                 neck: Optional[Union[dict, List[dict]]] = None):
        super(Detector2D, self).__init__()

        # backbone
        self.backbone = BACKBONE.build(backbone)

        # necks
        if neck is not None:
            if isinstance(neck, (list, tuple)):
                self.neck = nn.ModuleList()
                for neck_cfgs in neck:
                    self.neck.append(NECK.build(neck_cfgs))
            else:
                self.neck = NECK.build(neck)
        
        # heads
        if head is not None:
            self.head = HEAD.build(head)
    
    @property
    def input_names(self) -> list:
        return ['imgs']
    
    @property
    def output_names(self) -> list:
        return ['cls_scores', 'bboxes']

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None
    
    @property
    def with_head(self):
        """bool: whether the detector has a head"""
        return hasattr(self, 'head') and self.head is not None

    def extract_img_feats(self, img_inputs: Tensor) -> Tuple[Tensor]:
        # backbone
        img_feats = self.backbone(img_inputs)

        # neck
        if self.with_neck:
            if isinstance(self.neck, nn.ModuleList):
                for neck in self.neck:
                    img_feats = neck(img_feats)
            else:
                img_feats = self.neck(img_feats)
        
        return img_feats
    
    def forward(self, 
                inputs: dict,
                batch_data_samples: List[DetDataSample],
                training: bool = True):

        if isinstance(inputs, dict):
            img_inputs = inputs['imgs']
        else:
            img_inputs = inputs

        if training:
            return self.forward_train(img_inputs, batch_data_samples)
        else:
            return self.forward_test(img_inputs, batch_data_samples)

    def forward_train(self, img_inputs: Tensor, batch_data_samples: List[DetDataSample]) -> dict:
        # step1. backbone & neck
        img_feats = self.extract_img_feats(img_inputs)

        # step2. losses
        losses = self.head.loss(img_feats, batch_data_samples=batch_data_samples)

        return losses
    
    def forward_test(self, img_inputs: Tensor, batch_data_samples: List[DetDataSample]) -> List[DetDataSample]:
        # step1. backbone & neck
        img_feats = self.extract_img_feats(img_inputs)

        # step2. predictions
        predictions = self.head.predict(img_feats, batch_data_samples=batch_data_samples, rescale=True, to_numpy=True)

        return predictions
    
    def forward_deploy(self, img_inputs: Tensor) -> tuple:
        # step1. backbone & neck
        img_feats = self.extract_img_feats(img_inputs)
        
        # step2. head
        if hasattr(self.head, 'forward_deploy'):
            preds = self.head.forward_deploy(img_feats)
        else:
            preds = self.head(img_feats)

        return preds
