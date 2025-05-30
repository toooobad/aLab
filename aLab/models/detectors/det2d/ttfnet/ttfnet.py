# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

from aLab.register import MODEL
from ..centernet.centernet import CenterNet


__all__ = ['TTFNet']


@MODEL.register_module()
class TTFNet(CenterNet):
    
    @property
    def output_names(self) -> list:
        return ['heatmap', 'bboxes', 'local_maximum_heatmap']