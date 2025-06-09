# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

from aLab.register import MODEL
from ..detector2d import Detector2D


__all__ = ['CenterNet']


@MODEL.register_module()
class CenterNet(Detector2D):
    
    @property
    def output_names(self) -> list:
        return ['heatmap', 'wh', 'offset', 'local_maximum_heatmap']