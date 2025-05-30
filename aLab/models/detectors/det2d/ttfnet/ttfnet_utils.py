# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
from torch import Tensor
from typing import (Union, List, Tuple)

from aLab.models.detectors.det2d.centernet import gaussian2d


__all__ = ['draw_truncate_gaussian']


def draw_truncate_gaussian(heatmap: Tensor, obj_center: Union[List[int], Tuple[int]], h_radius: int, w_radius: int, k: int = 1) -> Tensor:
    """
    copy from ttfnet <https://github.com/ZJULearning/ttfnet/blob/master/mmdet/models/anchor_heads/ttf_head.py, line 273>
    """
    h, w = 2 * h_radius + 1, 2 * w_radius + 1
    sigma_x = w / 6
    sigma_y = h / 6
    gaussian = gaussian2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
    gaussian = heatmap.new_tensor(gaussian)

    x, y = int(obj_center[0]), int(obj_center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, w_radius), min(width - x, w_radius + 1)
    top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius - left:w_radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap