# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import torch.nn as nn

from torch import Tensor
from copy import deepcopy
from typing import (Union, List)

from aLab.structures import DetDataSample


__all__ = ['unpack_gt_instances', 'perspective_transform', 'inverse_sigmoid', 'clone_modules']


def unpack_gt_instances(batch_data_samples: List[DetDataSample]) -> tuple:
    batch_metainfos = []
    batch_gt_instances = []
    
    for data_sample in batch_data_samples:
        batch_metainfos.append(data_sample.metainfo)
        batch_gt_instances.append(data_sample.gt_instances)

    return (batch_gt_instances, batch_metainfos)


def perspective_transform(points: Tensor, M: Tensor, device='cpu'):
    points = torch.cat([points, torch.ones((points.shape[0], 1), device=device)], dim=1)
    transformed_points = torch.mm(points, M.t())
    transformed_points = transformed_points / transformed_points[:, 2:3]
    transformed_points = transformed_points[:, :2]

    return transformed_points


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def clone_modules(module: Union[nn.Module, nn.Sequential], N: int) -> nn.ModuleList:
    return nn.ModuleList([deepcopy(module) for _ in range(N)])