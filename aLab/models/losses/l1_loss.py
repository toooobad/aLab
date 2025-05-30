# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional

from aLab.register import LOSS


__all__ = ['L1Loss']


@LOSS.register_module()
class L1Loss(nn.Module):
    def __init__(self, loss_weight: float = 1.0) -> None:
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                preds: Tensor,
                targets: Tensor,
                weights: Optional[Tensor] = None,
                avg_factor: Optional[int] = None, 
                **kwargs) -> Tensor:
        
        if targets.numel() == 0:
            return preds.sum() * 0

        assert preds.size() == preds.size()
        loss = torch.abs(targets - preds)

        if weights is not None:
            loss = loss * weights

        if avg_factor is None:
            loss = loss.mean()
        else:
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)

        return loss * self.loss_weight
    
    def __repr__(self) -> str:
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(loss_weight={self.loss_weight})'
        return repr_str
