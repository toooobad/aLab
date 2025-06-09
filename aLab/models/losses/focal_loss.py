# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import torch.nn as nn

from torch import Tensor

from aLab.register import LOSS


__all__ = ['GaussianFocalLoss']


@LOSS.register_module()
class GaussianFocalLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, gamma: float = 4.0, loss_weight: float = 1.0) -> None:
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        pos_inds = targets.eq(1).float()
        neg_inds = targets.lt(1).float()

        neg_weights = torch.pow(1 - targets, self.gamma)  # reduce punishment

        pos_loss = -torch.log(preds) * torch.pow(1 - preds, self.alpha) * pos_inds
        neg_loss = -torch.log(1 - preds) * torch.pow(preds, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        if num_pos == 0:
            loss =  neg_loss
        else:
            loss = (pos_loss + neg_loss) / num_pos

        return loss * self.loss_weight
    
    def __repr__(self) -> str:
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(alpha={self.alpha}, '
        repr_str += f'gamma={self.gamma}, '
        repr_str += f'loss_weight={self.loss_weight})'
        return repr_str