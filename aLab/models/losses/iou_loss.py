# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional

from aLab.register import LOSS


__all__ = ['IoULoss', 'GIoULoss', 'FocalEIoULoss']


@LOSS.register_module()
class IoULoss(nn.Module):
    def __init__(self, eps: float = 1e-6, loss_weight: float = 1.0) -> None:
        super(IoULoss, self).__init__()
        self.eps = eps
        self.loss_weight = loss_weight
    
    def _get_pos_samples(self, preds: Tensor, targets: Tensor, weights: Tensor, avg_factor: Optional[int] = None) -> tuple:
        if weights.dim() > 1:
            weights = weights.mean(-1)

        # step1. 获取正例mask
        pos_mask = weights > 0

        # step2. avg_factor
        if avg_factor is None:
            avg_factor = pos_mask.float().sum()

        # step3. 正例结果
        pos_weights = weights[pos_mask].float()
        pos_pred_bboxes = preds[pos_mask].view(-1, 4)
        pos_target_bboxes = targets[pos_mask].view(-1, 4)

        return (pos_weights, pos_pred_bboxes, pos_target_bboxes, avg_factor)
    
    def forward(self,
                preds: Tensor,
                targets: Tensor,
                weights: Optional[Tensor],
                avg_factor: Optional[int] = None,
                **kwargs) -> Tensor:
        """
        preds (Tensor): [bhw, (x1, y1, x2, y2)]
        targets (Tensor): [bhw, (x1, y1, x2, y2)]
        weights (Tensor): [bhw, 1] or [bhw, 4] or [bhw]
        """
        # step1. get pose samples
        pos_weights, pos_pred_bboxes, pos_target_bboxes, avg_factor = self._get_pos_samples(preds, targets, weights, avg_factor)

        # step2. 交集
        i_x1s = torch.maximum(pos_pred_bboxes[:, 0], pos_target_bboxes[:, 0])
        i_y1s = torch.maximum(pos_pred_bboxes[:, 1], pos_target_bboxes[:, 2])
        i_x2s = torch.minimum(pos_pred_bboxes[:, 2], pos_target_bboxes[:, 2])
        i_y2s = torch.minimum(pos_pred_bboxes[:, 3], pos_target_bboxes[:, 3])
        i_ws = (i_x2s - i_x1s).clamp(min=0)
        i_hs = (i_y2s - i_y1s).clamp(min=0)
        i_areas = i_ws * i_hs

        # step3. 并集
        pos_pred_ws = (pos_pred_bboxes[:, 2] - pos_pred_bboxes[:, 0]).clamp(min=0)
        pos_pred_hs = (pos_pred_bboxes[:, 3] - pos_pred_bboxes[:, 1]).clamp(min=0)
        pos_pred_areas = pos_pred_ws * pos_pred_hs

        pos_target_ws = (pos_target_bboxes[:, 2] - pos_target_bboxes[:, 0]).clamp(min=0)
        pos_target_hs = (pos_target_bboxes[:, 3] - pos_target_bboxes[:, 1]).clamp(min=0)
        pos_target_areas = pos_target_ws * pos_target_hs

        u_areas = pos_pred_areas + pos_target_areas - i_areas

        # step4. ious
        ious = i_areas / (u_areas + self.eps)

        # step5. ious loss
        loss_ious = 1 - ious

        loss = (loss_ious * pos_weights).sum() / (avg_factor + self.eps)

        return loss * self.loss_weight
    
    def __repr__(self) -> str:
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(eps={self.eps}, '
        repr_str += f'loss_weight={self.loss_weight})'
        return repr_str


@LOSS.register_module()
class GIoULoss(IoULoss):
    def forward(self,
                preds: Tensor,
                targets: Tensor,
                weights: Optional[Tensor],
                avg_factor: Optional[int] = None,
                **kwargs) -> Tensor:
        """
        preds (Tensor): [bhw, (x1, y1, x2, y2)]
        targets (Tensor): [bhw, (x1, y1, x2, y2)]
        weights (Tensor): [bhw, 1] or [bhw, 4] or [bhw]
        """
        # step1. get pose samples
        pos_weights, pos_pred_bboxes, pos_target_bboxes, avg_factor = self._get_pos_samples(preds, targets, weights, avg_factor)

        # step2. 交集
        i_x1s = torch.maximum(pos_pred_bboxes[:, 0], pos_target_bboxes[:, 0])
        i_y1s = torch.maximum(pos_pred_bboxes[:, 1], pos_target_bboxes[:, 2])
        i_x2s = torch.minimum(pos_pred_bboxes[:, 2], pos_target_bboxes[:, 2])
        i_y2s = torch.minimum(pos_pred_bboxes[:, 3], pos_target_bboxes[:, 3])
        i_ws = (i_x2s - i_x1s).clamp(min=0)
        i_hs = (i_y2s - i_y1s).clamp(min=0)
        i_areas = i_ws * i_hs

        # step3. 并集
        pos_pred_ws = (pos_pred_bboxes[:, 2] - pos_pred_bboxes[:, 0]).clamp(min=0)
        pos_pred_hs = (pos_pred_bboxes[:, 3] - pos_pred_bboxes[:, 1]).clamp(min=0)
        pos_pred_areas = pos_pred_ws * pos_pred_hs

        pos_target_ws = (pos_target_bboxes[:, 2] - pos_target_bboxes[:, 0]).clamp(min=0)
        pos_target_hs = (pos_target_bboxes[:, 3] - pos_target_bboxes[:, 1]).clamp(min=0)
        pos_target_areas = pos_target_ws * pos_target_hs

        u_areas = pos_pred_areas + pos_target_areas - i_areas

        # step4. ious
        ious = i_areas / (u_areas + self.eps)
        
        # step5. c in paper
        c_x1s = torch.minimum(pos_pred_bboxes[:, 0], pos_target_bboxes[:, 0])
        c_y1s = torch.minimum(pos_pred_bboxes[:, 1], pos_target_bboxes[:, 1])
        c_x2s = torch.maximum(pos_pred_bboxes[:, 2], pos_target_bboxes[:, 2])
        c_y2s = torch.maximum(pos_pred_bboxes[:, 3], pos_target_bboxes[:, 3])
        c_ws = (c_x2s - c_x1s).clamp(min=0)
        c_hs = (c_y2s - c_y1s).clamp(min=0)
        c_areas = c_ws * c_hs

        # step6. gious
        gious = ious - (c_areas - u_areas) / (c_areas + self.eps)

        # step7. gious loss
        loss_gious = 1 - gious

        loss = (loss_gious * pos_weights).sum() / (avg_factor + self.eps)

        return loss * self.loss_weight


@LOSS.register_module()
class FocalEIoULoss(IoULoss):
    def __init__(self, eps: float = 1e-6, gamma: float = 0.5, loss_weight: float = 1.0) -> None:
        super(FocalEIoULoss, self).__init__(eps, loss_weight)
        self.gamma = gamma
    
    def forward(self,
                preds: Tensor,
                targets: Tensor,
                weights: Optional[Tensor],
                avg_factor: Optional[int] = None,
                **kwargs) -> Tensor:
        """
        preds (Tensor): [bhw, (x1, y1, x2, y2)]
        targets (Tensor): [bhw, (x1, y1, x2, y2)]
        weights (Tensor): [bhw, 1] or [bhw, 4] or [bhw]
        """
        # step1. get pose samples
        pos_weights, pos_pred_bboxes, pos_target_bboxes, avg_factor = self._get_pos_samples(preds, targets, weights, avg_factor)

        # step2. 交集
        i_x1s = torch.maximum(pos_pred_bboxes[:, 0], pos_target_bboxes[:, 0])
        i_y1s = torch.maximum(pos_pred_bboxes[:, 1], pos_target_bboxes[:, 2])
        i_x2s = torch.minimum(pos_pred_bboxes[:, 2], pos_target_bboxes[:, 2])
        i_y2s = torch.minimum(pos_pred_bboxes[:, 3], pos_target_bboxes[:, 3])
        i_ws = (i_x2s - i_x1s).clamp(min=0)
        i_hs = (i_y2s - i_y1s).clamp(min=0)
        i_areas = i_ws * i_hs

        # step3. 并集
        pos_pred_ws = (pos_pred_bboxes[:, 2] - pos_pred_bboxes[:, 0]).clamp(min=0)
        pos_pred_hs = (pos_pred_bboxes[:, 3] - pos_pred_bboxes[:, 1]).clamp(min=0)
        pos_pred_areas = pos_pred_ws * pos_pred_hs

        pos_target_ws = (pos_target_bboxes[:, 2] - pos_target_bboxes[:, 0]).clamp(min=0)
        pos_target_hs = (pos_target_bboxes[:, 3] - pos_target_bboxes[:, 1]).clamp(min=0)
        pos_target_areas = pos_target_ws * pos_target_hs

        u_areas = pos_pred_areas + pos_target_areas - i_areas

        # step4. ious
        ious = i_areas / (u_areas + self.eps)
        loss_ious = 1 - ious

        # step5. 最小外接矩形框
        c_x1s = torch.minimum(pos_pred_bboxes[:, 0], pos_target_bboxes[:, 0])
        c_y1s = torch.minimum(pos_pred_bboxes[:, 1], pos_target_bboxes[:, 1])
        c_x2s = torch.maximum(pos_pred_bboxes[:, 2], pos_target_bboxes[:, 2])
        c_y2s = torch.maximum(pos_pred_bboxes[:, 3], pos_target_bboxes[:, 3])
        c_ws = torch.pow((c_x2s - c_x1s).clamp(min=0), 2)
        c_hs = torch.pow((c_y2s - c_y1s).clamp(min=0), 2)

        # step6. dis loss
        pos_pred_cxs = (pos_pred_bboxes[:, 0] + pos_pred_bboxes[:, 2]) / 2
        pos_pred_cys = (pos_pred_bboxes[:, 1] + pos_pred_bboxes[:, 3]) / 2

        pos_target_cxs = (pos_target_bboxes[:, 0] + pos_target_bboxes[:, 2]) / 2
        pos_target_cys = (pos_target_bboxes[:, 1] + pos_target_bboxes[:, 3]) / 2

        center_dis = torch.pow((pos_pred_cxs - pos_target_cxs), 2) + torch.pow((pos_pred_cys - pos_target_cys), 2)
        loss_dis = center_dis / (c_ws + c_hs + self.eps)

        # step7. asp loss
        loss_w = torch.pow((pos_pred_ws - pos_target_ws), 2) / (c_ws + self.eps)
        loss_h = torch.pow((pos_pred_hs - pos_target_hs), 2) / (c_hs + self.eps)
        loss_asp = loss_w + loss_h
        
        # step8. eiou loss
        loss_eious = loss_ious + loss_dis + loss_asp
        
        # step9. focal eiou loss (更关注低质量的样本, 低iou)
        # TODO: 这里换成exp会不会更好？(使用exp, 大幅度增加低iou样本的权重, 正常样本的权重增加的小, 但最小为1), 但会不会变成过度关注低质量样本，影响正常样本的训练？
        if self.gamma > 0:
            loss_eious = torch.pow(loss_ious, self.gamma) * loss_eious

        loss = (loss_eious * pos_weights).sum() / (avg_factor + self.eps)

        return loss * self.loss_weight
    
    def __repr__(self) -> str:
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(eps={self.eps}, '
        repr_str += f'gamma={self.gamma}, '
        repr_str += f'loss_weight={self.loss_weight})'
        return repr_str