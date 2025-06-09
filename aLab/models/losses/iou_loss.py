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
            avg_factor = torch.sum(pos_mask).float().item() + 1e-6

        # step3. 正例结果
        pos_weights = weights[pos_mask].float()
        pos_pred_bboxes = preds[pos_mask].view(-1, 4)
        pos_target_bboxes = targets[pos_mask].view(-1, 4)

        return (pos_weights, pos_pred_bboxes, pos_target_bboxes, avg_factor)
    
    def _get_intersection_wh(self, pred_bboxes: Tensor, target_bboxes: Tensor) -> Tensor:
        intersection_lt = torch.max(pred_bboxes[:, :2], target_bboxes[:, :2])
        intersection_rb = torch.min(pred_bboxes[:, 2:], target_bboxes[:, 2:])
        intersection_wh = (intersection_rb - intersection_lt + 1).clamp(min=0)
        return intersection_wh

    def _get_bbox_wh(self, bboxes: Tensor) -> Tensor:
        bbox_wh = (bboxes[:, 2:] - bboxes[:, :2] + 1).clamp(min=0)
        return bbox_wh

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
        intersection_wh = self._get_intersection_wh(pos_pred_bboxes, pos_target_bboxes)
        intersection = intersection_wh[:, 0] * intersection_wh[:, 1]

        # step3. 并集
        pos_pred_wh = self._get_bbox_wh(pos_pred_bboxes)
        pos_pred_area = pos_pred_wh[:, 0] * pos_pred_wh[:, 1]

        pos_target_wh = self._get_bbox_wh(pos_target_bboxes)
        pos_target_area = pos_target_wh[:, 0] * pos_target_wh[:, 1]

        union = pos_pred_area + pos_target_area - intersection

        # step4. ious
        ious = intersection / (union + self.eps)

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
    def _get_enclose_wh(self, pred_bboxes: Tensor, target_bboxes: Tensor) -> Tensor:
        enclose_lt = torch.min(pred_bboxes[:, :2], target_bboxes[:, :2])
        enclose_rb = torch.max(pred_bboxes[:, 2:], target_bboxes[:, 2:])
        enclose_wh = (enclose_rb - enclose_lt + 1).clamp(min=0)
        return enclose_wh

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
        intersection_wh = self._get_intersection_wh(pos_pred_bboxes, pos_target_bboxes)
        intersection = intersection_wh[:, 0] * intersection_wh[:, 1]

        # step3. 并集
        pos_pred_wh = self._get_bbox_wh(pos_pred_bboxes)
        pos_pred_area = pos_pred_wh[:, 0] * pos_pred_wh[:, 1]

        pos_target_wh = self._get_bbox_wh(pos_target_bboxes)
        pos_target_area = pos_target_wh[:, 0] * pos_target_wh[:, 1]

        union = pos_pred_area + pos_target_area - intersection

        # step4. ious
        ious = intersection / (union + self.eps)
        
        # step5. 最小外接矩形框 (c in paper)
        enclose_wh = self._get_enclose_wh(pos_pred_bboxes, pos_target_bboxes)
        enclose = enclose_wh[:, 0] * enclose_wh[:, 1]

        # step6. gious
        gious = ious - (enclose - union) / enclose

        # step7. gious loss
        loss_gious = 1 - gious

        loss = (loss_gious * pos_weights).sum() / (avg_factor + self.eps)

        return loss * self.loss_weight


@LOSS.register_module()
class FocalEIoULoss(GIoULoss):
    def __init__(self, eps: float = 1e-6, gamma: float = 0.5, loss_weight: float = 1.0) -> None:
        super(FocalEIoULoss, self).__init__(eps, loss_weight)
        self.gamma = gamma

    def _get_bbox_center(self, bboxes: Tensor) -> Tensor:
        bbox_center = (bboxes[:, :2] + bboxes[:, 2:]) / 2
        return bbox_center

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
        intersection_wh = self._get_intersection_wh(pos_pred_bboxes, pos_target_bboxes)
        intersection = intersection_wh[:, 0] * intersection_wh[:, 1]

        # step3. 并集
        pos_pred_wh = self._get_bbox_wh(pos_pred_bboxes)
        pos_pred_area = pos_pred_wh[:, 0] * pos_pred_wh[:, 1]

        pos_target_wh = self._get_bbox_wh(pos_target_bboxes)
        pos_target_area = pos_target_wh[:, 0] * pos_target_wh[:, 1]

        union = pos_pred_area + pos_target_area - intersection

        # step4. ious
        ious = intersection / (union + self.eps)
        loss_ious = 1 - ious

        # step5. 最小外接矩形框
        enclose_wh = self._get_enclose_wh(pos_pred_bboxes, pos_target_bboxes)
        enclose_w = torch.pow(enclose_wh[:, 0], 2)
        enclose_h = torch.pow(enclose_wh[:, 1], 2)

        # step6. 中心点距离loss
        pos_pred_center = self._get_bbox_center(pos_pred_bboxes)
        pos_target_center = self._get_bbox_center(pos_target_bboxes)

        center_distance = torch.pow((pos_pred_center[:, 0] - pos_target_center[:, 0]), 2) + torch.pow((pos_pred_center[:, 1] - pos_target_center[:, 1]), 2)
        loss_distance = center_distance / (enclose_w + enclose_h + self.eps)

        # step7. aspect loss
        loss_w = torch.pow((pos_pred_wh[:, 0] - pos_target_wh[:, 0]), 2) / (enclose_w + self.eps)
        loss_h = torch.pow((pos_pred_wh[:, 1] - pos_target_wh[:, 1]), 2) / (enclose_h + self.eps)
        loss_aspect = loss_w + loss_h
        
        # step8. eiou loss
        loss_eious = loss_ious + loss_distance + loss_aspect
        
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