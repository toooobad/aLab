# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import (Tuple, List, Union, Optional)

from aLab.utils import multi_apply
from aLab.register import (HEAD, POSTPROCESS)
from aLab.structures import (DetDataSample, InstanceData)
from aLab.models.utils import (bias_init_with_prob, normal_init)

from ..anchorfree_head2d import AnchorFreeHead2D
from .centernet_utils import (draw_umich_gaussian, gaussian_radius)


__all__ = ['CenterNetHead', 'CenterNetPostprocess']


@HEAD.register_module()
class CenterNetHead(AnchorFreeHead2D):
    """
    Object as point
    code: <https://github.com/xingyizhou/CenterNet>
    paper: <https://arxiv.org/abs/1904.07850>
    """
    def __init__(self,
                 num_classes: int = 80,
                 in_channels: int = 128,
                 wh_feat_channels: int = 128,
                 offset_feat_channels: int = 128,
                 heatmap_feat_channels: int = 128,
                 loss_cfgs: dict = dict(
                    wh=dict(type='L1Loss', loss_weight=0.1),
                    offset=dict(type='L1Loss', loss_weight=1.0),
                    heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0)),
                 initial_cfgs: dict = dict(use_bg_cls=False, score_thrs=0.01)) -> None:

        # head channels
        self.wh_feat_channels = wh_feat_channels
        self.offset_feat_channels = offset_feat_channels
        self.heatmap_feat_channels = heatmap_feat_channels
        
        super(CenterNetHead, self).__init__(loss_cfgs=loss_cfgs, 
                                            num_classes=num_classes, 
                                            in_channels=in_channels, 
                                            initial_cfgs=initial_cfgs)
        # test params
        self.local_maximum_kernel = 3

        if self.initial_cfgs is not None:
            self.local_maximum_kernel = self.initial_cfgs.get('local_maximum_kernel', 3)
            
        self.local_maximum_padding = (self.local_maximum_kernel - 1) // 2

        # postprocess
        self.postprocesser = CenterNetPostprocess(self.local_maximum_kernel, self.local_maximum_padding, score_thrs=self.score_thrs)
    
    def _init_heads(self) -> None:
        self.wh_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.wh_feat_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.wh_feat_channels, 2, kernel_size=1))
        
        self.offset_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.offset_feat_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.offset_feat_channels, 2, kernel_size=1))
        
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.heatmap_feat_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.heatmap_feat_channels, self.num_classes, kernel_size=1))

    def _init_weights(self) -> None:
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)

        for layers in [self.wh_head, self.offset_head]:
            for m in layers.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, mean=0, std=0.001, bias=0)
    
    @staticmethod
    def _clip_sigmoid(x: Tensor) -> Tensor:
        return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    
    def forward(self, img_feats: Union[Tuple[Tensor], Tensor], training: bool = True) -> dict:
        if isinstance(img_feats, Tensor):
            img_feat = img_feats
        else:
            img_feat = img_feats[0]

        wh_preds = self.wh_head(img_feat)
        offset_preds = self.offset_head(img_feat)
        heatmap_preds = self.heatmap_head(img_feat)

        preds = dict(wh=wh_preds,
                     offset=offset_preds)

        if not training:
            heatmap_preds = heatmap_preds.sigmoid()
            local_maximum_heatmaps = F.max_pool2d(heatmap_preds, self.local_maximum_kernel, stride=1, padding=self.local_maximum_padding)
            preds['local_maximum_heatmap'] = local_maximum_heatmaps

        preds['heatmap'] = heatmap_preds

        return preds
    
    def forward_deploy(self, img_feats: Union[Tuple[Tensor], Tensor]) -> Tuple[Tensor]:
        img_feat = img_feats[0]

        wh_preds = self.wh_head(img_feat)
        offset_preds = self.offset_head(img_feat)
        heatmap_preds = self.heatmap_head(img_feat).sigmoid()
        
        local_maximum_heatmaps = F.max_pool2d(heatmap_preds, self.local_maximum_kernel, stride=1, padding=self.local_maximum_padding)

        return heatmap_preds, wh_preds, offset_preds, local_maximum_heatmaps
    
    # ------------------------- losses -------------------------
    def get_losses(self, batch_preds: dict, batch_gt_instances: List[InstanceData], batch_metainfos: List[dict]) -> dict:
        heatmap_preds = self._clip_sigmoid(batch_preds['heatmap'])
        
        assert heatmap_preds.size(0) == len(batch_metainfos) == len(batch_gt_instances)

        # targets
        batch_targets = self._get_targets(batch_gt_instances, heatmap_preds.shape[2:], batch_metainfos)

        # losses
        loss_heatmap = self.loss_heatmap(preds=heatmap_preds, 
                                         targets=batch_targets['heatmap'])
                                         
        loss_wh = self.loss_wh(preds=batch_preds['wh'],
                               targets=batch_targets['wh'],
                               weights=batch_targets['wh_weights'],
                               avg_factor=batch_targets['avg_factor'] * 2)
        
        loss_offset = self.loss_offset(preds=batch_preds['offset'], 
                                       targets=batch_targets['offset'],
                                       weights=batch_targets['offset_weights'], 
                                       avg_factor=batch_targets['avg_factor'] * 2)

        return {'loss_heatmap': loss_heatmap, 'loss_wh': loss_wh, 'loss_offset': loss_offset}

    def _get_targets(self, batch_gt_instances: List[InstanceData], output_size: Tuple[int, int], batch_metainfos: List[dict]) -> dict:
        batch_targets_list = multi_apply(self._get_single_targets, batch_gt_instances, batch_metainfos, output_size=output_size)
        
        (heatmap, wh, wh_weights, offset, offset_weights) = batch_targets_list
        (heatmap, wh, wh_weights, offset, offset_weights) = [torch.stack(t, dim=0).detach() for t in [heatmap, wh, wh_weights, offset, offset_weights]]
    
        batch_targets = dict(
                heatmap=heatmap,
                wh=wh,
                wh_weights=wh_weights,
                offset=offset,
                offset_weights=offset_weights,
                avg_factor=max(1, heatmap.eq(1).sum()))

        return batch_targets
    
    def _get_single_targets(self, gt_instances: InstanceData, metainfo: dict, output_size: Tuple[int]) -> Tuple[Tensor]:
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        output_h, output_w = output_size

        # init targets
        offset = gt_bboxes.new_zeros((2, output_h, output_w))
        offset_weights = gt_bboxes.new_zeros((2, output_h, output_w))

        wh = gt_bboxes.new_zeros((2, output_h, output_w))
        wh_weights = gt_bboxes.new_zeros((2, output_h, output_w))

        heatmap = gt_bboxes.new_zeros((self.num_classes, output_h, output_w))

        # down sample
        input_h, input_w = metainfo['input_size']
        output_bboxes = gt_bboxes.clone()
        output_bboxes[:, [0, 2]] = torch.clamp(output_bboxes[:, [0, 2]] / (input_w // output_w), min=0, max=output_w - 1)
        output_bboxes[:, [1, 3]] = torch.clamp(output_bboxes[:, [1, 3]] / (input_h // output_h), min=0, max=output_h - 1)

        for idx, output_bbox in enumerate(output_bboxes):
            w = output_bbox[2] - output_bbox[0]
            h = output_bbox[3] - output_bbox[1]

            ctx = (output_bbox[0] + output_bbox[2]) / 2
            ctx_int = ctx.int()

            cty = (output_bbox[1] + output_bbox[3]) / 2
            cty_int = cty.int()
            
            cls_id = gt_labels[idx]

            # heatmap                       
            radius = gaussian_radius([h, w],  min_overlap=0.3)
            radius = max(0, int(radius))
            draw_umich_gaussian(heatmap[cls_id], [ctx_int, cty_int], radius)

            # wh
            wh[0, cty_int, ctx_int] = w
            wh[1, cty_int, ctx_int] = h
            wh_weights[:, cty_int, ctx_int] = 1
            
            # offset
            offset[0, cty_int, ctx_int] = ctx - ctx_int
            offset[1, cty_int, ctx_int] = cty - cty_int
            offset_weights[:, cty_int, ctx_int] = 1
        
        return (heatmap, wh, wh_weights, offset, offset_weights)


@POSTPROCESS.register_module()
class CenterNetPostprocess:
    def __init__(self, 
                 local_maximum_kernel: Optional[int] = 3, 
                 local_maximum_padding: Optional[int] = None, 
                 score_thrs: Optional[float] = 0.01):
        self.score_thrs = score_thrs

        self.local_maximum_kernel = local_maximum_kernel
        self.local_maximum_padding = local_maximum_padding
        if self.local_maximum_padding is None:
            self.local_maximum_padding = (self.local_maximum_kernel - 1) // 2
    
    def postprocess(self, batch_preds: dict, batch_data_samples: List[DetDataSample] = None, rescale: bool = False, to_numpy: bool = True) -> List[DetDataSample]:
        heatmap_preds = batch_preds['heatmap']

        # step1. local maximum (nms)
        local_maximum_heatmaps = batch_preds.pop('local_maximum_heatmap', None)
        if local_maximum_heatmaps is None:
            heatmap_preds = heatmap_preds.sigmoid()
            local_maximum_heatmaps = F.max_pool2d(heatmap_preds, self.local_maximum_kernel, stride=1, padding=self.local_maximum_padding)

        keep = (heatmap_preds == local_maximum_heatmaps).float()
        batch_preds['heatmap'] = heatmap_preds * keep

        # step2. 每个batch分开操作
        results = []
        for batch_id, data_sample in enumerate(batch_data_samples):
            single_preds = dict()
            for k, v in batch_preds.items():
                single_preds[k] = v[batch_id]

            prediction = self._process_single_sample(single_preds, data_sample, rescale=rescale, to_numpy=to_numpy)
            
            results.append(prediction)

        return results
    
    def _process_single_sample(self, single_preds: dict, data_sample: DetDataSample, rescale: bool = False, to_numpy: bool = True) -> DetDataSample:
        """Transform outputs of a single image into bbox results.

        Args:
            heatmap_pred (Tensor): Center heatmap for current level with shape (num_classes, output_h, output_w).
            wh_pred (Tensor): WH heatmap for current level with shape (2, output_h, output_w).
            offset_pred (Tensor): Offset for current level with shape (2, output_h, output_w).
            metainfo (dict): Meta information of current image, e.g., image size, scaling factor, etc.

        Returns:
            Detection results of each image after the post process.
            Each item usually contains following keys.
                - img_id (int): image id
                - scores (ndarray): Classification scores, has a shape(num_instance, )
                - labels (ndarray): Labels of bboxes, has a shape (num_instances, ).
                - bboxes (ndarray): Has a shape (num_instances, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        heatmap_pred = single_preds['heatmap']
        output_h, output_w = heatmap_pred.shape[-2:]

        metainfo = data_sample.metainfo
        downsample_scale_w = metainfo['input_size'][1] / output_w
        downsample_scale_h = metainfo['input_size'][0] / output_h

        """
        这里没有按源码中操作
        """
        # 按channels维度取最大, 得到scores & labels, 每个point只预测一个目标
        pred_score, pred_label = torch.max(heatmap_pred, dim=0)  # [output_h, output_w]
        pred_score = pred_score.reshape(-1)
        pred_label = pred_label.reshape(-1)

        # 生成位置坐标
        coord_xs = torch.arange(0, output_w, 1, dtype=torch.float32, device=heatmap_pred.device)
        coord_ys = torch.arange(0, output_h, 1, dtype=torch.float32, device=heatmap_pred.device)
        grid_ys, grid_xs = torch.meshgrid(coord_ys, coord_xs)
        grid_coords = torch.stack((grid_xs, grid_ys), dim=0)  # [2, h, w]

        # 大于阈值的mask
        keep_mask = pred_score >= self.score_thrs   # [-1]

        # 根据mask取出对应目标
        pred_score = pred_score[keep_mask]
        pred_label = pred_label[keep_mask]

        wh_pred = single_preds['wh']
        wh_pred = wh_pred.permute(1, 2, 0).reshape(-1, 2)  # [-1, 2]
        wh_pred = wh_pred[keep_mask]

        offset_pred = single_preds['offset']
        offset_pred = offset_pred.permute(1, 2, 0).reshape(-1, 2)  # [-1, 2]
        offset_pred = offset_pred[keep_mask]

        grid_coords = grid_coords.permute(1, 2, 0).reshape(-1, 2)  # [-1, 2]
        grid_coords = grid_coords[keep_mask]

        # pred bbox
        grid_coords = grid_coords + offset_pred
        pred_bbox = grid_coords.new_zeros([grid_coords.size(0), 4])
        pred_bbox[:, 0] = (grid_coords[:, 0] - wh_pred[:, 0] / 2) * downsample_scale_w
        pred_bbox[:, 1] = (grid_coords[:, 1] - wh_pred[:, 1] / 2) * downsample_scale_h
        pred_bbox[:, 2] = (grid_coords[:, 0] + wh_pred[:, 0] / 2) * downsample_scale_w
        pred_bbox[:, 3] = (grid_coords[:, 1] + wh_pred[:, 1] / 2) * downsample_scale_h

        # rescale
        if rescale:
            scale_factor_w = metainfo['scale_factors'][0]
            scale_factor_h = metainfo['scale_factors'][1]
            
            pred_bbox[:, [0, 2]] = pred_bbox[:, [0, 2]] / scale_factor_w
            pred_bbox[:, [1, 3]] = pred_bbox[:, [1, 3]] / scale_factor_h

        result = data_sample.new()
        pred_instances = InstanceData()
        pred_instances.bboxes = pred_bbox
        pred_instances.scores = pred_score
        pred_instances.labels = pred_label

        if to_numpy:
            pred_instances = pred_instances.numpy()

        result.pred_instances = pred_instances

        return result