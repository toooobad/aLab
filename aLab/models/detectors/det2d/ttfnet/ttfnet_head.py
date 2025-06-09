# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import (Tuple, List, Union)

from aLab.utils import multi_apply
from aLab.register import (HEAD, POSTPROCESS)
from aLab.structures import (DetDataSample, InstanceData)
from aLab.models.utils import (bias_init_with_prob, normal_init)

from ..centernet import CenterNetPostprocess
from ..anchorfree_head2d import AnchorFreeHead2D
from .ttfnet_utils import draw_truncate_gaussian


__all__ = ['TTFNetHead', 'TTFNetPostprocess']


@HEAD.register_module()
class TTFNetHead(AnchorFreeHead2D):
    """
    Training-Time-Friendly Network for Real-Time Object Detection
    code: <https://github.com/ZJULearning/ttfnet>
    paper: <http://arxiv.org/abs/1909.00700>
    """
    def __init__(self,
                 num_classes: int = 80,
                 in_channels: int = 128,
                 beta: float = 0.54,
                 alpha: float = 0.54,
                 base_bbox_size: float = 16.,
                 bbox_area_process: str = 'log',
                 num_bbox_head_stacks: int = 2,
                 num_heatmap_head_stacks: int = 2,
                 bbox_feat_channels: int = 64,
                 heatmap_feat_channels: int = 256,
                 loss_cfgs: dict = dict(
                    bbox=dict(type='GIoULoss', loss_weight=5.0),
                    heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0)),
                 initial_cfgs: dict = dict(use_bg_cls=False, score_thrs=0.01)) -> None:
        self.beta = beta
        self.alpha = alpha

        self.base_bbox_size = base_bbox_size
        self.bbox_area_process = bbox_area_process

        # head stacks
        self.num_bbox_head_stacks = num_bbox_head_stacks
        self.num_heatmap_head_stacks = num_heatmap_head_stacks

        # head channels
        self.bbox_feat_channels = bbox_feat_channels
        self.heatmap_feat_channels = heatmap_feat_channels

        self.base_location = None
        
        super(TTFNetHead, self).__init__(loss_cfgs=loss_cfgs, 
                                            num_classes=num_classes, 
                                            in_channels=in_channels, 
                                            initial_cfgs=initial_cfgs)
        # test params
        self.local_maximum_kernel = 3

        if self.initial_cfgs is not None:
            self.local_maximum_kernel = self.initial_cfgs.get('local_maximum_kernel', 3)
            
        self.local_maximum_padding = (self.local_maximum_kernel - 1) // 2

        # postprocess
        self.postprocesser = TTFNetPostprocess(self.local_maximum_kernel, self.local_maximum_padding, score_thrs=self.score_thrs)
    
    def _init_heads(self) -> None:
        # bbox head
        bbox_head = []
        for i in range(self.num_bbox_head_stacks):
            inp = self.in_channels if i == 0 else self.bbox_feat_channels
            bbox_head.extend([nn.Conv2d(inp, self.bbox_feat_channels, kernel_size=3, stride=1, padding=1),
                              nn.ReLU(inplace=True)])

        bbox_head.append(nn.Conv2d(self.bbox_feat_channels, 4, 1))
        self.bbox_head = nn.Sequential(*bbox_head)

        # heatmap head
        heatmap_head = []
        for i in range(self.num_heatmap_head_stacks):
            inp = self.in_channels if i == 0 else self.heatmap_feat_channels
            heatmap_head.extend([nn.Conv2d(inp, self.heatmap_feat_channels, kernel_size=3, stride=1, padding=1),
                                 nn.ReLU(inplace=True)])

        heatmap_head.append(nn.Conv2d(self.heatmap_feat_channels, self.num_classes, 1))
        self.heatmap_head = nn.Sequential(*heatmap_head)

    def _init_weights(self) -> None:
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.01)
        normal_init(self.heatmap_head[-1], std=0.01, bias=bias_init)

        for _, bbox_module in self.bbox_head.named_modules():
            if isinstance(bbox_module, nn.Conv2d):
                normal_init(bbox_module, std=0.001)
    
    @staticmethod
    def _clip_sigmoid(x: Tensor) -> Tensor:
        return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    
    def forward(self, img_feats: Union[Tuple[Tensor], Tensor], training: bool = True) -> dict:
        if isinstance(img_feats, Tensor):
            img_feat = img_feats
        else:
            img_feat = img_feats[0]

        bbox_preds = self.bbox_head(img_feat)
        bbox_preds = F.relu(bbox_preds) * self.base_bbox_size
        heatmap_preds = self.heatmap_head(img_feat)

        preds = dict(bboxes=bbox_preds)

        if not training:
            heatmap_preds = heatmap_preds.sigmoid()
            local_maximum_heatmaps = F.max_pool2d(heatmap_preds, self.local_maximum_kernel, stride=1, padding=self.local_maximum_padding)
            preds['local_maximum_heatmap'] = local_maximum_heatmaps

        preds['heatmap'] = heatmap_preds

        return preds
    
    def forward_deploy(self, img_feats: Union[Tuple[Tensor], Tensor]) -> Tuple[Tensor]:
        img_feat = img_feats[0]

        bbox_preds = self.bbox_head(img_feat)
        bbox_preds = F.relu(bbox_preds) * self.base_bbox_size
        heatmap_preds = self.heatmap_head(img_feat).sigmoid()
        
        local_maximum_heatmaps = F.max_pool2d(heatmap_preds, self.local_maximum_kernel, stride=1, padding=self.local_maximum_padding)

        return heatmap_preds, bbox_preds, local_maximum_heatmaps
    
    # ------------------------- losses -------------------------
    def get_losses(self, batch_preds: dict, batch_gt_instances: List[InstanceData], batch_metainfos: List[dict]) -> dict:
        heatmap_preds = self._clip_sigmoid(batch_preds['heatmap'])
        
        assert heatmap_preds.size(0) == len(batch_metainfos) == len(batch_gt_instances)
        output_h, output_w = heatmap_preds.shape[-2:]

        # targets
        batch_targets = self._get_targets(batch_gt_instances, (output_h, output_w), batch_metainfos)

        # losses
        loss_heatmap = self.loss_heatmap(preds=heatmap_preds, 
                                         targets=batch_targets['heatmap'])
        
        if (self.base_location is None) or (output_h != self.base_location.shape[1]) or (output_w != self.base_location.shape[0]):
            input_size = batch_metainfos[0]['input_size']
            base_step_h = input_size[0] // output_h
            base_step_w = input_size[1] // output_w

            shifts_x = torch.arange(0, 
                                    input_size[1] - 1, 
                                    base_step_w,
                                    dtype=torch.float32, device=heatmap_preds.device)
            shifts_y = torch.arange(0, 
                                    input_size[0] - 1, 
                                    base_step_h,
                                    dtype=torch.float32, device=heatmap_preds.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_location = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)
        
        base_location = self.base_location.clone().unsqueeze(0)
        bbox_preds = torch.cat((base_location - batch_preds['bboxes'][:, [0, 1]],
                                base_location + batch_preds['bboxes'][:, [2, 3]]), dim=1)  # [bs, 4, h, w]

        bbox_preds = bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4)  # [bs, h, w, 4] -> [bs*h*w, 4]
        bbox_targets = batch_targets['bboxes'].permute(0, 2, 3, 1).reshape(-1, 4)  # [bs*h*w, 4]
        bbox_target_weights = batch_targets['bbox_weights'].permute(0, 2, 3, 1).reshape(-1, 1)  # [bs*h*w, 1]

        reg_avg_factor = bbox_target_weights.sum() + 1e-4

        loss_bbox = self.loss_bbox(bbox_preds, bbox_targets, bbox_target_weights, avg_factor=reg_avg_factor)
                                         
        return {'loss_heatmap': loss_heatmap, 'loss_bbox': loss_bbox}

    def _get_targets(self, batch_gt_instances: List[InstanceData], output_size: Tuple[int, int], batch_metainfos: List[dict]) -> dict:
        batch_targets_list = multi_apply(self._get_single_targets, batch_gt_instances, batch_metainfos, output_size=output_size)
        
        (heatmap, bbox_target, bbox_target_weight) = batch_targets_list
        (heatmap, bbox_target, bbox_target_weight) = [torch.stack(t, dim=0).detach() for t in [heatmap, bbox_target, bbox_target_weight]]
    
        batch_targets = dict(
                heatmap=heatmap,
                bboxes=bbox_target,
                bbox_weights=bbox_target_weight,
                avg_factor=max(1, heatmap.eq(1).sum()))

        return batch_targets
    
    def _get_bbox_areas(self, bboxes: Tensor) -> Tensor:
        bbox_areas = (bboxes[:, 3] - bboxes[:, 1] + 1) * (bboxes[:, 2] - bboxes[:, 0] + 1)

        if self.bbox_area_process == 'log':
            bbox_areas = bbox_areas.log()
        elif self.bbox_area_process == 'sqrt':
            bbox_areas = bbox_areas.sqrt()

        topk_bbox_areas, topk_bbox_inds = torch.topk(bbox_areas, bbox_areas.size(0))

        if self.bbox_area_process == 'norm':
            topk_bbox_areas[:] = 1.
        
        return topk_bbox_areas, topk_bbox_inds

    def _get_single_targets(self, gt_instances: InstanceData, metainfo: dict, output_size: Tuple[int]) -> Tuple[Tensor]:
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        num_objs = gt_bboxes.size(0)

        output_h, output_w = output_size

        # init targets
        heatmap = gt_bboxes.new_zeros((self.num_classes, output_h, output_w))
        fake_heatmap = gt_bboxes.new_zeros((output_h, output_w))

        bbox_target = gt_bboxes.new_zeros((4, output_h, output_w))
        bbox_target_weight = gt_bboxes.new_zeros((1, output_h, output_w))

        # bbox areas
        topk_bbox_areas, topk_bbox_inds = self._get_bbox_areas(gt_bboxes)   # input-size

        # sort
        gt_bboxes = gt_bboxes[topk_bbox_inds]
        gt_labels = gt_labels[topk_bbox_inds]

        # down sample
        input_h, input_w = metainfo['input_size']
        down_sample_w = input_w // output_w
        down_sample_h = input_h // output_h

        output_bboxes = gt_bboxes.clone()
        output_bboxes[:, [0, 2]] = output_bboxes[:, [0, 2]] / down_sample_w
        output_bboxes[:, [1, 3]] = output_bboxes[:, [1, 3]] / down_sample_h
        output_bboxes[:, [0, 2]] = torch.clamp(output_bboxes[:, [0, 2]], min=0, max=output_w - 1)
        output_bboxes[:, [1, 3]] = torch.clamp(output_bboxes[:, [1, 3]], min=0, max=output_h - 1)
        output_hs, output_ws = (output_bboxes[:, 3] - output_bboxes[:, 1],
                                output_bboxes[:, 2] - output_bboxes[:, 0])
        
        # we calc the center and ignore area based on the gt-boxes of the origin scale no peak will fall between pixels
        output_ctxs = (output_bboxes[:, 0] + output_bboxes[:, 2]) / 2
        output_ctys = (output_bboxes[:, 1] + output_bboxes[:, 3]) / 2
        output_cts_int = torch.stack([output_ctxs, output_ctys], dim=1).to(torch.int)

        # gaussian radius
        h_radiuses_alpha = (output_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (output_ws / 2. * self.alpha).int()
        if self.alpha != self.beta:
            h_radiuses_beta = (output_hs / 2. * self.beta).int()
            w_radiuses_beta = (output_ws / 2. * self.beta).int()
        
        for obj_id in range(num_objs):
            cls_id = gt_labels[obj_id]

            # heatmap
            fake_heatmap = fake_heatmap.zero_()
            draw_truncate_gaussian(fake_heatmap, 
                                   output_cts_int[obj_id],
                                   h_radiuses_alpha[obj_id].item(), 
                                   w_radiuses_alpha[obj_id].item())
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)

            # bboxes
            if self.alpha != self.beta:
                fake_heatmap = fake_heatmap.zero_()
                draw_truncate_gaussian(fake_heatmap,
                                       output_cts_int[obj_id],
                                       h_radiuses_beta[obj_id].item(),
                                       w_radiuses_beta[obj_id].item())
            bbox_target_inds = fake_heatmap > 0

            bbox_target[:, bbox_target_inds] = gt_bboxes[obj_id][:, None]

            local_heatmap = fake_heatmap[bbox_target_inds]
            ct_div = local_heatmap.sum()
            local_heatmap *= topk_bbox_areas[obj_id]
            bbox_target_weight[0, bbox_target_inds] = local_heatmap / ct_div
        
        return (heatmap, bbox_target, bbox_target_weight)


@POSTPROCESS.register_module()
class TTFNetPostprocess(CenterNetPostprocess):
    def _process_single_sample(self, single_preds: dict, data_sample: DetDataSample, rescale: bool = False, to_numpy: bool = True) -> DetDataSample:
        """Transform outputs of a single image into bbox results.

        Args:
            heatmap_pred (Tensor): Center heatmap for current level with shape (num_classes, output_h, output_w).
            bbox_pred (Tensor): WH heatmap for current level with shape (4, output_h, output_w).
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
        coord_xs = torch.arange(0, 
                                metainfo['input_size'][1] - 1, 
                                downsample_scale_w,
                                dtype=torch.float32, device=heatmap_pred.device)
        coord_ys = torch.arange(0, 
                                metainfo['input_size'][0] - 1, 
                                downsample_scale_h,
                                dtype=torch.float32, device=heatmap_pred.device)
        grid_ys, grid_xs = torch.meshgrid(coord_ys, coord_xs)
        grid_coords = torch.stack((grid_xs, grid_ys), dim=0)  # [2, h, w]
        grid_coords = grid_coords.permute(1, 2, 0).reshape(-1, 2)  # [hw, 2]

        # 大于阈值的mask
        keep_mask = pred_score >= self.score_thrs   # [-1]

        # 根据mask取出对应目标
        pred_score = pred_score[keep_mask]
        pred_label = pred_label[keep_mask]

        # pred bbox
        bbox_pred = single_preds['bboxes'].permute(1, 2, 0).reshape(-1, 4)  # [hw, 4]
        pred_bbox = torch.cat((grid_coords - bbox_pred[:, [0, 1]],
                               grid_coords + bbox_pred[:, [2, 3]]), dim=-1)
        pred_bbox = pred_bbox[keep_mask]

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