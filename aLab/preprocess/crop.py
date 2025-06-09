# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import cv2
import numpy as np

from typing import (Union, Tuple, List)

from aLab.register import PREPROCESS

from .utils import GTKeys


ROOT = os.path.dirname(__file__)


__all__ = ['RandomCrop']


@PREPROCESS.register_module()
class RandomCrop:
    def __init__(self,
                 crop_scales: Union[float, List[float], Tuple[float]] = (0.6, 0.9),
                 keep_aspect: bool = True,
                 center_crop: bool = False,
                 max_aspect: float = 2.0,
                 prob: float = 0.5,
                 debug: bool = False):
        super().__init__()

        self.prob = prob
        self.debug = debug
        self.max_aspect = max_aspect
        self.center_crop = center_crop
        self.keep_aspect = keep_aspect
        if isinstance(crop_scales, float):
            self.crop_scales = [crop_scales, crop_scales]
        else:
            self.crop_scales = crop_scales

        if not self.keep_aspect:
            assert self.max_aspect >= 1.0, ValueError(f'When "keep_aspect=False", the "max_aspect" of the crop area is >= 1, but got {self.max_aspect}.')

    def _get_cropped_coords(self, ori_size: Tuple[int, int], crop_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        crop_h, crop_w = crop_size
        ori_img_h, ori_img_w = ori_size
        
        if self.center_crop:
            ctx = ori_img_w / 2
            cty = ori_img_h / 2
            x1 = int(ctx - crop_w / 2)
            y1 = int(cty - crop_h / 2)
            x2 = int(ctx + crop_w / 2)
            y2 = int(cty + crop_h / 2)
        else:
            if crop_w == ori_img_w:
                x1 = 0
            else:
                x1 = np.random.randint(0, ori_img_w - crop_w)
            
            if crop_h == ori_img_h:
                y1 = 0
            else:
                y1 = np.random.randint(0, ori_img_h - crop_h)
            x2 = x1 + crop_w
            y2 = y1 + crop_h
        
        return (x1, y1, x2, y2)

    def _crop_gts(self, data: dict, cropped_size: Tuple[int, int], cropped_coords: Tuple[int, int, int, int]) -> dict:
        cropped_h, cropped_w = cropped_size
        (cropped_x1, cropped_y1, cropped_x2, cropped_y2) = cropped_coords

        # 保存处理后的gts
        gt_processed = dict()

        for (primary_key, other_keys) in GTKeys.items():
            if primary_key in data:
                # 复制一份bbox
                primary_gt = data[primary_key].copy()

                # 根据crop坐标调整bbox
                primary_gt[:, 0::2] = np.clip(primary_gt[:, 0::2] - cropped_x1, a_min=0, a_max=cropped_w)
                primary_gt[:, 1::2] = np.clip(primary_gt[:, 1::2] - cropped_y1, a_min=0, a_max=cropped_h)

                # 判断裁剪后有效bbox
                temp_bbox_ws = primary_gt[:, 2] - primary_gt[:, 0]
                temp_bbox_hs = primary_gt[:, 3] - primary_gt[:, 1]
                keep = (temp_bbox_ws > 0) & (temp_bbox_hs > 0)

                # 获取裁剪后的bbox
                if sum(keep):
                    cropped_bboxes = primary_gt[keep]
                    gt_processed[primary_key] = cropped_bboxes

                    # 更新labels
                    for other_key in other_keys:
                        if (other_key is not None) and (other_key in data):
                            temp_labels = data[other_key].copy()
                            cropped_labels = temp_labels[keep]

                            gt_processed[other_key] = cropped_labels

        return gt_processed

    def __call__(self, data: dict) -> dict:
        if data.get('img', None) is None:
            return None
        
        data['cropped'] = False
        if np.random.random() < self.prob:
            ori_img_h, ori_img_w = data['img'].shape[:-1]

            # 计算crop后的宽高比
            if self.keep_aspect:
                crop_aspect = ori_img_w / ori_img_h
            else:
                crop_aspect = np.random.uniform(1.0, self.max_aspect)
            
            # 获得裁剪比例
            crop_scale = np.random.uniform(self.crop_scales[0], self.crop_scales[1])

            # 获得裁剪尺寸
            crop_w = int(ori_img_w * crop_scale)
            crop_h = int(crop_w / crop_aspect)

            # 确保裁剪区域不会超过图像尺寸
            crop_h = min(crop_h, ori_img_h)
            crop_w = min(crop_w, ori_img_w)

            # 获取cropped坐标
            cropped_coords = self._get_cropped_coords(ori_size=(ori_img_h, ori_img_w), crop_size=(crop_h, crop_w))
            cropped_x1, cropped_y1, cropped_x2, cropped_y2 = cropped_coords

            # 裁剪图像
            cropped_image = data['img'][cropped_y1: cropped_y2, cropped_x1: cropped_x2]

            # 将边界框坐标从绝对像素值转换为相对于裁剪区域的坐标
            gt_processed = self._crop_gts(data=data, cropped_size=(crop_h, crop_w), cropped_coords=cropped_coords)
            if len(gt_processed):
                data.update(gt_processed)
                data['img'] = cropped_image

                data['cropped'] = True

        # debug
        if self.debug:
            self._debug(data)
        
        return data
    
    def __repr__(self) -> str:
        repr_str = f'self.__class__.__name__('
        repr_str += f'(crop_scales=[{self.crop_scales}], '
        repr_str += f'keep_aspect={self.keep_aspect}, '
        repr_str += f'center_crop={self.center_crop}, '
        repr_str += f'max_aspect={self.max_aspect}, '
        repr_str += f'prob={self.prob}'
        repr_str += ')'

        return repr_str
    
    def _debug(self, data: dict) -> None:
        img = data['img'].copy()

        if data.get('to_rgb', False):   # reverse
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if data.get('to_float32', True):
            img = img.astype(np.uint8)

        if 'gt_bboxes' in data:
            for bbox in data['gt_bboxes'].copy():
                x1, y1, x2, y2 = [int(x) for x in bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        debug_dir = os.path.join(ROOT, 'debug', self.__class__.__name__)
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir, exist_ok=True)

        cv2.imwrite(os.path.join(debug_dir, os.path.basename(data['filename'])), img)
