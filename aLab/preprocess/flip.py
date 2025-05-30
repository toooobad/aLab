# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import cv2
import numpy as np

from aLab.register import PREPROCESS

from .utils import GTKeys


ROOT = os.path.dirname(__file__)


__all__ = ['HorizontalFlip']


@PREPROCESS.register_module()
class HorizontalFlip:
    def __init__(self, prob: float = 0.5, debug: bool = False):
        super().__init__()

        self.prob = prob
        self.debug = debug
    
    def _filp_gts(self, data: dict, ori_img_w: int) -> dict:
        # 保存处理后的gts
        gt_processed = dict()

        for (primary_key, other_keys) in GTKeys.items():
            if primary_key in data:
                # 复制一份bbox
                primary_gt = data[primary_key].copy()

                # flip
                primary_gt[:, [0, 2]] = ori_img_w - primary_gt[:, [2, 0]] - 1

                gt_processed[primary_key] = primary_gt
        
        return gt_processed
    
    def __call__(self, data: dict) -> dict:
        ori_img = data.get('img', None)
        if ori_img is None:
            return None
        
        data['flipped'] = False

        if np.random.random() < self.prob:
            flipped_img = ori_img.copy()
            flipped_img = np.fliplr(flipped_img)
            data['img'] = flipped_img

            # 处理gts
            gt_processed = self._filp_gts(data, ori_img_w=flipped_img.shape[1])
            if len(gt_processed):
                data.update(gt_processed)
            
            data['flipped'] = True
        
        if self.debug:
            self._debug(data)
        
        return data
    
    def __repr__(self) -> str:
        repr_str = f'self.__class__.__name__'
        repr_str += f'(prob=[{self.prob}])'

        return repr_str
    
    def _debug(self, data: dict) -> None:
        img = data['img'].copy()

        if data.get('to_rgb', False):   # reverse
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if data.get('to_float32', True):
            img = img.astype(np.uint8)

        for bbox in data['gt_bboxes'].copy():
            x1, y1, x2, y2 = [int(x) for x in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        debug_dir = os.path.join(ROOT, 'debug', self.__class__.__name__)
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir, exist_ok=True)

        cv2.imwrite(os.path.join(debug_dir, os.path.basename(data['filename'])), img)
