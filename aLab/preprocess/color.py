# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import cv2
import numpy as np

from typing import (Union, Sequence)

from aLab.register import PREPROCESS


ROOT = os.path.dirname(__file__)


__all__ = ['PhotoMetricDistortion']


@PREPROCESS.register_module()
class PhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 hue_delta: float = 18,
                 brightness_delta: float = 32,
                 contrast_range: Union[list, tuple] = (0.5, 1.5),
                 saturation_range: Union[list, tuple] = (0.5, 1.5)):
        self.hue_delta = hue_delta
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
    
    def _apply_hsv_augment(self, img: np.ndarray, is_rgb: bool = False) -> np.ndarray:
        img = img.astype(np.float32)

        # random brightness
        if np.random.randint(2):
            delta = np.random.uniform(-self.brightness_delta, self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = np.random.uniform(self.contrast_lower, 
                                          self.contrast_upper)
                img *= alpha

        # convert color from BGR/RGB to HSV
        if is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # random saturation
        if np.random.randint(2):
            img[..., 1] *= np.random.uniform(self.saturation_lower,
                                             self.saturation_upper)

        # random hue
        if np.random.randint(2):
            img[..., 0] += np.random.uniform(-self.hue_delta, 
                                             self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        if is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = np.random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]
        
        return img

    def __call__(self, data: dict) -> dict:
        """Call function to perform photometric distortion on images.

        Args:
            data (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        ori_imgs = data.get('img', None)
        if ori_imgs is None:
            return None
        
        if isinstance(ori_imgs, Sequence):
            new_imgs = []
            for ori_img in ori_imgs:
                new_img = self._apply_hsv_augment(ori_img, is_rgb=data.get('to_rgb', False))
            
                new_imgs.append(new_img)
            data['img'] = new_imgs
        else:
            data['img'] = self._apply_hsv_augment(ori_imgs, is_rgb=data.get('to_rgb', False))

        return data
