# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import cv2
import numpy as np

from typing import (Sequence, Union)

from aLab.register import PREPROCESS


__all__ = ['Normalize']


@PREPROCESS.register_module()
class Normalize:
    def __init__(self, mean: Sequence[Union[int, float]], std: Sequence[Union[int, float]]) -> None:
        
        self.std = np.array(std, dtype=np.float32)
        self.mean = np.array(mean, dtype=np.float32)
    
    def __call__(self, data: dict) -> dict:
        imgs = data.get('img', None)
        if imgs is None:
            return None

        mean = np.float64(self.mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.std.reshape(1, -1))
        
        if isinstance(imgs, Sequence):
            new_imgs = []
            for img in imgs:
                if img.dtype != np.float32:
                    img = img.astype(np.float32)
                    
                cv2.subtract(img, mean, img)  # inplace
                cv2.multiply(img, stdinv, img)  # inplace

                new_imgs.append(img)
            
            data['img'] = new_imgs
        else:
            if imgs.dtype != np.float32:
                imgs = imgs.astype(np.float32)

            cv2.subtract(imgs, mean, imgs)  # inplace
            cv2.multiply(imgs, stdinv, imgs)  # inplace

            data['img'] = imgs

        data['img_norm_cfg'] = dict(mean=self.mean, std=self.std)

        return data
    
    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}('
        repr_str += f'mean={self.mean}, '
        repr_str += f'std={self.std}, '
        repr_str += ')'
        return repr_str