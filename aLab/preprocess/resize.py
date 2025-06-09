# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import cv2
import numpy as np

from typing import (Union, Tuple, Any)

from aLab.register import PREPROCESS
from aLab.utils import get_homography_matrix

from .utils import GTKeys


ROOT = os.path.dirname(__file__)


__all__ = ['ImageResize', 'TLFTRImageResize']


CV2_INTERP_CODES = {
    'area': cv2.INTER_AREA,
    'bicubic': cv2.INTER_CUBIC,
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'lanczos': cv2.INTER_LANCZOS4,
}


@PREPROCESS.register_module()
class ImageResize:
    def __init__(self,
                 debug: bool = False,
                 padding: bool = False,
                 keep_aspect: bool = True,
                 random_interp: bool = False,
                 pad_value: Tuple[int, int, int] = (0, 0, 0),
                 input_size: Union[tuple, int] = (512, 512)):
        super().__init__()

        self.debug = debug
        self.padding = padding
        self.pad_value = pad_value
        self.keep_aspect = keep_aspect
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.input_size = self._check_input_size()

        if random_interp:
            self.interp_methods = ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos']
        else:
            self.interp_methods = ['bilinear']
    
    def _check_input_size(self) -> Tuple[int, int]:
        """
        保证尺寸缩放后能被32整除
        """
        input_h, input_w = self.input_size

        if input_h % 32 != 0:  # 不能被32整除
            input_h = int((input_h // 32 + 1) * 32)

        if input_w % 32 != 0:  # 不能被32整除
            input_w = int((input_w // 32 + 1) * 32)
        
        return (input_h, input_w)

    def _random_interp_method(self) -> Any:
        """Randomly select an scale from given candidates.

        Returns:
            (tuple, int): interp methods
        """

        index = np.random.randint(len(self.interp_methods))
        
        return self.interp_methods[index]
    
    def _resize_gts(self, data: dict) -> dict:
        scale_factors = data['scale_factors']

        # 保存处理后的gts
        gt_processed = dict()

        for (primary_key, other_keys) in GTKeys.items():
            if primary_key in data:
                # copy
                primary_gt = data[primary_key].copy()

                # resize
                primary_gt[:, 0::2] *= scale_factors[0]
                primary_gt[:, 1::2] *= scale_factors[1]

                gt_processed[primary_key] = primary_gt
        
        return gt_processed

    def __call__(self, data: dict) -> dict:
        ori_img = data.get('img', None)
        if ori_img is None:
            return None
        
        ori_h, ori_w = ori_img.shape[:-1]

        # 计算宽和高的缩放比例
        if self.keep_aspect:
            scale_h = scale_w = min(self.input_size[0] / ori_h, self.input_size[1] / ori_w)
        else:
            scale_h = self.input_size[0] / ori_h
            scale_w = self.input_size[1] / ori_w

        # 插值方式
        interp_method = self._random_interp_method()

        # resize
        ressized_img = cv2.resize(ori_img, 
                                  (int(ori_w * scale_w), int(ori_h * scale_h)),
                                  interpolation=CV2_INTERP_CODES[interp_method])

        data['interp_method'] = interp_method
        data['resize_shape'] = ressized_img.shape[:-1]
        data['scale_factors'] = (scale_w, scale_h, scale_w, scale_h)

        # padding
        if self.padding:
            resized_h, resized_w = data['resize_shape']
            pad_h = max(self.input_size[0] - resized_h, 0)
            pad_w = max(self.input_size[1] - resized_w, 0)

            # 右下角填充
            ressized_img = cv2.copyMakeBorder(ressized_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.pad_value)

        data['img'] = ressized_img
        data['input_size'] = self.input_size
        
        # resize gts
        gt_processed = self._resize_gts(data)
        if len(gt_processed):
            data.update(gt_processed)

        # debug
        if self.debug:
            self._debug(data)
        
        return data
    
    def __repr__(self) -> str:
        repr_str = f'self.__class__.__name__('
        repr_str += f'(input_size=[{self.input_size}], '
        repr_str += f'keep_aspect={self.keep_aspect}, '
        repr_str += f'interp_method={self.interp_methods} [{CV2_INTERP_CODES}], '
        repr_str += f'padding={self.padding}, '
        repr_str += f'pad_value={self.pad_value}'
        repr_str += ')'

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


@PREPROCESS.register_module()
class TLFTRImageResize(ImageResize):
    def _resize_gts(self, data: dict) -> dict:
        scale_factors = data['scale_factors']

        # 保存处理后的gts
        gt_processed = dict()

        for (primary_key, other_keys) in GTKeys.items():
            if '30' in primary_key:
                scale_factor = scale_factors[0]
            else:
                scale_factor = scale_factors[1]

            if primary_key in data:
                # copy
                primary_gt = data[primary_key].copy()

                # resize
                primary_gt[:, 0::2] *= scale_factor[0]
                primary_gt[:, 1::2] *= scale_factor[1]

                gt_processed[primary_key] = primary_gt
        
        return gt_processed

    def __call__(self, data: dict) -> dict:
        ori_imgs = data.get('img', None)
        if ori_imgs is None:
            return None
        
        # 插值方式
        interp_method = self._random_interp_method()
        data['interp_method'] = interp_method

        imgs = []
        resize_shapes = []
        scale_factors = []

        for ori_img in ori_imgs:
            ori_h, ori_w = ori_img.shape[:-1]

            # 计算宽和高的缩放比例
            if self.keep_aspect:
                scale_h = scale_w = min(self.input_size[0] / ori_h, self.input_size[1] / ori_w)
            else:
                scale_h = self.input_size[0] / ori_h
                scale_w = self.input_size[1] / ori_w

            # resize
            ressized_img = cv2.resize(ori_img, 
                                      (int(ori_w * scale_w), int(ori_h * scale_h)),
                                      interpolation=CV2_INTERP_CODES[interp_method])
            resize_shape = ressized_img.shape[:-1]
            resize_shapes.append(resize_shape)
            scale_factors.append((scale_w, scale_h, scale_w, scale_h))
            
            # padding
            if self.padding:
                resized_h, resized_w = resize_shape
                pad_h = max(self.input_size[0] - resized_h, 0)
                pad_w = max(self.input_size[1] - resized_w, 0)

                # 右下角填充
                ressized_img = cv2.copyMakeBorder(ressized_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.pad_value)

            imgs.append(ressized_img)
        
        data['img'] = imgs
        data['input_size'] = self.input_size
        data['resize_shape'] = resize_shapes
        data['scale_factors'] = scale_factors

        # resize gts
        gt_processed = self._resize_gts(data)
        if len(gt_processed):
            data.update(gt_processed)
        
        # calibs
        scale_factors30, scale_factors120 = scale_factors
        intrinsic30 = data['calibs']['intrinsic30']
        intrinsic30[0, 0] = intrinsic30[0, 0] * scale_factors30[0]
        intrinsic30[1, 1] = intrinsic30[1, 1] * scale_factors30[1]
        intrinsic30[0, 2] = intrinsic30[0, 2] * scale_factors30[0] + scale_factors30[0] * 0.5 - 0.5
        intrinsic30[1, 2] = intrinsic30[1, 2] * scale_factors30[1] + scale_factors30[1] * 0.5 - 0.5
        data['calibs']['intrinsic30'] = intrinsic30

        intrinsic120 = data['calibs']['intrinsic120']
        intrinsic120[0, 0] = intrinsic120[0, 0] * scale_factors120[0]
        intrinsic120[1, 1] = intrinsic120[1, 1] * scale_factors120[1]
        intrinsic120[0, 2] = intrinsic120[0, 2] * scale_factors120[0] + scale_factors120[0] * 0.5 - 0.5
        intrinsic120[1, 2] = intrinsic120[1, 2] * scale_factors120[1] + scale_factors120[1] * 0.5 - 0.5
        data['calibs']['intrinsic120'] = intrinsic120

        data['M'] = get_homography_matrix(
            src_intrinsic=data['calibs']['intrinsic30'],
            src_extrinsic=data['calibs']['extrinsic30'],
            dst_intrinsic=data['calibs']['intrinsic120'],
            dst_extrinsic=data['calibs']['extrinsic120'],
            src_car_center=data['calibs']['car_center30'],
            dst_car_center=data['calibs']['car_center120'])

        # debug
        if self.debug:
            self._debug(data)
        
        return data
    
    def _debug(self, data: dict) -> None:
        imgs = data['img']

        for i, img in enumerate(imgs):
            if data.get('to_rgb', False):   # reverse
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if data.get('to_float32', True):
                img = img.astype(np.uint8)
            
            if i == 0: # img30
                gt_bboxes = data['gt_bboxes30']
                gt_labels = data['gt_labels30']
            else:
                gt_bboxes = data['gt_bboxes120']
                gt_labels = data['gt_labels120']
            
            for bbox, label in zip(gt_bboxes, gt_labels):
                x1, y1, x2, y2 = [int(x) for x in bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, str(label), (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 1, cv2.LINE_AA)
            
            debug_dir = os.path.join(ROOT, 'debug', self.__class__.__name__, f'{"fov30" if i == 0 else "fov120"}')
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir, exist_ok=True)
            
            filename = data['filename'][i]
            if filename is None:
                filename = data['filename'][1 if i == 0 else 0]

            cv2.imwrite(os.path.join(debug_dir, os.path.basename(filename)), img)
        
        # fusion
        img = imgs[-1]
        if data.get('to_rgb', False):   # reverse
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if data.get('to_float32', True):
            img = img.astype(np.uint8)
        
        gt_bboxes = data['gt_bboxes']
        gt_labels = data['gt_labels']

        for bbox, label in zip(gt_bboxes, gt_labels):
            x1, y1, x2, y2 = [int(x) for x in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, str(label), (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 1, cv2.LINE_AA)
        
        debug_dir = os.path.join(ROOT, 'debug', self.__class__.__name__, 'fusion')
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir, exist_ok=True)
        
        filename = data['filename'][1]
        if filename is None:
            filename = data['filename'][0]

        cv2.imwrite(os.path.join(debug_dir, os.path.basename(filename)), img)