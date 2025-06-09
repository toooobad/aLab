# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import cv2
import numpy as np

from copy import deepcopy
from typing import (Tuple, )

from aLab.register import PREPROCESS
from aLab.utils import get_homography_matrix


ROOT = os.path.dirname(__file__)


__all__ = ['LoadImageFromFile', 'TLFTRLoadMultiViewImageFromFile']


@PREPROCESS.register_module()
class LoadImageFromFile:
    def __init__(self, 
                 backend: str = 'cv2', 
                 to_rgb: bool = False, 
                 conf_file: str = None,
                 to_float32: bool = True,
                 debug: bool = False):

        self.debug = debug
        self.to_rgb = to_rgb
        self.backend = backend
        self.conf_file = conf_file
        self.to_float32 = to_float32

        if self.backend == 'petrel':
            assert self.conf_file is not None
            try:
                from aoss_client.client import Client
            except:
                from petrel_client.client import Client
            
            self.mclient = Client(conf_path=self.conf_file)
    
    def _load_img(self, filename: str) -> np.ndarray:
        try:
            if self.backend == 'petrel':
                img_bytes = self.mclient.get(filename)
                if img_bytes is None:
                    return None
                
                img_mem_view = memoryview(img_bytes)
                img_array = np.frombuffer(img_mem_view, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)   # bgr
            else:
                img = cv2.imread(filename)
        except:
            return None
        
        return np.array(img)

    def __call__(self, data: dict) -> dict:
        filename = data.get('filename', None)
        if filename is None:
            return None
        
        # load
        img = self._load_img(filename)
        if img is None:
            return None
        
        # to rgb
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # to float32
        if self.to_float32:
            img = img.astype(np.float32)
            
        data['img'] = img
        data['to_rgb'] = self.to_rgb
        data['ori_shape'] = img.shape
        data['to_float32'] = self.to_float32
        
        # debug
        if self.debug:
            self._debug(data)

        return data

    def _debug(self, data: dict) -> None:
        img = data['img'].copy()

        if self.to_rgb:   # reverse
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.to_float32:
            img = img.astype(np.uint8)

        debug_dir = os.path.join(ROOT, 'debug', self.__class__.__name__)
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir, exist_ok=True)

        cv2.imwrite(os.path.join(debug_dir, os.path.basename(data['filename'])), img)

    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}('
        repr_str += f'to_rgb={self.to_rgb}, '
        repr_str += f'backend="{self.backend}", '
        repr_str += f'conf_file={self.conf_file}, '
        repr_str += f'to_float32={self.to_float32}, '
        repr_str += f'debug={self.debug}'
        repr_str += ')'
        return repr_str
    
    
@PREPROCESS.register_module()
class TLFTRLoadMultiViewImageFromFile(LoadImageFromFile):
    def _pseudo_img30(self, ori_img120: np.ndarray, ori_data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        def _simple_ious(crop_area, bboxes):
            crop_area = crop_area.reshape([1, 4])
            lt = np.maximum(crop_area[:, :2], bboxes[:, :2])
            rb = np.minimum(crop_area[:, 2:], bboxes[:, 2:])
            wh = (rb - lt).clip(min=0)
            inter_area = wh[:, 0] * wh[:, 1]
            return inter_area > 0
        
        # 30 -> 120的投影矩阵
        M = get_homography_matrix(
            src_intrinsic=ori_data['calibs']['intrinsic30'],
            src_extrinsic=ori_data['calibs']['extrinsic30'],
            dst_intrinsic=ori_data['calibs']['intrinsic120'],
            dst_extrinsic=ori_data['calibs']['extrinsic120'],
            src_car_center=ori_data['calibs']['car_center30'],
            dst_car_center=ori_data['calibs']['car_center120'])

        # 计算crop区域(将30图投影到120图像上的区域坐标)
        ori_h, ori_w = ori_img120.shape[:-1]
        area30 = cv2.perspectiveTransform(np.array([0, 0, ori_w, ori_h]).reshape([-1, 1, 2]).astype(np.float32), M).reshape([-1, 2]).reshape(-1)
        
        # pseudo img30
        pseudo_img30 = ori_img120[int(area30[1]): int(area30[3]), int(area30[0]): int(area30[2]), :].copy()

        # 缩放到与120图同样的尺寸
        pseudo_img30_h, pseudo_img30_w = pseudo_img30.shape[:-1]
        scale_h = ori_h / pseudo_img30_h
        scale_w = ori_w / pseudo_img30_w
        img30 = cv2.resize(pseudo_img30, (int(pseudo_img30_w * scale_w), int(pseudo_img30_h * scale_h)))

        # 获取30图像的bboxes & labels
        img30_bboxes = []
        img30_labels = []
        crop_w = area30[2] - area30[0]
        crop_h = area30[3] - area30[1]

        img120_bboxes = ori_data['gt_bboxes120'].copy()
        img120_labels = ori_data['gt_labels120'].copy()

        if len(img120_bboxes):
            keep = _simple_ious(area30, img120_bboxes)
            if sum(keep):
                img30_bboxes = img120_bboxes[keep]
                img30_labels = img120_labels[keep]
                img30_bboxes[:, [0, 2]] = (img30_bboxes[:, [0, 2]] - area30[0]).clip(min=0, max=crop_w) * scale_w
                img30_bboxes[:, [1, 3]] = (img30_bboxes[:, [1, 3]] - area30[1]).clip(min=0, max=crop_h) * scale_w

        if len(img30_bboxes):
            img30_bboxes = np.array(img30_bboxes, dtype=np.float32)
            img30_labels = np.array(img30_labels, dtype=np.int64)
        else:
            img30_bboxes = np.zeros((0, 4), dtype=np.float32)
            img30_labels = np.array([], dtype=np.int64)
        
        return img30, img30_bboxes, img30_labels
    
    def _pseudo_img120(self, ori_img30: np.ndarray, ori_data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        M = get_homography_matrix(
            src_intrinsic=ori_data['calibs']['intrinsic30'],
            src_extrinsic=ori_data['calibs']['extrinsic30'],
            dst_intrinsic=ori_data['calibs']['intrinsic120'],
            dst_extrinsic=ori_data['calibs']['extrinsic120'],
            src_car_center=ori_data['calibs']['car_center30'],
            dst_car_center=ori_data['calibs']['car_center120'])
        
        # pseudo img120
        ori_h, ori_w = ori_img30.shape[:-1]
        img120 = cv2.warpPerspective(ori_img30, M, (ori_w, ori_h))

        # 转换坐标
        img30_bboxes = ori_data['gt_bboxes30'].copy()
        img30_labels = ori_data['gt_labels30'].copy()
        if len(img30_bboxes):
            img30_points = deepcopy(img30_bboxes).reshape([-1, 1, 2]).astype(np.float32)
            img120_points = cv2.perspectiveTransform(img30_points, M).reshape([-1, 2])
            img120_bboxes = img120_points.reshape([-1, 4])
        else:
            img120_bboxes = img30_bboxes.copy()
        
        img120_labels = img30_labels.copy()

        return img120, img120_bboxes, img120_labels
    
    def __call__(self, data: dict) -> dict:
        filename = data.get('filename', None)
        if filename is None:
            return None
        
        filename30, filename120 = filename

        if (filename30 is None) and (filename120 is not None):  # 从120crop一部分当作30
            img120 = self._load_img(filename120)
            if img120 is None:
                return None
            
            img30, img30_bboxes, img30_labels = self._pseudo_img30(img120, data)
            data['gt_bboxes30'] = img30_bboxes
            data['gt_labels30'] = img30_labels
        elif (filename30 is not None) and (filename120 is None): # 将30投影到120平面作为120图
            img30 = self._load_img(filename30)
            if img30 is None:
                return None
            
            img120, img120_bboxes, img120_labels = self._pseudo_img120(img30, data)
            data['gt_bboxes120'] = img120_bboxes
            data['gt_labels120'] = img120_labels

            # fusion-instances是120平面的标注，需要更新
            data['gt_bboxes'] = img120_bboxes.copy()
            data['gt_labels'] = img120_labels.copy()
        else:
            img30 = self._load_img(filename30)
            img120 = self._load_img(filename120)
        
        # to rgb
        if self.to_rgb:
            img30 = cv2.cvtColor(img30, cv2.COLOR_BGR2RGB)
            img120 = cv2.cvtColor(img120, cv2.COLOR_BGR2RGB)
        
        # to float32
        if self.to_float32:
            img30 = img30.astype(np.float32)
            img120 = img120.astype(np.float32)

        data['img'] = [img30, img120]
        data['to_rgb'] = self.to_rgb
        data['ori_shape'] = [img30.shape, img120.shape]
        data['to_float32'] = self.to_float32
        
        # debug
        if self.debug:
            self._debug(data)

        return data
    
    def _debug(self, data: dict) -> None:
        imgs = data['img']

        for i, img in enumerate(imgs):
            if self.to_rgb:   # reverse
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if self.to_float32:
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
        if self.to_rgb:   # reverse
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.to_float32:
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