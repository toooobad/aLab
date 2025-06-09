# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import numpy as np
import os.path as osp

from loguru import logger
from functools import partial
from argparse import Namespace

from .logger_uils import heading


__all__ = ['multi_apply', 'get_root_dir', 'auto_scale_lr', 'cosine_similarity', 'get_homography_matrix']


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def get_root_dir() -> str:
    return osp.abspath(osp.expanduser(osp.join(osp.dirname(__file__), '../..')))


def auto_scale_lr(cfgs: Namespace) -> Namespace:
    heading('auto scale lr')
    if hasattr(cfgs, 'base_batch_size'):
        base_batch_size = cfgs.base_batch_size
    else:
        base_batch_size = cfgs.batch_size
    
    origin_lr = cfgs.optimizer['lr']
    scale_ratio = cfgs.batch_size / base_batch_size
    scaled_lr = origin_lr * scale_ratio
    cfgs.optimizer['lr'] = scaled_lr
    logger.info(f'Scale LR, from {origin_lr} to {scaled_lr}, with a ratio of {scale_ratio}')

    return cfgs


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    vector1 = vector1.reshape(-1)
    vector2 = vector2.reshape(-1)
    
    dot_product = np.dot(vector1, vector2)

    norm_vec1 = np.linalg.norm(vector1)
    norm_vec2 = np.linalg.norm(vector2)

    cosine_sim = dot_product / (norm_vec1 * norm_vec2)

    return cosine_sim


def get_matrix(intrinsic: np.ndarray, extrinsic: np.ndarray, car_center_to_ground: np.ndarray) -> np.ndarray:
    """
    单应矩阵(Homography-Matrix), 用于描述两个平面之间的投影变换关系, 是一个3x3的矩阵
    本函数用于计算单应矩阵，用于将图像平面上的点投影到一个与车辆相关的平面上(例如地面平面)

    intrinsic: 相机内参
    [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ]

    extrinsic: 相机外参, 将相机坐标系转到自车坐标系
    [
        [R, T],
        [0, 1]
    ]
    其中R是旋转矩阵, [3, 3]
    T是平移矩阵 [3, 1]

    car_center_to_ground: 用于将车辆中心转到地平面的矩阵
    """
    # step1. 提取R和T矩阵
    R = extrinsic[0:3, 0:3]  # 转转矩阵
    T = extrinsic[0:3, 3].reshape(3, 1)  # 平移矩阵

    # step2. 将相机坐标系的原点调整到车辆中心的高度
    height = car_center_to_ground[2, 3]
    T[2, 0] = T[2, 0] + height

    # step3. 定义平面法向量n & 偏移量shift
    n =  np.zeros((3, 1))
    n[2, 0] = -1.0  # 平面法向量，指向下方（假设平面为水平平面）

    shift = np.zeros((3, 1))
    shift[2, 0] = 1.0  # 偏移量，用于调整投影平面

    # step4. 转换法向量, 从世界坐标系转换到相机坐标系
    n = np.matmul(R.transpose(), n)

    # step5. 计算单应矩阵
    H = R + np.matmul((T + shift), n.transpose()) / T[2, 0]   # 构建一个投影矩阵
    H = np.matmul(H, np.linalg.inv(intrinsic))

    return np.ascontiguousarray(H)


def get_homography_matrix(src_intrinsic: np.ndarray, 
                          src_extrinsic: np.ndarray, 
                          dst_intrinsic: np.ndarray, 
                          dst_extrinsic: np.ndarray, 
                          src_car_center: np.ndarray, 
                          dst_car_center: np.ndarray) -> np.ndarray:
    """
    src_intrinsic: 原始相机的内参 [3, 3]
    dst_intrinsic: 目标相机的内参 [3, 3]
    src_extrinsic: 原始相机的外参 [4, 4]
    dst_extrinsic: 目标相机的外参 [4, 4]
    src_car_center: 未知 [4, 4]对角矩阵
    dst_car_center: 未知 [4, 4]对角矩阵
    """
    src_homography_matrix = get_matrix(src_intrinsic, src_extrinsic, src_car_center)
    dst_homography_matrix = get_matrix(dst_intrinsic, dst_extrinsic, dst_car_center)

    H_dst_src = np.matmul(np.linalg.inv(dst_homography_matrix), src_homography_matrix)
    M = H_dst_src / H_dst_src[2, 2]

    return np.ascontiguousarray(M)