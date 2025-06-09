# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import torch.nn as nn

from loguru import logger

from .logger_uils import parameter


__all__ = ['save_checkpoint', 'load_checkpoint']


def save_checkpoint(model_dict: dict, save_path: str) -> None:
    torch.save(model_dict, save_path)


def load_checkpoint(model: nn.Module, checkpoints: dict):
    # 模型权重
    model_weights = model.state_dict()

    # 预训练权重
    pretrained_weights = checkpoints.get('model', dict())

    unexpect_keys = []
    unmatched_keys = {}
    for pre_k, pre_v in pretrained_weights.items():
        if pre_k in model_weights:
            pre_shape = pre_v.shape
            model_shape = model_weights[pre_k].shape
            if pre_shape == model_shape:   # shape相同
                model_weights[pre_k] = pre_v
            else:
                unmatched_keys[pre_k] = (tuple(model_shape), tuple(pre_shape))    # shape不同
        else:
            unexpect_keys.append(pre_k)
    
    miss_keys = []
    for model_k in model_weights.keys():
        if model_k not in pretrained_weights:
            miss_keys.append(model_k)
    
    if len(miss_keys):
        parameter('miss weights', '')
        for miss in miss_keys:
            logger.info(miss)
    
    if len(unexpect_keys):
        parameter('unexpect', '')
        for unexpect in unexpect_keys:
            logger.info(unexpect)
    
    if len(unmatched_keys): # dict
        parameter('unmatched', '')
        for k, shapes in unmatched_keys.items():
            logger.info(f'{k} -> model{shapes[0]} vs ckpt{shapes[1]}')
    
    return model_weights