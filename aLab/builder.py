# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import torch.nn as nn

from copy import deepcopy
from loguru import logger
from argparse import Namespace
from typing import (Any, Optional)

from .register import *
from .utils import (heading, parameter)


__all__ = ['build_model', 'build_optimizer', 'build_lr_scheduler']


def build_model(cfgs: Namespace) -> nn.Module:
    heading('model')
    model_cfgs = deepcopy(cfgs.model)
    model = MODEL.build(model_cfgs)
    model_strs = str(model).split('\n')
    for layer in model_strs:
        logger.info(layer)

    # params
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters = round(n_parameters / 1e6, 2)
    parameter('model parameters', f'{n_parameters}M')

    return model


def build_optimizer(model: nn.Module, cfgs: Namespace) -> torch.optim.Optimizer:
    heading('optimizer')
    parameter('type', 'AdamW')
    optimizer_cfg = deepcopy(cfgs.optimizer)
    for k, v in optimizer_cfg.items():
        parameter(k, v)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_cfg)
    
    return optimizer


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfgs: Namespace) -> torch.optim.lr_scheduler._LRScheduler:
    heading('lr scheduler')
    lr_scheduler_cfg = deepcopy(cfgs.lr_scheduler)
    parameter('type', lr_scheduler_cfg['type'])
    for k, v in lr_scheduler_cfg.items():
        if k != 'type':
            parameter(k, v)

    lr_scheduler_cfg['optimizer'] = optimizer
    return LRSCHEDULER.build(lr_scheduler_cfg)


def build_dataset(cfgs: Namespace, mode: str) -> torch.utils.data.Dataset:
    heading(f'{mode} dataset')
    if mode == 'train':
        dataset_cfg = deepcopy(cfgs.train_dataset)
    elif mode == 'val':
        dataset_cfg = deepcopy(cfgs.val_dataset)
    else:
        dataset_cfg = deepcopy(cfgs.test_dataset)

    parameter('type', dataset_cfg['type'])
    for k, v in dataset_cfg.items():
        if k == 'preprocesses':
            parameter(k, '')
            for prepcocess in v:
                logger.info(f'\t\t{prepcocess}')
        elif k != 'type':
            parameter(k, v)

    dataset = DATASET.build(dataset_cfg)
    parameter(f'{mode} samples', len(dataset))

    return dataset


def build_metric(cfgs: Namespace, prefix: Optional[str] = None) -> Any:
    if prefix is None:
        heading('metric')
    else:
        heading(f'{prefix} metric')

    metric_cfg = deepcopy(cfgs.metric)
    parameter('type', metric_cfg['type'])
    for k, v in metric_cfg.items():
        if k != 'type':
            parameter(k, v)

    return METRIC.build(metric_cfg)