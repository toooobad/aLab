# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

from .utils import Registry


__all__ = [
    # engines
    'TESTER', 'TRAINER', 'DEPLOYER', 'EVALUATOR',

    # models
    'MODEL', 'BACKBONE', 'NECK', 'HEAD', 
    'POSTPROCESS',
    'LOSS',
    'TRANSFORMER',
    'POSITIONAL_ENCODING',
    'TRANSFORMER_ENCODER',
    'TRANSFORMER_DECODER',
    'ASSIGNER', 'MATCH_COST',

    # schedulers
    'LRSCHEDULER',

    # datasets
    'DATASET', 'PREPROCESS',

    # metric
    'METRIC'
    ]


# Engines
TESTER = Registry(name='TESTER')
TRAINER = Registry(name='TRAINER')
DEPLOYER = Registry(name='DEPLOYER')
EVALUATOR = Registry(name='EVALUATOR')

# Models
NECK = Registry(name='NECK')
HEAD = Registry(name='HEAD')
LOSS = Registry(name='LOSS')
MODEL = Registry(name='Models')
BACKBONE = Registry(name='BACKBONE')

POSTPROCESS = Registry(name='POSTPROCESS')

POSITIONAL_ENCODING = Registry(name='POSITIONAL_ENCODING')

TRANSFORMER = Registry(name='TRANSFORMER')
TRANSFORMER_ENCODER = Registry(name='TRANSFORMER_ENCODER')
TRANSFORMER_DECODER = Registry(name='TRANSFORMER_DECODER')

ASSIGNER = Registry(name='ASSIGNER')
MATCH_COST = Registry(name='MATCH_COST')

# LR schedulers
LRSCHEDULER = Registry(name='LRSCHEDULER')

# Dataset
DATASET = Registry(name='DATASET')
PREPROCESS = Registry(name='PREPROCESS')

# Metric
METRIC = Registry(name='METRIC')