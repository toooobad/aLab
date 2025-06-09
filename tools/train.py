# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import argparse

from loguru import logger

from aLab.register import TRAINER
from aLab.utils import (get_train_cfgs, setup_logger, auto_scale_lr, merge_cfgs, dump_cfgs, set_randomness)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/mnt/e/dl/code/aLab/configs/det2d/ttfnet/ttfnet_base.yaml', help='Training config file path. Default: None')
    parser.add_argument('--task', type=str, default='det2d', help='Perception task. Currently supported: [det2d]. Default: det2d')
    parser.add_argument('--work-dir', type=str, default=None, help='A working directory that holds the various logs and models generated at run time. Default: None')

    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint. Default: None')
    parser.add_argument('--load-from', type=str, default=None, help='Pre-trained model weights file. Default: None')
    
    parser.add_argument('--num-workers', default=4, type=int, help='The number of subprocesses used during data loading. Default: 4')
    parser.add_argument('--log-interval', default=10, type=int, help='The interval for printing logs during training. Default: 10')

    # training parameters
    parser.add_argument('--seed', default=42, type=int, help='Random number seed to keep the randomness consistent between runs. Default: 42')
    parser.add_argument('--max-norm', default=-1, type=float, help='Gradient clipping max norm, -1 means no gradient clipping. Default: -1')
    parser.add_argument('--batch-size', type=int, default=-1, help='The number of samples per iteration, when the value is -1, is set in config. Default: -1')
    parser.add_argument('--auto-scale-lr', action='store_true', help='Auto scale learning rate, Default: False')
    
    # 获取默认值, 用来判断是否被修改
    default_cfgs = dict()
    for action in parser._actions:
        default_cfgs[action.dest] = action.default

    return default_cfgs, parser.parse_args()


@logger.catch
def main(default_cfgs: dict, cfgs: argparse.Namespace):
    # step1. train cfgs
    cfgs, args = get_train_cfgs(cfgs)

    # step2. setup logger
    setup_logger(cfgs.work_dir, filename=f'{cfgs.timestamp}.log', mode='a')

    # step3. merge cfgs and args
    cfgs = merge_cfgs(cfgs, args, default_cfgs)

    # step4. dump cfgs
    dump_cfgs(cfgs)

    # step5. auto scale-lr
    if cfgs.auto_scale_lr:
        cfgs = auto_scale_lr(cfgs)

    # step6. fix the seed for reproducibility
    set_randomness(cfgs.seed)
    
    # step7. train
    trainer_cfg = dict(type=cfgs.task, cfgs=cfgs)
    trainer = TRAINER.build(trainer_cfg)
    trainer.train()


if __name__ == '__main__':
    default_cfgs, cfgs = parse_args()
    
    main(default_cfgs, cfgs)