# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import argparse

from loguru import logger

from aLab.register import TESTER
from aLab.utils import (get_test_cfgs, setup_logger, merge_cfgs, dump_cfgs, set_randomness)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', type=str, default=None, help='The path of the test sample, supporting images and folders containing images. Default: None')
    parser.add_argument('--input-w', type=int, default=-1, help='The input width of the sample during testing. -1 means using the settings in the config file. Default: -1')
    parser.add_argument('--input-h', type=int, default=-1, help='The input height of the sample during testing. -1 means using the settings in the config file. Default: -1')
    parser.add_argument('--save2video', action='store_true', help='Save the test results as a video file. Default: False')
    parser.add_argument('--save-preds', action='store_true', help='Save the prediction results of each sample. Default: False')
    parser.add_argument('--draw-gts', action='store_true', help='Draw GTs to sample. Default: False')

    parser.add_argument('--config', type=str, default=None, help='Training config file path. Default: None')
    parser.add_argument('--task', type=str, default='det2d', help='Perception task. Currently supported: [det2d]. Default: det2d')
    parser.add_argument('--work-dir', type=str, default=None, help='A working directory that holds the various logs and models generated at run time. Default: None')

    parser.add_argument('--onnx-model', type=str, default=None, help='ONNX model file.')
    parser.add_argument('--torch-model', type=str, default='/mnt/e/dl/code/aLab/work_dirs/det2d/ttfnet/ttfnet_base/train/2025-06-07-15-02/checkpoints/latest.pth', help='Pre-trained model weights file. Default: None')
    
    parser.add_argument('--num-workers', default=4, type=int, help='The number of subprocesses used during data loading. Default: 4')
    parser.add_argument('--log-interval', default=10, type=int, help='The interval for printing logs during training. Default: 10')

    # training parameters
    parser.add_argument('--seed', default=42, type=int, help='Random number seed to keep the randomness consistent between runs. Default: 42')

    # 获取默认值, 用来判断是否被修改
    default_cfgs = dict()
    for action in parser._actions:
        default_cfgs[action.dest] = action.default

    return default_cfgs, parser.parse_args()


@logger.catch
def main(default_cfgs: dict, cfgs: argparse.Namespace):
    # step1. test cfgs
    cfgs, args = get_test_cfgs(cfgs)

    # step2. setup logger
    setup_logger(cfgs.work_dir, filename=f'{cfgs.timestamp}.log', mode='a')

    # step3. merge cfgs and args
    cfgs = merge_cfgs(cfgs, args, default_cfgs)

    # step4. dump cfgs
    dump_cfgs(cfgs)

    # step5. fix the seed for reproducibility
    set_randomness(cfgs.seed)
    
    # step7. test
    tester_cfg = dict(type=cfgs.task, cfgs=cfgs)
    tester = TESTER.build(tester_cfg)
    tester.test()


if __name__ == '__main__':
    default_cfgs, cfgs = parse_args()
    
    main(default_cfgs, cfgs)