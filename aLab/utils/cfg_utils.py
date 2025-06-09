# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import time
import yaml
import torch
import os.path as osp

from loguru import logger
from argparse import Namespace

from .logger_uils import heading
from .misc_utils import get_root_dir


__all__ = ['get_train_cfgs', 'get_test_cfgs', 'get_eval_cfgs', 'get_deploy_cfgs', 'merge_cfgs', 'dump_cfgs']


def load_yaml_file(yaml_file: str) -> dict:
    """
    加载yaml文件, 其中包含模型相关配置
    Retur: 
        yaml_args (dict): yaml文件中的配置
    """
    yaml_file = osp.abspath(osp.expanduser(yaml_file))

    with open(yaml_file, 'r', encoding='utf-8') as yaml_cfg:
        yaml_args = yaml.load(yaml_cfg.read(), Loader=yaml.FullLoader)  # type: dict
    
    return yaml_args


def get_train_cfgs(args: Namespace) -> Namespace:
    # step1. 从终端输入的参数 & 默认参数
    terminal_args = vars(args)  # type: dict

    # step2. 判断是否resume
    resume = terminal_args.get('resume', None)
    if resume is None:
        resume = False
        terminal_args['resume'] = False
    else:
        terminal_args['load_from'] = resume
        resume = True
        terminal_args['resume'] = True

    # step3. 判断是否定义了"config", 若没定义再判断是否定义了"load_from"
    yaml_file = terminal_args.get('config', None)
    load_from = terminal_args.get('load_from', None)
    assert (yaml_file is not None) or (load_from is not None), ValueError('"config" and "load_from" must define either one, but got "config=None", "load_from=None".')

    # step4. 获取训练配置
    if yaml_file is not None:
        cfgs = load_yaml_file(yaml_file)   # load from yaml
    else:
        checkpoints = torch.load(load_from, map_location='cpu')
        cfgs = checkpoints.get('cfgs', None)
        assert cfgs is not None, ValueError('cfgs in checkpoints is None and you need to define the config file.')

    # step5. work-dir
    startup_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
    cfgs['timestamp'] = startup_time

    cfgs['mode'] = 'train'

    if resume:
        assert (load_from is not None) and (osp.exists(load_from)), ValueError('"load_from" is None or not exists!')
        work_dir = osp.dirname(load_from).split('checkpoints')[0]
    else:
        work_dir = terminal_args.get('work_dir', None)
        if work_dir is None:
            work_dir = osp.join(get_root_dir(), 'work_dirs')

        model_cfg = cfgs.get('model', None)
        assert model_cfg is not None, ValueError('The model configuration is None.')
        model_name = model_cfg.get('type', None)
        assert model_name is not None, ValueError('')

        cfgs['model_name'] = model_name.lower()
        cfgs['experiment_name'] = osp.splitext(osp.basename(terminal_args['config']))[0]

        work_dir = osp.join(work_dir, 
                            terminal_args['task'], 
                            cfgs['model_name'], 
                            cfgs['experiment_name'],
                            'train',
                            startup_time)
        # work_dir = osp.join(work_dir, 'train', startup_time)

    cfgs['work_dir'] = work_dir

    return Namespace(**cfgs), Namespace(**terminal_args)


def get_test_cfgs(args: Namespace) -> Namespace:
    # step1. 从终端输入的参数 & 默认参数
    terminal_args = vars(args)  # type: dict

    # step2. 判断"onnx_model"和"torch_model"是否存在, 并获取cfgs
    onnx_model = terminal_args.get('onnx_model', None)
    torch_model = terminal_args.get('torch_model', None)
    infer_model = []

    cfgs = None
    if torch_model is not None:
        assert osp.exists(torch_model), ValueError('"torch_model" is None or not exists.')
        infer_model.append('torch')

        checkpoints = torch.load(torch_model, map_location='cpu')
        cfgs = checkpoints.get('cfgs', None)
    
    if onnx_model is not None:
        assert osp.exists(onnx_model), ValueError('"onnx_model" is None or not exists.')
        infer_model.append('onnx')

    if cfgs is None:
        assert terminal_args['config'] is not None, ValueError('cfgs not in checkpoints and "config" is None.')
        cfgs = load_yaml_file(terminal_args['config'])   # load from yaml

    cfgs['mode'] = 'test'
    cfgs['infer_model'] = infer_model

    # step3. work-dir
    startup_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
    cfgs['timestamp'] = startup_time

    work_dir = terminal_args['work_dir']
    if work_dir is None:
        work_dir = osp.join(get_root_dir(), 'work_dirs')

    experiment_name = cfgs.get('experiment_name', None)
    if experiment_name is None:
        experiment_name = osp.splitext(osp.basename(terminal_args['config']))[0]

    model_name = cfgs.get('model_name', None)
    if model_name is None:
        model_name = cfgs['model']['type'].lower()

    work_dir = osp.join(work_dir, terminal_args['task'], model_name, experiment_name, cfgs['mode'])
    work_dir = osp.join(work_dir, startup_time)
    cfgs['work_dir'] = work_dir

    return Namespace(**cfgs), Namespace(**terminal_args)


def get_deploy_cfgs(args: Namespace) -> Namespace:
    # step1. 从终端输入的参数 & 默认参数
    terminal_args = vars(args)  # type: dict

    # step2. 判断"load_from"是否存在
    load_from = terminal_args.get('load_from', None)
    assert (load_from is not None) and (osp.exists(load_from)), ValueError('"load_from" is None or not exists.')

    # step3. 获取训练参数
    checkpoints = torch.load(load_from, map_location='cpu')
    cfgs = checkpoints.get('cfgs', None)
    if cfgs is None:
        assert terminal_args['config'] is not None, ValueError('cfgs not in checkpoints and "config" is None.')
        cfgs = load_yaml_file(terminal_args['config'])   # load from yaml

    cfgs['mode'] = 'deploy'

    # step5. work-dir
    startup_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
    cfgs['timestamp'] = startup_time

    work_dir = terminal_args['work_dir']
    if work_dir is None:
        work_dir = osp.dirname(load_from)
    
    basename = osp.splitext(osp.basename(load_from))[0]
    work_dir = osp.join(work_dir, cfgs['mode'], basename, startup_time)
    cfgs['work_dir'] = work_dir

    return Namespace(**cfgs), Namespace(**terminal_args)


def get_eval_cfgs(args: Namespace) -> Namespace:
    # step1. 从终端输入的参数 & 默认参数
    terminal_args = vars(args)  # type: dict

    # step2. 判断"onnx_model"和"torch_model"是否存在, 并获取cfgs
    onnx_model = terminal_args.get('onnx_model', None)
    torch_model = terminal_args.get('torch_model', None)
    infer_model = []

    cfgs = None
    if torch_model is not None:
        assert osp.exists(torch_model), ValueError('"torch_model" is None or not exists.')
        infer_model.append('torch')

        checkpoints = torch.load(torch_model, map_location='cpu')
        cfgs = checkpoints.get('cfgs', None)
    
    if onnx_model is not None:
        assert osp.exists(onnx_model), ValueError('"onnx_model" is None or not exists.')
        infer_model.append('onnx')

    if cfgs is None:
        assert terminal_args['config'] is not None, ValueError('cfgs not in checkpoints and "config" is None.')
        cfgs = load_yaml_file(terminal_args['config'])   # load from yaml

    cfgs['mode'] = 'eval'
    cfgs['infer_model'] = infer_model

    # step3. work-dir
    startup_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
    cfgs['timestamp'] = startup_time

    work_dir = terminal_args['work_dir']
    if work_dir is None:
        work_dir = osp.join(get_root_dir(), 'work_dirs')

    experiment_name = cfgs.get('experiment_name', None)
    if experiment_name is None:
        experiment_name = osp.splitext(osp.basename(terminal_args['config']))[0]

    model_name = cfgs.get('model_name', None)
    if model_name is None:
        model_name = cfgs['model']['type'].lower()

    work_dir = osp.join(work_dir, terminal_args['task'], model_name, experiment_name, cfgs['mode'])
    work_dir = osp.join(work_dir, startup_time)
    cfgs['work_dir'] = work_dir

    return Namespace(**cfgs), Namespace(**terminal_args)


def merge_cfgs(dst_cfgs: Namespace, src_cfgs: Namespace, default_cfgs: dict) -> Namespace:
    heading('merge config')

    dst_cfgs = vars(dst_cfgs)
    src_cfgs = vars(src_cfgs)

    for src_key, src_value in src_cfgs.items():
        if src_key == 'work_dir':
            continue

        if src_key not in dst_cfgs:           # dst_cfgs没有的参数, 直接赋值
            dst_cfgs[src_key] = src_value
        else:
            dst_value = dst_cfgs[src_key]

            if src_key == 'auto_scale_lr':    # 如果dst_cfgs中有对应值, 或运算
                dst_cfgs[src_key] = dst_value | src_value
            else:                             # 如果dst_cfgs中有对应值, 并且与src_cfgs中不相等, 判断是否默认值, 不是默认值的话再赋值
                if (dst_value != src_value) and (src_value != default_cfgs[src_key]):
                    logger.info(f'The "{src_key}" origin is "{dst_value}" and will be set to "{src_value}".')
                    dst_cfgs[src_key] = src_value

    return Namespace(**dst_cfgs)


def dump_cfgs(cfgs: Namespace) -> None:
    heading('dump config')
    basename = osp.basename(cfgs.config)
    dump_file = osp.join(cfgs.work_dir, basename)
    with open(dump_file, 'w') as file:
        yaml.dump(vars(cfgs), file, default_flow_style=False)

    logger.info(f'Dump config to "{dump_file}"')
